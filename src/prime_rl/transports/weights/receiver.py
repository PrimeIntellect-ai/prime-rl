"""Consumer-side counterpart of ``WeightBroadcast``.

A ``WeightBroadcastReceiver`` discovers new policy versions from the
transport's sentinels and moves the inference engines onto them. It is the
single owner of the per-transport consumer protocol — the weight watcher, the
orchestrator's startup rendezvous, and the evals process all drive the same
object instead of hand-rolling transport branches.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from httpx import AsyncClient
from modelexpress import p2p_pb2
from modelexpress.client import MxClient

from prime_rl.configs.trainer import WeightBroadcastConfig
from prime_rl.transports.weights.base import FINISHED_MARKER, RECEIVER_READY_MARKER, STARTED_MARKER
from prime_rl.transports.weights.nixl.model_express import ModelExpressSession
from prime_rl.utils.client import (
    init_nccl_broadcast,
    init_nixl_broadcast,
    load_lora_adapter,
    update_weights,
)
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_all_ckpt_steps, get_step_path, wait_for_path


class WeightBroadcastReceiver(ABC):
    """Moves the inference engines onto trainer-published policy versions."""

    # The marker that announces a new version to this receiver: a finished
    # broadcast on disk for filesystem, an in-flight one for live transports
    # (the trainer is already blocked waiting for the receiver).
    discovery_marker: ClassVar[str] = FINISHED_MARKER
    # Whether the consumer may skip versions. A live transport strands the
    # trainer inside the transfer when a version is never received.
    can_skip_versions: ClassVar[bool] = True

    def __init__(
        self,
        broadcast_dir: Path,
        config: WeightBroadcastConfig,
        admin_clients: list[AsyncClient],
        model_name: str,
        max_version: int | None = None,
    ) -> None:
        self.logger = get_logger()
        self.broadcast_dir = broadcast_dir
        self.config = config
        self.admin_clients = admin_clients
        self.model_name = model_name
        self.max_version = max_version

    async def initialize(self) -> None:
        """One-time transport bootstrap (rendezvous groups, sessions)."""

    def step_dir(self, step: int) -> Path:
        return get_step_path(self.broadcast_dir, step)

    def is_published(self, step: int) -> bool:
        """Whether the trainer has announced v{step} (see ``discovery_marker``)."""
        return (self.step_dir(step) / self.discovery_marker).exists()

    def next_version(self, current: int) -> int:
        """Newest version announced beyond ``current``; ``current`` if none."""
        published = [step for step in get_all_ckpt_steps(self.broadcast_dir) if self.is_published(step)]
        return max(published, default=current)

    @abstractmethod
    async def wait_published(self, step: int, cancelled: Callable[[], bool] | None = None) -> None:
        """Block until v{step} is committed by the trainer. Runs before the
        orchestrator advances ``policy.version`` — the version must never move
        ahead of a confirmed publish."""

    @abstractmethod
    async def receive(self, step: int) -> None:
        """Move the engines onto the published v{step}."""

    async def sync_startup(self, step: int, timeout: float) -> None:
        """Rendezvous with the trainer's startup broadcast of v{step}."""
        await asyncio.wait_for(self.wait_published(step), timeout=timeout)
        await self.receive(step)


class FileSystemReceiver(WeightBroadcastReceiver):
    """Loads finished broadcasts from the shared filesystem. An adapter
    broadcast (PEFT dir) is hot-swapped under live traffic — an in-place
    adapter reload is a vLLM-native op that needs no engine pause; a full
    checkpoint pauses the engines for the load."""

    discovery_marker = FINISHED_MARKER
    can_skip_versions = True

    async def wait_published(self, step: int, cancelled: Callable[[], bool] | None = None) -> None:
        finished = self.step_dir(step) / FINISHED_MARKER
        if not finished.exists():
            self.logger.info(
                f"Orchestrator paused: waiting for trainer to broadcast checkpoint {step}. "
                "Training is progressing normally."
            )
            await wait_for_path(finished)

    async def receive(self, step: int) -> None:
        weights_dir = self.step_dir(step)
        if (weights_dir / "adapter_config.json").exists():
            await load_lora_adapter(self.admin_clients, self.model_name, weights_dir)
        else:
            await update_weights(self.admin_clients, weights_dir, step=step)


class NCCLReceiver(WeightBroadcastReceiver):
    """Joins the trainer's NCCL collective. The trainer raises ``.started``
    and blocks; the receiver pauses the engines, raises ``.receiver_ready``, and sends
    them into the receive RPC — only then does the trainer enter the
    collective, so the handshake can never race a stale marker."""

    discovery_marker = STARTED_MARKER
    can_skip_versions = False

    async def initialize(self) -> None:
        await init_nccl_broadcast(
            self.admin_clients,
            self.config.host,
            self.config.port,
            self.config.timeout,
            inference_world_size=self.config.inference_world_size,
            quantize_in_weight_transfer=self.config.quantize_in_weight_transfer,
        )

    async def wait_published(self, step: int, cancelled: Callable[[], bool] | None = None) -> None:
        started = self.step_dir(step) / STARTED_MARKER
        if not started.exists():
            await wait_for_path(started)

    async def receive(self, step: int) -> None:
        step_dir = self.step_dir(step)
        await update_weights(
            self.admin_clients,
            step_dir,
            step=step,
            on_paused=lambda: (step_dir / RECEIVER_READY_MARKER).touch(),
        )


class NIXLReceiver(WeightBroadcastReceiver):
    """Drives the orchestrator's side of the ModelExpress rendezvous. Statuses
    are unversioned, so versions are counted: one READY/INITIALIZING cycle per
    policy version, capped at ``max_version`` (the trainer's final broadcast)."""

    discovery_marker = STARTED_MARKER
    can_skip_versions = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.session = ModelExpressSession(
            client=MxClient(server_url=f"{self.config.host}:{self.config.port}"),
            role="orchestrator",
            rank=0,
            session_id=self.config.session_id,
            worker_id="orchestrator",
        )

    async def initialize(self) -> None:
        await init_nixl_broadcast(
            self.admin_clients,
            self.config.host,
            self.config.port,
            self.config.timeout,
            self.config.inference_world_size,
            self.config.session_id,
        )
        self.session.publish()
        await self.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)

    async def set_status(self, status: int) -> None:
        await asyncio.to_thread(self.session.set_status, status)

    def next_version(self, current: int) -> int:
        if self.max_version is not None and current >= self.max_version:
            return current
        return current + 1

    async def wait_published(self, step: int, cancelled: Callable[[], bool] | None = None) -> None:
        """The trainer flips to READY when it starts broadcasting the next version."""
        while cancelled is None or not cancelled():
            found = await asyncio.to_thread(
                self.session.exists_role_with_status, "trainer", p2p_pb2.SOURCE_STATUS_READY
            )
            if found:
                return
            await asyncio.sleep(1.0)
        raise asyncio.CancelledError

    async def receive(self, step: int) -> None:
        # ACK the trainer (it waits for the orchestrator's READY before the
        # engines' INITIALIZING), run the transfer, then close the cycle.
        await self.set_status(p2p_pb2.SOURCE_STATUS_READY)
        await update_weights(self.admin_clients, None, step=step)
        await self.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
        await asyncio.to_thread(
            self.session.wait_for,
            "trainer",
            count=1,
            status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
            timeout=self.config.timeout,
        )

    async def sync_startup(self, step: int, timeout: float) -> None:
        # The trainer's startup broadcast sets READY before waiting for the
        # orchestrator, so receive() alone completes the rendezvous.
        await self.receive(step)


def setup_weight_receiver(
    broadcast_dir: Path,
    config: WeightBroadcastConfig,
    admin_clients: list[AsyncClient],
    model_name: str,
    max_version: int | None = None,
) -> WeightBroadcastReceiver:
    receivers = {"filesystem": FileSystemReceiver, "nccl": NCCLReceiver, "nixl": NIXLReceiver}
    if config.type not in receivers:
        raise ValueError(f"Invalid weight broadcast type: {config.type}")
    return receivers[config.type](broadcast_dir, config, admin_clients, model_name, max_version)
