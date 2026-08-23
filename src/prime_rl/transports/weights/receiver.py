"""Consumer-side counterpart of ``WeightBroadcast``.

A ``WeightBroadcastReceiver`` discovers offered policy versions from the
shared sentinels and moves the inference engines onto them. It is the single
owner of the per-transport consumer protocol — the weight watcher, the
orchestrator's startup rendezvous, and the evals process all drive the same
object instead of hand-rolling transport branches. Every transport runs the
same handshake: the trainer offers a version (``.sender_ready``) and blocks
until the receiver acknowledges (``.receiver_ready``), so every offered
version must be received.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

from httpx import AsyncClient
from modelexpress import p2p_pb2
from modelexpress.client import MxClient

from prime_rl.configs.trainer import WeightBroadcastConfig
from prime_rl.orchestrator.clients import (
    init_nccl_broadcast,
    init_nixl_broadcast,
    load_lora_adapter,
    update_weights,
)
from prime_rl.transports.weights.base import FINISHED_MARKER, RECEIVER_READY_MARKER, SENDER_READY_MARKER
from prime_rl.transports.weights.nixl.model_express import ModelExpressSession
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_all_ckpt_steps, get_step_path, wait_for_path


class WeightBroadcastReceiver(ABC):
    """Moves the inference engines onto trainer-offered policy versions."""

    def __init__(
        self,
        broadcast_dir: Path,
        config: WeightBroadcastConfig,
        admin_clients: list[AsyncClient],
        model_name: str,
    ) -> None:
        self.logger = get_logger()
        self.broadcast_dir = broadcast_dir
        self.config = config
        self.admin_clients = admin_clients
        self.model_name = model_name

    async def initialize(self) -> None:
        """One-time transport bootstrap (rendezvous groups, sessions)."""

    def step_dir(self, step: int) -> Path:
        return get_step_path(self.broadcast_dir, step)

    def is_published(self, step: int) -> bool:
        """Whether the trainer has offered v{step}."""
        return (self.step_dir(step) / SENDER_READY_MARKER).exists()

    def next_version(self, current: int) -> int:
        """Newest version offered beyond ``current``; ``current`` if none."""
        published = [step for step in get_all_ckpt_steps(self.broadcast_dir) if self.is_published(step)]
        return max(published, default=current)

    async def wait_published(self, step: int, cancelled: Callable[[], bool] | None = None) -> None:
        """Block until the trainer offers v{step}. Runs before the orchestrator
        advances ``policy.version`` — the version must never move ahead of a
        confirmed offer."""
        sender_ready = self.step_dir(step) / SENDER_READY_MARKER
        while not sender_ready.exists():
            if cancelled is not None and cancelled():
                raise asyncio.CancelledError
            await asyncio.sleep(0.2)

    def _ack(self, step: int) -> None:
        """Acknowledge the offered version — unblocks the waiting trainer."""
        (self.step_dir(step) / RECEIVER_READY_MARKER).touch()

    @abstractmethod
    async def receive(self, step: int) -> None:
        """Acknowledge the offered v{step} and move the engines onto it."""

    async def sync_startup(self, step: int, timeout: float) -> None:
        """Rendezvous with the trainer's startup broadcast of v{step}."""
        await asyncio.wait_for(self.wait_published(step), timeout=timeout)
        await self.receive(step)


class FileSystemReceiver(WeightBroadcastReceiver):
    """Loads broadcasts from the shared filesystem. The acknowledgement lets
    the trainer start writing; the engines are only touched once the weights
    are fully on disk. An adapter broadcast (PEFT dir) is hot-swapped under
    live traffic — an in-place adapter reload is a vLLM-native op that needs
    no engine pause; a full checkpoint pauses the engines for the load."""

    async def receive(self, step: int) -> None:
        weights_dir = self.step_dir(step)
        self._ack(step)
        await wait_for_path(weights_dir / FINISHED_MARKER)
        if (weights_dir / "adapter_config.json").exists():
            await load_lora_adapter(self.admin_clients, self.model_name, weights_dir)
        else:
            await update_weights(self.admin_clients, weights_dir, step=step)


class NCCLReceiver(WeightBroadcastReceiver):
    """Joins the trainer's NCCL collective. The receiver pauses the engines,
    acknowledges, and sends them into the receive RPC — only then does the
    trainer enter the collective, so the handshake can never race a stale
    marker."""

    async def initialize(self) -> None:
        await init_nccl_broadcast(
            self.admin_clients,
            self.config.host,
            self.config.port,
            self.config.timeout,
            inference_world_size=self.config.inference_world_size,
            quantize_in_weight_transfer=self.config.quantize_in_weight_transfer,
        )

    async def receive(self, step: int) -> None:
        await update_weights(
            self.admin_clients,
            self.step_dir(step),
            step=step,
            on_paused=lambda: self._ack(step),
        )


class NIXLReceiver(WeightBroadcastReceiver):
    """Drives the orchestrator's side of the ModelExpress rendezvous. Version
    discovery runs on the shared sentinels; the unversioned ModelExpress
    statuses only choreograph the transfer itself."""

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

    async def receive(self, step: int) -> None:
        # ACK the trainer (it waits for the orchestrator's READY before the
        # engines' INITIALIZING), run the transfer, then close the cycle.
        self._ack(step)
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


def setup_weight_receiver(
    broadcast_dir: Path,
    config: WeightBroadcastConfig,
    admin_clients: list[AsyncClient],
    model_name: str,
) -> WeightBroadcastReceiver:
    receivers = {"filesystem": FileSystemReceiver, "nccl": NCCLReceiver, "nixl": NIXLReceiver}
    if config.type not in receivers:
        raise ValueError(f"Invalid weight broadcast type: {config.type}")
    return receivers[config.type](broadcast_dir, config, admin_clients, model_name)
