"""ModelExpress weight broadcast for FSDP trainers."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from pathlib import Path

import grpc
import torch
import torch.distributed as dist
import torch.nn as nn
from modelexpress_rl import (
    FSDPTrainerContext,
    ModelExpressControlClient,
    ModelExpressTrainerClient,
    ModelExpressTrainerConfig,
    TrainerStagingMode,
    WeightPayloadFormat,
    WeightVersionRef,
    WeightVersionState,
)

from prime_rl.configs.trainer import MXRefitWeightBroadcastConfig
from prime_rl.orchestrator.clients import init_mx_refit_broadcast
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.transports.weights.base import SENDER_READY_MARKER, WeightReceiver, WeightSender
from prime_rl.transports.weights.mx_phases import PhaseTimer, timed_refit

RELEASE_POLL_INTERVAL = 0.05
READY_POLL_INTERVAL = 0.1


def weight_version_uid(offer_token: str, step: int) -> str:
    """Return the ModelExpress version ID for an offered step."""
    return f"{offer_token}:{step}"


def version_missing(error: grpc.RpcError) -> bool:
    """Whether ``error`` means the trainer has not created the version yet."""
    return error.code() is grpc.StatusCode.NOT_FOUND


async def resolve_ready_version(
    control: ModelExpressControlClient,
    uid: str,
    timeout: float,
    poll_interval: float = READY_POLL_INTERVAL,
    stopped: asyncio.Event | None = None,
) -> str:
    """Wait until all trainer ranks have published ``uid``."""
    deadline = time.monotonic() + timeout
    while stopped is None or not stopped.is_set():
        try:
            version = await asyncio.to_thread(control.get_weight_version, uid)
        except grpc.RpcError as error:
            if not version_missing(error):
                raise
        else:
            if version.state is WeightVersionState.READY:
                return uid
        if time.monotonic() > deadline:
            raise TimeoutError(f"No trainer published weight version {uid} within {timeout}s")
        await asyncio.sleep(poll_interval)
    raise asyncio.CancelledError


class MXRefitWeightSender(WeightSender):
    """Stages every rank's FSDP shards into a per-step ModelExpress version."""

    def __init__(
        self,
        output_dir: Path,
        config: MXRefitWeightBroadcastConfig,
        parallel_dims: ParallelDims,
        model_name: str,
    ) -> None:
        super().__init__(output_dir, config.timeout)
        self.config = config
        self.parallel_dims = parallel_dims
        self.model_name = model_name
        self._initialized = False
        self._client: ModelExpressTrainerClient | None = None
        self._control: ModelExpressControlClient | None = None
        self._expected_slots: list[str] = []
        self._offer_token: str | None = None

    @property
    def server_url(self) -> str:
        return f"{self.config.host}:{self.config.port}"

    def _initialize(self, model: nn.Module) -> None:
        # Publishers stay unpinned because every receiver reads from every rank.
        self._client = ModelExpressTrainerClient.initialize(
            ModelExpressTrainerConfig(
                engine_context=FSDPTrainerContext(),
                model_name=self.model_name,
                device_id=self.world.local_rank,
                server_url=self.server_url,
                staging_mode=TrainerStagingMode.COPY_TO_DEVICE,
                payload_format=WeightPayloadFormat.FULL_TENSOR,
            )
        )
        slot = self._client.bind_tensors(model.state_dict())
        if self.world.is_master:
            self._control = ModelExpressControlClient.connect(server_url=self.server_url)

        gathered: list[str] = [""] * self.world.world_size
        dist.all_gather_object(gathered, slot)
        self._expected_slots = sorted(set(gathered))
        self._initialized = True

    def _offer(self, step_dir: Path) -> None:
        """Publish a unique offer token atomically."""
        self._offer_token = f"{self.config.run_uid}.{uuid.uuid4().hex[:8]}"
        staged = step_dir / f"{SENDER_READY_MARKER}.offer"
        staged.write_text(f"{self._offer_token}\n")
        os.replace(staged, step_dir / SENDER_READY_MARKER)

    def _wait_for_receiver_ready(self, step_dir: Path) -> None:
        # ModelExpress stages versions asynchronously. _wait_released provides
        # the stronger guarantee that a generator consumed the version.
        del step_dir

    @torch.no_grad()
    def _broadcast(self, model: nn.Module, step: int, step_dir: Path) -> None:
        del step_dir  # mx_refit addresses versions by uid, not by path
        with timed_refit("trainer", step, weight_version_uid(self._offer_token or "", step)) as timer:
            with timer.phase("handshake"):
                if self.world.world_size > 1:
                    offered = [self._offer_token]
                    dist.broadcast_object_list(offered, src=0)
                    self._offer_token = offered[0]
            if self._offer_token is None:
                raise RuntimeError("mx_refit broadcast reached publication without an offer token")
            uid = weight_version_uid(self._offer_token, step)
            timer.identify(uid)

            if not self._initialized:
                with timer.phase("init"):
                    self._initialize(model)
            assert self._client is not None

            with timer.phase("publish"):
                if self.world.is_master:
                    assert self._control is not None
                    self._control.create_weight_version(
                        model_name=self.model_name,
                        idempotency_key=uid,
                        payload_format=WeightPayloadFormat.FULL_TENSOR,
                        expected_source_slots=self._expected_slots,
                        uid=uid,
                    )
                dist.barrier()
                self._client.publish_version(version=WeightVersionRef(uid))
                for name, value in self._client.pop_metrics().items():
                    timer.mark(name, float(value))

            with timer.phase("rendezvous"):
                if self.world.is_master:
                    self._wait_released(uid)
                dist.barrier()

            with timer.phase("release"):
                self._client.release_version(version=WeightVersionRef(uid))

    def _wait_released(self, uid: str) -> None:
        assert self._control is not None
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                state = self._control.get_weight_version(uid).state
            except grpc.RpcError as error:
                if version_missing(error):
                    return  # already retired == generator done
                raise
            if state is WeightVersionState.RELEASING:
                return
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"No generator pulled version {uid} within {self.timeout}s (state={state}). "
                    + (
                        "The consumer is not looking for this step; if only one side restarted, "
                        "restart both so they resync to the same step."
                        if state is WeightVersionState.READY
                        else "Not every trainer rank published its shard."
                    )
                )
            time.sleep(RELEASE_POLL_INTERVAL)


class MXRefitWeightReceiver(WeightReceiver):
    """Install trainer-published ModelExpress versions on inference workers."""

    _control: ModelExpressControlClient | None = None

    async def initialize(self) -> None:
        await init_mx_refit_broadcast(
            self.admin_plane,
            self.config.host,
            self.config.port,
            self.config.timeout,
        )
        self._control = ModelExpressControlClient.connect(server_url=f"{self.config.host}:{self.config.port}")

    async def receive(self, step: int) -> None:
        assert self._control is not None
        with timed_refit("orchestrator", step, "") as timer:
            self._mark_offer_lag(step, timer)
            self._ack(step)
            with timer.phase("discovery"):
                uid = weight_version_uid(await self._read_offer_token(step), step)
                timer.identify(uid)
                await resolve_ready_version(self._control, uid, timeout=self.config.timeout)
            try:
                with timer.phase("update_rpc"):
                    await self.admin_plane.update_weights(
                        None,
                        transport="mx_refit",
                        step=step,
                        version_uid=uid,
                    )
            finally:
                # Retiring releases the trainer even when installation fails.
                with timer.phase("retire"):
                    await asyncio.shield(asyncio.to_thread(self._control.delete_weight_version, uid))

    async def _read_offer_token(self, step: int) -> str:
        """Read the unique token from the trainer's offer marker."""
        marker = self.step_dir(step) / SENDER_READY_MARKER
        token = await asyncio.to_thread(marker.read_text)
        return token.strip()

    def _mark_offer_lag(self, step: int, timer: PhaseTimer) -> None:
        """Record the delay between publishing and observing the offer marker."""
        try:
            offered = (self.step_dir(step) / SENDER_READY_MARKER).stat().st_mtime
        except OSError:
            return
        timer.mark("offer_lag_s", max(0.0, time.time() - offered))
