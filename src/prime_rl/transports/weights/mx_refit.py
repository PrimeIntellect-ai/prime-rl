"""Publish FSDP weights through ModelExpress's slice-reshard transport.

Every trainer rank hands its FSDP state_dict to MX's trainer client, which stages
the rank-local shards and advertises them under a per-step WeightVersion. Rank 0
creates the version and every rank publishes its shard; both sides name it
``{run_uid}:{step}``, so a stale version from an earlier run against a long-lived
MX server can never be mistaken for this run's.

The version lifecycle nests inside the shared sentinel handshake that
``WeightSender.broadcast`` runs for every transport, and the two are not
redundant: the markers say *which step* is being offered, the version state says
*whether every rank has published into it*. The sender only reaches
``_broadcast`` once the receiver has acknowledged the step, and it blocks there
until the receiver has retired the version, so training cannot overwrite staging
buffers that a generator is still reading.
"""

from __future__ import annotations

import asyncio
import os
import time
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
from prime_rl.transports.weights.base import WeightReceiver, WeightSender

# How often rank 0 checks whether the generator has started pulling.
RELEASE_POLL_INTERVAL = 0.05

# How often the receiver checks whether every trainer rank has published into
# the offered version. Deliberately short: this is a cheap status RPC on a
# version whose name this side already knows, and the trainer is blocked inside
# its broadcast for the whole wait, so time asleep here is time the trainer
# spends idle. Measured at a 1.0s interval this cost 1.3-2.4s per step against a
# wire transfer of a few seconds, which made the handshake a significant share
# of what looked like transfer time.
READY_POLL_INTERVAL = 0.1


def weight_version_uid(run_uid: str, step: int) -> str:
    """Return the WeightVersion uid the trainer publishes for ``step``.

    MX lets the caller choose a version's identity and both sides derive it from
    configuration, so the identity never has to travel between them. Naming
    versions after the step also makes a stalled refit legible on the server:
    the live uid says which step is in flight.
    """
    return f"{run_uid}:{step}"


def version_missing(error: grpc.RpcError) -> bool:
    """Whether ``error`` means the trainer has not created the version yet."""
    return error.code() is grpc.StatusCode.NOT_FOUND


async def resolve_ready_version(
    control: ModelExpressControlClient,
    uid: str,
    poll_interval: float = READY_POLL_INTERVAL,
    stopped: asyncio.Event | None = None,
) -> str:
    """Wait until ``uid`` exists and every trainer rank has published into it.

    NOT_FOUND is a wait rather than an error: this side acknowledges a step
    before the trainer creates its version, so the absence of the version is the
    expected initial state rather than a failure. ``stopped``, when given, lets
    the caller cancel the poll.
    """
    while stopped is None or not stopped.is_set():
        try:
            version = await asyncio.to_thread(control.get_weight_version, uid)
        except grpc.RpcError as error:
            if not version_missing(error):
                raise
        else:
            if version.state is WeightVersionState.READY:
                return uid
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

    @property
    def server_url(self) -> str:
        return f"{self.config.host}:{self.config.port}"

    def _initialize(self, model: nn.Module) -> None:
        # The MX client factory picks its trainer adapter from the environment,
        # so the FSDP adapter has to be selected before the client is built.
        os.environ["MX_TRAINER_ENGINE"] = "FSDP"
        # Publishers are deliberately left unpinned to a single RDMA rail (unlike
        # receivers, see mx_rdma.apply_rdma_defaults): every receiver needs shards
        # from every rank, so on a fabric with isolated rail subnets a single-rail
        # publisher would be unreachable from receivers pinned to other rails.
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

        # Collect every rank's actual source slot so rank 0 can pass the complete
        # expected_source_slots to create_weight_version. Reading the real slots
        # avoids hardcoding the slot-naming convention; set() stays safe if DP
        # replicas ever share a logical slot.
        gathered: list[str] = [""] * self.world.world_size
        dist.all_gather_object(gathered, slot)
        self._expected_slots = sorted(set(gathered))
        self._initialized = True

    @torch.no_grad()
    def _broadcast(self, model: nn.Module, step: int, step_dir: Path) -> None:
        del step_dir  # mx_refit addresses versions by uid, not by path
        if not self._initialized:
            self._initialize(model)
        assert self._client is not None

        # Every rank derives the uid, so there is nothing to broadcast. The
        # collective still has to stay, as a barrier: it is what guarantees rank 0
        # has created the version before any rank publishes a shard into it.
        uid = weight_version_uid(self.config.run_uid, step)
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

        # Block until the generator has pulled (the receiver retired the version,
        # moving it to RELEASING) so training does not overwrite the staging
        # arenas mid-read. Rank 0 observes RELEASING; the barrier propagates that
        # to every rank before any of them withdraw their shard.
        if self.world.is_master:
            self._wait_released(uid)
        dist.barrier()
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
                raise TimeoutError(f"No generator pulled version {uid} within {self.timeout}s (state={state})")
            time.sleep(RELEASE_POLL_INTERVAL)


class MXRefitWeightReceiver(WeightReceiver):
    """Moves the engines onto a trainer-offered ModelExpress version.

    Version detection rides on the shared ``.sender_ready`` markers inherited
    from ``WeightReceiver``, so this side needs no server-side scan to learn
    which step is on offer -- only a status check on the one version it is
    already waiting for.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._control: ModelExpressControlClient | None = None

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
        # Acknowledge first, then wait for the version. The sender is held at the
        # marker handshake until this ack lands and only creates the version
        # afterwards, so waiting for the version before acknowledging would
        # deadlock the pair.
        self._ack(step)
        uid = await resolve_ready_version(
            self._control,
            weight_version_uid(self.config.run_uid, step),
        )
        await self.admin_plane.update_weights(
            None,
            transport="mx_refit",
            step=step,
            version_uid=uid,
        )
        # Retiring the version moves it to RELEASING, which is what unblocks the
        # trainer's broadcast.
        await asyncio.to_thread(self._control.delete_weight_version, uid)
