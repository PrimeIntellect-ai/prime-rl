"""Publish FSDP weights through ModelExpress's slice-reshard transport (Model B).

Every trainer rank hands its FSDP state_dict to MX's trainer client, which stages
the rank-local shards and advertises them under a per-step WeightVersion. This
rank owns the version lifecycle: rank 0 creates the version, every rank publishes
its shard, and the version id is handed to the orchestrator through a
broadcast-dir marker. ``broadcast_weights`` blocks until the generator has pulled
(the orchestrator moves the version to RELEASING) before letting training reuse
its staging buffers.
"""

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
    ModelExpressControlClient,
    ModelExpressTrainerClient,
    ModelExpressTrainerConfig,
    TrainerStagingMode,
    WeightPayloadFormat,
    WeightVersionRef,
    WeightVersionState,
)

from prime_rl.configs.trainer import MXRefitWeightBroadcastConfig
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.world import get_world
from prime_rl.transports.weights.base import WeightBroadcast
from prime_rl.utils.pathing import get_broadcast_dir, get_step_path, wait_for_path

# TODO: remove the filesystem uid marker (write here + reads in the orchestrator
# watcher) once MX exposes an API to query the latest version_id/version_number the
# trainer published, so the trainer->orchestrator handoff no longer needs a shared PVC.
UID_MARKER = "MX_VERSION_UID"
RELEASE_POLL_INTERVAL = 0.05


def read_uid_marker(broadcast_dir: Path, step: int) -> str | None:
    """Return the version uid the trainer published for ``step``, or None."""
    marker = get_step_path(broadcast_dir, step) / UID_MARKER
    return marker.read_text().strip() if marker.exists() else None


def get_latest_uid_marker_step(broadcast_dir: Path) -> int | None:
    """Return the highest step whose MX_VERSION_UID marker the trainer has written."""
    step_dirs = list(broadcast_dir.glob("step_*"))
    steps = sorted(int(step_dir.name.split("_")[-1]) for step_dir in step_dirs)
    for step in reversed(steps):
        if (broadcast_dir / f"step_{step}" / UID_MARKER).exists():
            return step
    return None


async def resolve_ready_version(
    control: ModelExpressControlClient,
    broadcast_dir: Path,
    step: int,
    poll_interval: float = 0.5,
    stopped: asyncio.Event | None = None,
) -> str:
    """Wait for the trainer's uid marker for ``step``, then for the version to be READY.

    Shared by the orchestrator's startup sync and the watcher's steady-state gate:
    the mx_refit version lifecycle is uniform, so v0 and every later version use
    the same resolve path. ``stopped``, when given, lets the caller cancel the poll.
    """
    uid = read_uid_marker(broadcast_dir, step)
    if uid is None:
        await wait_for_path(get_step_path(broadcast_dir, step) / UID_MARKER)
        uid = read_uid_marker(broadcast_dir, step)
    assert uid is not None
    while stopped is None or not stopped.is_set():
        version = await asyncio.to_thread(control.get_weight_version, uid)
        if version.state is WeightVersionState.READY:
            return uid
        await asyncio.sleep(poll_interval)
    raise asyncio.CancelledError


class MXRefitWeightBroadcast(WeightBroadcast):
    def __init__(
        self,
        output_dir: Path,
        config: MXRefitWeightBroadcastConfig,
        parallel_dims: ParallelDims,
        model_name: str,
    ) -> None:
        super().__init__(output_dir)
        self.config = config
        self.parallel_dims = parallel_dims
        self.model_name = model_name
        self.world = get_world()
        self._run_id = uuid.uuid4().hex[:8]
        self._initialized = False
        self._client: ModelExpressTrainerClient | None = None
        self._control: ModelExpressControlClient | None = None
        self._expected_slots: list[str] = []

    @property
    def server_url(self) -> str:
        return f"{self.config.host}:{self.config.port}"

    def _initialize(self, model: nn.Module) -> None:
        # FSDP adapter selection is env-gated in the MX client factory.
        os.environ["MX_TRAINER_ENGINE"] = "FSDP"
        self._client = ModelExpressTrainerClient.initialize(
            ModelExpressTrainerConfig(
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
    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        if not self._initialized:
            self._initialize(model)
        assert self._client is not None

        # Rank 0 creates the per-step version; broadcast its uid to all ranks.
        uid_holder: list[str | None] = [None]
        if self.world.is_master:
            assert self._control is not None
            version = self._control.create_weight_version(
                model_name=self.model_name,
                idempotency_key=f"{self._run_id}:{step}",
                payload_format=WeightPayloadFormat.FULL_TENSOR,
                version_number=step,
                expected_source_slots=self._expected_slots,
            )
            uid_holder[0] = version.version_id
        dist.broadcast_object_list(uid_holder, src=0)
        uid = uid_holder[0]
        assert uid is not None

        self._client.publish_version(version=WeightVersionRef(uid))
        if self.world.is_master:
            self._write_uid_marker(step, uid)

        # Block until the generator has pulled (orchestrator moved the version to
        # RELEASING) so training does not overwrite the staging arenas mid-read.
        # Rank 0 observes RELEASING; the barrier propagates that to every rank
        # before any of them withdraw their shard.
        if self.world.is_master:
            self._wait_released(uid)
        dist.barrier()
        self._client.release_version(version=WeightVersionRef(uid))

    def _wait_released(self, uid: str) -> None:
        assert self._control is not None
        while True:
            try:
                state = self._control.get_weight_version(uid).state
            except grpc.RpcError as error:
                if error.code() is grpc.StatusCode.NOT_FOUND:
                    return  # already retired == generator done
                raise
            if state is WeightVersionState.RELEASING:
                return
            time.sleep(RELEASE_POLL_INTERVAL)

    def _write_uid_marker(self, step: int, uid: str) -> None:
        step_dir = get_step_path(get_broadcast_dir(self.output_dir), step)
        step_dir.mkdir(parents=True, exist_ok=True)
        (step_dir / UID_MARKER).write_text(uid)
