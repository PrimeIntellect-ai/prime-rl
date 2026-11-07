"""FSDP weight updates through ModelExpress's public clients."""

import asyncio
import atexit
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import torch
import torch.distributed as dist
from modelexpress_rl import (
    FSDPTrainerContext,
    ModelExpressControlClient,
    ModelExpressTrainerClient,
    ModelExpressTrainerConfig,
    TrainerStagingMode,
    TrainerTensorsMetadata,
    WeightPayloadFormat,
    WeightVersionRef,
    WeightVersionState,
)
from torch import nn

from prime_rl.configs.shared import ModelExpressWeightBroadcastConfig
from prime_rl.transports.weights.base import WeightReceiver, WeightSender
from prime_rl.utils.pathing import wait_for_path

VERSION_MARKER = ".mx_version"
INSTALLED_MARKER = ".installed"


class ModelExpressWeightSender(WeightSender):
    def __init__(self, output_dir: Path, config: ModelExpressWeightBroadcastConfig, *, model_name: str):
        super().__init__(output_dir, config.timeout)
        self.config = config
        self.model_name = model_name
        self._trainer: ModelExpressTrainerClient | None = None
        self._control: ModelExpressControlClient | None = None
        self._mesh_id: str | None = None

    def _initialize(self, model: nn.Module) -> None:
        tensors = model.state_dict()
        keep_fp32 = getattr(model, "keep_in_fp32_for_weight_transfer", None)
        overrides = {name: torch.float32 for name in tensors if keep_fp32 is not None and keep_fp32(name)}
        self._trainer = ModelExpressTrainerClient.initialize(
            ModelExpressTrainerConfig(
                engine_context=FSDPTrainerContext(wire_dtype_overrides=overrides),
                model_name=self.model_name,
                device_id=self.world.local_rank,
                server_url=f"{self.config.host}:{self.config.port}",
                staging_mode=TrainerStagingMode[self.config.staging_mode],
                payload_format=WeightPayloadFormat.FULL_TENSOR,
            )
        )
        atexit.register(self._trainer.close)
        metadata = self._trainer.bind_tensors(tensors)
        gathered_workers = [None] * self.world.world_size
        dist.all_gather_object(gathered_workers, (self._trainer.worker_id, metadata))
        if self.world.is_master:
            workers: dict[str, TrainerTensorsMetadata] = dict(gathered_workers)
            self._control = ModelExpressControlClient.connect(server_url=f"{self.config.host}:{self.config.port}")
            atexit.register(self._control.close)
            mesh = self._control.create_trainer_mesh(
                model_name=self.model_name,
                idempotency_key=uuid.uuid5(uuid.NAMESPACE_URL, repr((self.model_name, sorted(workers)))).hex,
                workers=workers,
            )
            self._mesh_id = mesh.mesh_id
            atexit.register(self._control.delete_trainer_mesh, self._mesh_id)

    def _publish(self, model: nn.Module, step: int, step_dir: Path) -> WeightVersionRef:
        if self._trainer is None:
            self._initialize(model)
        assert self._trainer is not None
        offered = [None]
        if self.world.is_master:
            assert self._control is not None and self._mesh_id is not None
            version = self._control.create_weight_version(
                model_name=self.model_name,
                idempotency_key=uuid.uuid4().hex,
                payload_format=WeightPayloadFormat.FULL_TENSOR,
                trainer_mesh_id=self._mesh_id,
                version_number=step,
            )
            offered[0] = version.version_id
            marker = step_dir / VERSION_MARKER
            pending = marker.with_suffix(".pending")
            pending.write_text(version.version_id)
            pending.replace(marker)
        dist.broadcast_object_list(offered, src=0)
        version = WeightVersionRef(offered[0])
        self._trainer.publish_version(version=version)
        return version

    @torch.no_grad()
    def _broadcast(self, model: nn.Module, step: int, step_dir: Path) -> None:
        version = self._publish(model, step, step_dir)
        assert self._trainer is not None

        # Installation acknowledgment is a PrimeRL outcome, not MX retirement.
        installation_failed = [False]
        installation_error = None
        if self.world.is_master:
            try:
                installed = step_dir / INSTALLED_MARKER
                deadline = time.monotonic() + self.timeout
                while not installed.exists():
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"Inference did not install version {version.version_id} within {self.timeout}s"
                        )
                    time.sleep(0.1)
                if installed.read_text() != version.version_id:
                    raise RuntimeError("Inference acknowledged a different weight version")
            except Exception as exc:
                installation_failed[0] = True
                installation_error = exc
        dist.broadcast_object_list(installation_failed, src=0)
        if installation_failed[0]:
            if installation_error is not None:
                raise installation_error
            raise RuntimeError("Inference weight installation failed on trainer rank zero")
        self._trainer.release_version(version=version)
        dist.barrier()


class ModelExpressPullWeightSender(ModelExpressWeightSender):
    """Publishes each version and moves on. Engines pull the newest one on their own
    (``prime_rl.inference.vllm.weight_pull``), so the trainer never waits for inference."""

    waits_for_receiver = False

    def __init__(self, output_dir: Path, config: ModelExpressWeightBroadcastConfig, *, model_name: str):
        super().__init__(output_dir, config, model_name=model_name)
        self._published: list[WeightVersionRef] = []

    def _initialize(self, model: nn.Module) -> None:
        super()._initialize(model)
        if not self.world.is_master:
            return
        assert self._control is not None
        # Versions a previous trainer left READY died with its memory; engines must not chase them.
        for version in self._control.list_weight_versions(model_name=self.model_name):
            if version.state is WeightVersionState.READY and version.trainer_mesh_id != self._mesh_id:
                self._control.delete_weight_version(version.version_id)
                self.logger.info(f"Deleted version {version.version_number} of a previous trainer")

    @torch.no_grad()
    def _broadcast(self, model: nn.Module, step: int, step_dir: Path) -> None:
        self._published.append(self._publish(model, step, step_dir))
        # Engines are still copying the previous version when this one appears, so the two newest
        # stay. An older one an engine still leases is retried on the next publish.
        older, newest = self._published[:-2], self._published[-2:]
        self._published = [version for version in older if not self._retire(version)] + newest
        dist.barrier()

    def _retire(self, version: WeightVersionRef) -> bool:
        """Release and delete a superseded version; False while an engine still leases it."""
        assert self._trainer is not None
        try:
            self._trainer.release_version(version=version)
        except Exception as error:
            # modelexpress_rl has no typed error for a leased version; its own release loop matches the
            # server's message the same way (modelexpress_rl/train/methods/full_tensor.py).
            details = getattr(error, "details", None)
            if not callable(details) or details() != "weight version has an active lease":
                raise
            return False
        if self.world.is_master:
            assert self._control is not None
            self._control.delete_weight_version(version.version_id)
        return True


class ModelExpressWeightReceiver(WeightReceiver):
    async def initialize(self) -> None:
        self._control = ModelExpressControlClient.connect(server_url=f"{self.config.host}:{self.config.port}")
        atexit.register(self._control.close)
        await self.admin_plane.initialize_modelexpress(
            self.config.host,
            self.config.port,
            self.config.timeout,
            self.config.inference_world_size,
            self.config.staging_buffer_bytes,
            self.config.staging_buffers_count,
        )

    async def receive(self, step: int) -> None:
        self._ack(step)
        marker = self.step_dir(step) / VERSION_MARKER
        await asyncio.wait_for(wait_for_path(marker, interval=0.01), timeout=self.config.timeout)
        uid = marker.read_text()
        deadline = time.monotonic() + self.config.timeout
        while True:
            version = await asyncio.to_thread(self._control.get_weight_version, uid)
            if version.state is WeightVersionState.READY:
                break
            if version.state is WeightVersionState.RELEASING:
                raise RuntimeError(f"Weight version {uid} was retired before installation")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Weight version {uid} was not ready within {self.config.timeout}s")
            await asyncio.sleep(0.1)
        await self.admin_plane.update_modelexpress_weights(version_uid=uid, step=step)
        await asyncio.to_thread(self._control.delete_weight_version, uid)
        installed = self.step_dir(step) / INSTALLED_MARKER
        pending = installed.with_suffix(".pending")
        pending.write_text(uid)
        pending.replace(installed)


class ModelExpressPullWeightReceiver(ModelExpressWeightReceiver):
    """Follows published versions without moving bytes: engines install them on their own and
    every token records the version that sampled it."""

    async def initialize(self) -> None:
        self._control = ModelExpressControlClient.connect(server_url=f"{self.config.host}:{self.config.port}")
        atexit.register(self._control.close)

    def is_published(self, step: int) -> bool:
        newest = self._newest_published()
        return newest is not None and newest >= step

    def next_version(self, current: int) -> int:
        return max(self._newest_published() or current, current)

    async def receive(self, step: int) -> None:
        return None

    async def wait_published(self, step: int, cancelled: Callable[[], bool] | None = None) -> None:
        while not await asyncio.to_thread(self.is_published, step):
            if cancelled is not None and cancelled():
                raise asyncio.CancelledError
            await asyncio.sleep(0.1)

    def _newest_published(self) -> int | None:
        """The trainer keeps its newest version READY until a newer one is."""
        versions = self._control.list_weight_versions(model_name=self.model_name)
        return max(
            (version.version_number for version in versions if version.state is WeightVersionState.READY), default=None
        )
