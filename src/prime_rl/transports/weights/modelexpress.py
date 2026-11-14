"""FSDP weight updates through ModelExpress's public clients."""

import asyncio
import atexit
import json
import os
import time
import uuid
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
from prime_rl.transports.weights.base import WeightReceiver, WeightSender, marker_exists
from prime_rl.utils.distributed import all_gather_bytes, broadcast_bytes
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
        os.environ["MX_METADATA_PORT"] = str(self.config.metadata_port)
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
        local_worker = json.dumps(
            [self._trainer.worker_id, metadata.logical_shard_id, metadata.metadata_endpoint],
            separators=(",", ":"),
        ).encode()
        gathered_workers = []
        for payload in all_gather_bytes(local_worker):
            worker_id, logical_shard_id, metadata_endpoint = json.loads(payload)
            gathered_workers.append(
                (
                    worker_id,
                    TrainerTensorsMetadata(
                        logical_shard_id=logical_shard_id,
                        metadata_endpoint=metadata_endpoint,
                    ),
                )
            )
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

    @torch.no_grad()
    def _broadcast(self, model: nn.Module, step: int, step_dir: Path) -> None:
        if self._trainer is None:
            self._initialize(model)
        assert self._trainer is not None
        offered = None
        if self.world.is_master:
            assert self._control is not None and self._mesh_id is not None
            version = self._control.create_weight_version(
                model_name=self.model_name,
                idempotency_key=uuid.uuid4().hex,
                payload_format=WeightPayloadFormat.FULL_TENSOR,
                trainer_mesh_id=self._mesh_id,
                version_number=step,
            )
            offered = version.version_id
            marker = step_dir / VERSION_MARKER
            pending = marker.with_suffix(".pending")
            pending.write_text(version.version_id)
            pending.replace(marker)
        offered = broadcast_bytes(offered.encode() if offered is not None else None).decode()
        version = WeightVersionRef(offered)
        self._trainer.publish_version(version=version)

        # Installation acknowledgment is a PrimeRL outcome, not MX retirement.
        installation_failed = False
        installation_error = None
        if self.world.is_master:
            try:
                installed = step_dir / INSTALLED_MARKER
                deadline = time.monotonic() + self.timeout
                while not marker_exists(installed):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"Inference did not install version {version.version_id} within {self.timeout}s"
                        )
                    time.sleep(0.1)
                if installed.read_text() != version.version_id:
                    raise RuntimeError("Inference acknowledged a different weight version")
            except Exception as exc:
                installation_failed = True
                installation_error = exc
        installation_failed = broadcast_bytes(bytes([installation_failed]) if self.world.is_master else None) == b"\x01"
        if installation_failed:
            if installation_error is not None:
                raise installation_error
            raise RuntimeError("Inference weight installation failed on trainer rank zero")
        self._trainer.release_version(version=version)
        dist.barrier()


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
