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
from prime_rl.transports.weights.base import FINISHED_MARKER, WeightReceiver, WeightSender, marker_exists
from prime_rl.utils.distributed import all_gather_bytes, broadcast_bytes
from prime_rl.utils.pathing import wait_for_path

VERSION_MARKER = ".mx_version"
INSTALLED_MARKER = ".installed"


class ModelExpressWeightSender(WeightSender):
    """Publishes each policy version to ModelExpress. Push: the orchestrator installs it on every engine, and the
    trainer waits for that before freeing it. Pull (``weight_broadcast.pull``): the engines install it on their own
    (``prime_rl.inference.vllm.weight_pull``), and the trainer waits for no engine.

    The trainer holds one version: each rank publishes from one host copy, refilled on every publish. Under pull a
    version therefore stays available until the next publish, which first retires it: no new reader can lease it,
    and the reads already in flight finish."""

    def __init__(self, output_dir: Path, config: ModelExpressWeightBroadcastConfig, *, model_name: str):
        super().__init__(output_dir, config.timeout)
        self.config = config
        self.model_name = model_name
        self.acknowledged = not config.pull
        self._trainer: ModelExpressTrainerClient | None = None
        self._control: ModelExpressControlClient | None = None
        self._mesh_id: str | None = None
        self._tensors: dict[str, torch.Tensor] = {}
        self._published: tuple[int, WeightVersionRef] | None = None
        """The step and version published last under pull, which engines may still read."""

    def _initialize(self, model: nn.Module) -> None:
        tensors = self._tensors
        tensors.update(model.state_dict())
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
            if self.config.pull:
                self._retire_other_meshes()

    def _retire_other_meshes(self) -> None:
        """A previous trainer's versions died with its memory, but stay READY: retire them, so engines that pull do
        not try to install them."""
        assert self._control is not None
        for version in self._control.list_weight_versions(model_name=self.model_name):
            if version.state is WeightVersionState.READY and version.trainer_mesh_id != self._mesh_id:
                self._control.delete_weight_version(version.version_id)
                self.logger.info(
                    f"Retired version {version.version_number} ({version.version_id}) of a previous trainer"
                )

    @torch.no_grad()
    def _broadcast(self, model: nn.Module, step: int, step_dir: Path) -> None:
        if self._trainer is None:
            self._initialize(model)
        else:
            # The trainer client keeps the dict bound at initialization and re-reads its values on every publish.
            # A state dict entry can be a fresh copy on each call (a packed parameter split across FSDP shards),
            # so refresh the values in place instead of publishing the copies from the first broadcast.
            self._tensors.update(model.state_dict())
        if self.config.pull:
            self._publish_for_pull(step, step_dir)
            return
        version = self._publish(step, step_dir)

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
        # Drop the state dict until the next broadcast refreshes it. Its copied entries (the packed parameters'
        # splits) would otherwise stay allocated through every training step.
        self._tensors.clear()
        dist.barrier()

    def _publish_for_pull(self, step: int, step_dir: Path) -> None:
        """Retire the version published last, waiting only for reads of it still in flight, then publish v{step}
        from the host copy it frees. No engine install is awaited."""
        start = time.perf_counter()
        if self._published is not None:
            self._retire(self._published[1])
        retired = time.perf_counter()
        self._published = (step, self._publish(step, step_dir))
        self._tensors.clear()
        dist.barrier()
        self.logger.debug(
            f"Published v{step} for engines to pull in {time.perf_counter() - start:.2f}s "
            f"(retiring the previous version took {retired - start:.2f}s)"
        )

    def _retire(self, version: WeightVersionRef) -> None:
        """Stop new reads of ``version`` and wait until no engine reads it, so its host copy can be refilled."""
        assert self._trainer is not None
        if self.world.is_master:
            assert self._control is not None
            # RELEASING accepts no new lease; engines holding one renew it until their read is done.
            self._control.delete_weight_version(version.version_id)
        dist.barrier()
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                # Retries while a lease is held, for the client's RPC timeout, then raises
                self._trainer.release_version(version=version)
                return
            except Exception as error:
                if time.monotonic() >= deadline:
                    raise
                self.logger.warning(f"An engine may still read version {version.version_id}; waiting: {error}")

    def close(self) -> None:
        """Keep the last version available until the engines serve it (the consumer marks it installed), then
        retire it. A trainer that exited right after its last publish would leave it unreadable."""
        if not self.config.pull or self._published is None:
            return
        step, version = self._published
        installed = self.step_dir(step) / INSTALLED_MARKER
        served = True
        if self.world.is_master:
            deadline = time.monotonic() + self.timeout
            while not marker_exists(installed) and time.monotonic() < deadline:
                time.sleep(0.5)
            served = marker_exists(installed)
            if not served:
                self.logger.warning(f"The engines did not report v{step} as served within {self.timeout}s")
        broadcast_bytes(bytes([served]) if self.world.is_master else None)
        self._retire(version)
        self._published = None

    def _publish(self, step: int, step_dir: Path) -> WeightVersionRef:
        """Create v{step} on the master, write its id to the step dir, and publish every rank's shard of it."""
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
        return version


class ModelExpressWeightReceiver(WeightReceiver):
    async def initialize(self) -> None:
        self._control = ModelExpressControlClient.connect(server_url=f"{self.config.host}:{self.config.port}")
        atexit.register(self._control.close)
        await self.admin_plane.initialize_modelexpress(
            self.config.host,
            self.config.port,
            self.config.timeout,
            self.config.staging_buffer_bytes,
            self.config.staging_buffers_count,
            self.config.staging_host_bytes,
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


class ModelExpressPullWeightReceiver(WeightReceiver):
    """The consumer side of ``weight_broadcast.pull``: the engines watch ModelExpress and move themselves onto each
    version (``prime_rl.inference.vllm.weight_pull``), so receiving a version only waits until every engine serves
    it or a newer one, and marks it installed. Nothing is initialized, paused or pushed from here."""

    engines_pull = True
    poll_interval = 0.1

    def serves(self, version: int | None, step: int) -> bool:
        """Whether an engine serving ``version`` serves v{step} or a newer version of this trainer. A trainer that
        restarted from a checkpoint drops the broadcasts beyond it, so a newer version without its broadcast is from
        before the restart, and the engine has yet to move back."""
        if version is None or version < step:
            return False
        return version == step or marker_exists(self.step_dir(version) / FINISHED_MARKER)

    async def receive(self, step: int) -> None:
        start = last_log = time.monotonic()
        while True:
            served = await self.admin_plane.served_versions()
            if all(self.serves(version, step) for version in served):
                break
            now = time.monotonic()
            if now - start > self.config.timeout:
                raise TimeoutError(f"The engines did not pull policy v{step} within {self.config.timeout}s: {served}")
            if now - last_log > 60:
                self.logger.warning(f"Still waiting for the engines to pull policy v{step}; they serve {served}")
                last_log = now
            await asyncio.sleep(self.poll_interval)
        # The trainer keeps its last version available until it is marked installed (``close``)
        installed = self.step_dir(step) / INSTALLED_MARKER
        version_marker = self.step_dir(step) / VERSION_MARKER
        pending = installed.with_suffix(".pending")
        pending.write_text(version_marker.read_text() if version_marker.exists() else "")
        pending.replace(installed)
