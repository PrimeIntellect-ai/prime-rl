"""FSDP weight updates through ModelExpress's public clients."""

import asyncio
import atexit
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
from torch.distributed.tensor import DTensor

from prime_rl.configs.shared import ModelExpressWeightBroadcastConfig
from prime_rl.transports.weights.base import FINISHED_MARKER, WeightReceiver, WeightSender
from prime_rl.utils.pathing import wait_for_path
from prime_rl.utils.process import set_modelexpress_port_defaults

VERSION_MARKER = ".mx_version"
INSTALLED_MARKER = ".installed"


class ModelExpressWeightSender(WeightSender):
    max_trainer_readers = 0
    release_after_trainer_reads = False

    def __init__(self, output_dir: Path, config: ModelExpressWeightBroadcastConfig, *, model_name: str):
        super().__init__(output_dir, config.timeout)
        self.config = config
        self.model_name = model_name
        self._trainer: ModelExpressTrainerClient | None = None
        self._control: ModelExpressControlClient | None = None
        self._mesh_id: str | None = None
        self._bound: dict[str, torch.Tensor] = {}

    def _initialize(self, model: nn.Module) -> None:
        set_modelexpress_port_defaults(trainer=True)
        tensors = self._bound = model.state_dict()
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
                release_after_trainer_reads=self.release_after_trainer_reads,
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

    @torch.no_grad()
    def _refresh_bound(self, model: nn.Module) -> None:
        """Copy the live value into every bound tensor that is not the live tensor itself. ModelExpress publishes
        the tensors bound on the first publish, but a packed parameter sharded along its packing dimension yields
        its logical tensors as copies (see ``PackedParameterSpec``), which would keep their first values."""
        for name, live in model.state_dict().items():
            bound, live = _local(self._bound[name]), _local(live)
            if bound.data_ptr() != live.data_ptr():
                bound.copy_(live)

    def _publish(self, model: nn.Module, step: int, step_dir: Path) -> WeightVersionRef:
        if self._trainer is None:
            self._initialize(model)
        else:
            self._refresh_bound(model)
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
                max_trainer_readers=self.max_trainer_readers,
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
    """Publishes every version. Engines install them on their own (``prime_rl.inference.vllm.weight_pull``), so
    the trainer waits for no install, only for an engine still reading its previous copy.

    The trainer holds one copy, so before publishing it frees the previous one, waiting for any engine still
    reading that from the trainer, at most one install. Skipping the publish instead would leave engines on a
    version the orchestrator no longer samples from while the trainer waits for a batch. Older versions stay
    READY for engines to copy from peers that installed them, until they fall ``RETAINED_VERSIONS`` behind."""

    RETAINED_VERSIONS = 3
    RELEASE_AT_EXIT_TIMEOUT_S = 10
    # One engine at a time reads a version from the trainer and each engine feeds one peer at a time, so a version
    # spreads through the fleet in a doubling tree instead of every engine sharing the trainer's network.
    max_trainer_readers = 1
    release_after_trainer_reads = True
    waits_for_receiver = False

    def __init__(self, output_dir: Path, config: ModelExpressWeightBroadcastConfig, *, model_name: str):
        super().__init__(output_dir, config, model_name=model_name)
        self._published: list[WeightVersionRef] = []

    def _initialize(self, model: nn.Module) -> None:
        super()._initialize(model)
        # Runs before the client closes: an engine copying the last version gets a moment to finish.
        atexit.register(self._release_newest, self.RELEASE_AT_EXIT_TIMEOUT_S)
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
        released = torch.tensor([int(self._release_newest(self.timeout))], device=torch.cuda.current_device())
        # Every rank publishes together, so all agree whether every copy of the previous version was freed.
        dist.all_reduce(released, op=dist.ReduceOp.MIN)
        if not released.item():
            raise TimeoutError(f"An engine still read the previous version from the trainer after {self.timeout}s")
        self._published.append(self._publish(model, step, step_dir))
        if len(self._published) > self.RETAINED_VERSIONS:
            retired = self._published.pop(0)
            if self.world.is_master:
                assert self._control is not None
                self._control.delete_weight_version(retired.version_id)
        dist.barrier()

    def close(self) -> None:
        """Retire every version once training ends, so the trainer's mesh can be deleted at exit."""
        if not self._published:
            return
        self._release_newest(self.RELEASE_AT_EXIT_TIMEOUT_S)
        dist.barrier()
        if self.world.is_master:
            assert self._control is not None
            for version in self._published:
                self._control.delete_weight_version(version.version_id)
        self._published.clear()

    def _release_newest(self, timeout: float) -> bool:
        """Free the newest version's trainer copy once no engine reads it from the trainer, within ``timeout``
        seconds. An engine that dies stops renewing its version lease, which then expires, so only a live read
        holds the trainer."""
        if not self._published or self._trainer is None:
            return True
        return self._trainer.release_version(version=self._published[-1], timeout=timeout)


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


class ModelExpressPullWeightReceiver(WeightReceiver):
    """Follows published versions without moving bytes: engines install them on their own and every token
    records the version that sampled it. A step is published once the trainer finished its broadcast having
    created its version."""

    sender_waits = False

    def is_published(self, step: int) -> bool:
        step_dir = self.step_dir(step)
        return (step_dir / FINISHED_MARKER).exists() and (step_dir / VERSION_MARKER).exists()

    async def receive(self, step: int) -> None:
        return None


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor
