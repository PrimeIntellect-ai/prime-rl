"""vLLM extension using the public ModelExpress generator client."""

import atexit
import os
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import torch
from modelexpress_rl import (
    ModelExpressGeneratorClient,
    ModelExpressGeneratorConfig,
    VllmGeneratorContext,
    WeightSource,
    WeightVersionRef,
)
from torch import nn
from vllm.logger import init_logger

from prime_rl.inference.vllm.worker.weight_transfer import WeightRPCMixin

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker
else:
    Worker = object

logger = init_logger(__name__)


@dataclass
class StagedVersion:
    """A version this worker stages, or staged, for an engine that pulls its weights."""

    version_uid: str
    done: threading.Event = field(default_factory=threading.Event)
    handle: object | None = None
    """The generator client's staged handle."""
    error: str | None = None


def keep_integer_parameters_as_buffers(model: nn.Module) -> list[str]:
    """Re-register the engine's integer parameters as non-persistent buffers, keeping their loaded values.

    ModelExpress publishes only floating-point trainer tensors, and a full refit must write every engine parameter,
    so an integer parameter fails every update. Integer parameters are lookup tables the trainer never changes, such
    as DeepSeek-V4's hash-routing table `tid2eid`, so they leave the refit's coverage as buffers.

    The refit installs through vLLM's layerwise reload, which rebuilds each layer from the tensors recorded at load
    and copies the rebuilt values back into the live ones. The table is dropped from that record, so the reload
    neither rebuilds it nor waits for it, and it is non-persistent, so the copy-back skips it while unwritten: the
    live table keeps the values loaded from the checkpoint.

    Other modules may hold the same parameter as a plain attribute (DeepSeek-V4's routed experts keep the gate's table
    as `hash_indices_table`). Those references move to the buffer too: the refit restores plain tensor attributes
    with `setattr`, which would try to register a leftover `Parameter` as a new parameter of that module.
    """
    from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info

    moved, buffers = [], {}
    for module_name, module in model.named_modules():
        for name, param in list(module.named_parameters(recurse=False)):
            if not param.is_floating_point():
                del module._parameters[name]
                module.register_buffer(name, param.data, persistent=False)
                get_layerwise_info(module).restore_metadata[0].pop(name, None)
                buffers[id(param)] = getattr(module, name)
                moved.append(f"{module_name}.{name}" if module_name else name)
    for module in model.modules():
        for attribute, value in list(vars(module).items()):
            if id(value) in buffers:
                object.__setattr__(module, attribute, buffers[id(value)])
    return moved


class ModelExpressWeightUpdateWorker(WeightRPCMixin, Worker):
    def liveness_probe(self) -> None:
        return None

    def init_broadcaster(
        self,
        host,
        port,
        timeout,
        session_id,
        staging_buffer_bytes=None,
        staging_buffers_count=1,
        model_name=None,
    ):
        # The id names this worker's MX registration, version leases and NIXL agent, so it must be unique per worker.
        # vLLM's ranks and device indices are not: they restart in every data-parallel engine, every external-LB server
        # and every P/D role. The GPU's UUID and the process id are, without any knowledge of the inference layout.
        worker_id = f"{session_id}:{torch.cuda.get_device_properties(self.device).uuid}:{os.getpid()}"
        # Admin planes that discover engines call this before every update; the client the engine runs stays.
        previous = getattr(self, "_generator", None)
        if previous is not None and previous.worker_id == worker_id:
            return
        if previous is not None:
            self.modelexpress_discard()
            previous.close()
        # A new client has no record of any trainer's layout yet
        self._mx_mesh = None

        from transformers import AutoConfig

        from prime_rl.trainer.models import get_custom_causal_lm_cls
        from prime_rl.trainer.models.conversion_ops import apply_prime_to_hf

        # The trainer publishes PrimeRL-native tensors, so the conversion chain must come from the
        # PrimeRL config class registered on import above, not vLLM's own config class.
        model_config = self.model_runner.model_config
        hf_config = AutoConfig.from_pretrained(model_config.model, trust_remote_code=model_config.trust_remote_code)
        chain = get_custom_causal_lm_cls(hf_config).conversion_chain(hf_config)
        model = cast(nn.Module, self.model_runner.get_model())
        keep_integer_parameters_as_buffers(model)
        self._generator = ModelExpressGeneratorClient.initialize(
            ModelExpressGeneratorConfig(
                engine_context=VllmGeneratorContext(
                    model=model,
                    vllm_config=self.vllm_config,
                    convert_native_to_hf=(lambda tensors: apply_prime_to_hf(tensors, chain)) if chain else None,
                ),
                # The name the trainer publishes versions under; the served model's by default
                model_name=model_name or self.model_runner.model_config.model,
                server_url=f"{host}:{port}",
                source_order=(WeightSource.TRAINER,),
                worker_id=worker_id,
                staging_buffer_bytes=staging_buffer_bytes,
                staging_buffers_count=staging_buffers_count,
            )
        )
        atexit.register(self._generator.close)

    # Engines that pull their weights (``prime_rl.inference.vllm.weight_pull``) stage a version while they serve and
    # install it in a pause of their own. Staging writes only ModelExpress's own buffers, except the first staging
    # of a trainer's versions: it records how the trainer's tensors map onto the engine's on the live model, so it
    # must run paused. Later versions of the same trainer reuse that record.

    def modelexpress_can_stage_live(self, mesh: str) -> bool:
        """Whether this worker can stage a version of trainer ``mesh`` while the engine serves."""
        return getattr(self, "_mx_mesh", None) == mesh

    def modelexpress_stage(self, version_uid: str, live: bool) -> None:
        """Stage a version, on a background thread while the engine serves if ``live``, else right away."""
        self.modelexpress_discard()
        staged = StagedVersion(version_uid)
        self._mx_staged = staged
        if not live:
            self._stage(staged)
            if staged.error is not None:
                raise RuntimeError(staged.error)
            return
        if getattr(self, "_mx_stream", None) is None:
            self._mx_stream = torch.cuda.Stream(device=self.device)
        threading.Thread(target=self._stage, args=(staged, self._mx_stream), name="mx-stage", daemon=True).start()

    def _stage(self, staged: StagedVersion, stream: torch.cuda.Stream | None = None) -> None:
        try:
            if stream is None:
                staged.handle = self._generator.stage_weight(version=WeightVersionRef(staged.version_uid))
            else:
                # Copies a stage makes run on their own stream, beside the engine's steps
                torch.cuda.set_device(self.device)
                with torch.cuda.stream(stream):
                    staged.handle = self._generator.stage_weight(version=WeightVersionRef(staged.version_uid))
                stream.synchronize()
        except Exception as error:
            logger.warning(f"Staging ModelExpress version {staged.version_uid} failed: {error!r}")
            staged.error = f"{type(error).__name__}: {error}"
        finally:
            staged.done.set()

    def modelexpress_stage_status(self) -> str:
        """``staging``, ``staged``, ``idle`` (nothing staged) or ``failed: <error>``."""
        staged = getattr(self, "_mx_staged", None)
        if staged is None:
            return "idle"
        if not staged.done.is_set():
            return "staging"
        return "staged" if staged.error is None else f"failed: {staged.error}"

    def modelexpress_discard(self) -> None:
        """Release the staged version and the version lease it holds. A stage in flight finishes first."""
        staged, self._mx_staged = getattr(self, "_mx_staged", None), None
        if staged is None:
            return
        staged.done.wait()
        if staged.handle is not None:
            staged.handle.release()

    @torch.no_grad()
    def modelexpress_apply(self, version_uid: str, mesh: str) -> None:
        """Install the staged version into the live weights. Call while the engine is paused."""
        staged = getattr(self, "_mx_staged", None)
        if staged is None or staged.version_uid != version_uid or not staged.done.is_set() or staged.handle is None:
            raise RuntimeError(f"ModelExpress version {version_uid} is not staged on this worker")
        self._mx_staged = None
        # Until the install completes, the weights may be partly written: the next stage must run paused, and the
        # worker serves no version
        self._mx_mesh = None
        self._mx_served = None
        try:
            self._generator.apply_weight(staged.handle)
        finally:
            staged.handle.release()
        self._mx_mesh = mesh
        self._mx_served = version_uid

    def modelexpress_served(self) -> str | None:
        """The ModelExpress version this worker installed last, None before the first or after a failed install."""
        return getattr(self, "_mx_served", None)

    @torch.no_grad()
    def update_weights_from_modelexpress(self, version_uid: str) -> None:
        if not version_uid:
            raise ValueError("modelexpress requires version_uid")
        version = WeightVersionRef(version_uid)
        staged = self._generator.stage_weight(version=version)
        self._mx_served = None
        try:
            self._generator.apply_weight(staged)
        finally:
            staged.release()
        self._mx_served = version_uid
