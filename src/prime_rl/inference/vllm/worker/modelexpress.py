"""vLLM extension using the public ModelExpress generator client."""

import atexit
from typing import TYPE_CHECKING, cast

import torch
from modelexpress_rl import (
    ModelExpressGeneratorClient,
    ModelExpressGeneratorConfig,
    RefitSourceUnavailable,
    VllmGeneratorContext,
    WeightSource,
    WeightVersionRef,
)
from torch import nn

from prime_rl.inference.vllm.worker.weight_transfer import WeightRPCMixin

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker
else:
    Worker = object


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
        rank_offset,
        inference_world_size,
        timeout,
        session_id="default",
        staging_buffer_bytes=None,
        staging_buffers_count=1,
    ):
        """Push: the orchestrator installs every version on every engine at once, so only the trainer has it."""
        del inference_world_size, timeout
        worker_id = f"{session_id}:{rank_offset + self.rank}"
        # Admin planes that discover engines call this before every update; the client the engine runs stays.
        previous = getattr(self, "_generator", None)
        if previous is not None and previous.worker_id == worker_id:
            return
        if previous is not None:
            previous.close()

        self._init_generator(
            model_name=self.model_runner.model_config.model,
            server_url=f"{host}:{port}",
            source_order=(WeightSource.TRAINER,),
            worker_id=worker_id,
            reader_group=None,
            staging_buffer_bytes=staging_buffer_bytes,
            staging_buffers_count=staging_buffers_count,
            install_through_loader=False,
        )

    def init_modelexpress_pull(
        self, model_name: str, server_url: str, server_id: str, staging_buffer_bytes: int, staging_buffers_count: int
    ) -> None:
        """Pull: a version is read from the trainer or any engine that installed it, and all workers of this
        server count once against the trainer's reader cap."""
        self._init_generator(
            model_name=model_name,
            server_url=server_url,
            source_order=None,
            worker_id=None,
            reader_group=server_id,
            staging_buffer_bytes=staging_buffer_bytes,
            staging_buffers_count=staging_buffers_count,
            install_through_loader=True,
        )

    def stage_weights_from_modelexpress(self, version_uid: str) -> str | None:
        """Prepare a version without touching the live weights, so every worker can stage before any applies.
        Returns why it could not, holding nothing: a source that is busy or does not have the version yet is an
        expected answer, not a failure."""
        self.discard_staged_modelexpress()
        try:
            self._staged = self._generator.stage_weight(version=WeightVersionRef(version_uid))
        except RefitSourceUnavailable as error:
            return str(error)
        return None

    def discard_staged_modelexpress(self) -> None:
        """Release a staged version, freeing its version lease and the source read it holds."""
        if self._staged is not None:
            self._staged.release()
            self._staged = None

    @torch.no_grad()
    def update_weights_from_modelexpress(self, version_uid: str) -> None:
        """Install a version: the one pull staged, or, under push, stage it now."""
        if not version_uid:
            raise ValueError("modelexpress requires version_uid")
        staged, self._staged = self._staged, None
        if staged is None:
            staged = self._generator.stage_weight(version=WeightVersionRef(version_uid))
        try:
            self._generator.apply_weight(staged)
        finally:
            staged.release()

    def _init_generator(
        self,
        *,
        model_name: str,
        server_url: str,
        source_order: tuple[WeightSource, ...] | None,
        worker_id: str | None,
        reader_group: str | None,
        staging_buffer_bytes: int | None,
        staging_buffers_count: int,
        install_through_loader: bool,
    ) -> None:
        from transformers import AutoConfig

        from prime_rl.trainer.models import get_custom_causal_lm_cls
        from prime_rl.trainer.models.conversion_ops import apply_prime_to_hf

        # The trainer publishes PrimeRL-native tensors, so the conversion chain must come from the
        # PrimeRL config class registered on import above, not vLLM's own config class.
        model_config = self.model_runner.model_config
        hf_config = AutoConfig.from_pretrained(model_config.model, trust_remote_code=model_config.trust_remote_code)
        chain = get_custom_causal_lm_cls(hf_config).conversion_chain(hf_config)
        model = cast(nn.Module, self.model_runner.get_model())
        self._generator = ModelExpressGeneratorClient.initialize(
            ModelExpressGeneratorConfig(
                engine_context=VllmGeneratorContext(
                    model=model,
                    vllm_config=self.vllm_config,
                    convert_native_to_hf=(lambda tensors: apply_prime_to_hf(tensors, chain)) if chain else None,
                    # Pull installs through vLLM's own loader, as NCCL does, so it refits what vLLM can reload.
                    install_through_loader=install_through_loader,
                ),
                model_name=model_name,
                server_url=server_url,
                source_order=source_order,
                worker_id=worker_id,
                reader_group=reader_group,
                staging_buffer_bytes=staging_buffer_bytes,
                staging_buffers_count=staging_buffers_count,
                # vLLM's loaders read host tensors, as from safetensors, and copy only this rank's slices.
                staging_device="cpu" if install_through_loader else None,
            )
        )
        atexit.register(self._generator.close)
        self._staged = None
