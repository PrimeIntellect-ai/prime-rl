"""vLLM extension using the public ModelExpress generator client."""

import atexit
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
        # The id names this worker's MX registration, version leases and NIXL agent, so it must be unique per
        # inference GPU. `self.rank` restarts at 0 in every data-parallel engine of a server; the device index does not.
        worker_id = f"{session_id}:{rank_offset + self.device.index}"
        # Admin planes that discover engines call this before every update; the client the engine runs stays.
        previous = getattr(self, "_generator", None)
        if previous is not None and previous.worker_id == worker_id:
            return
        if previous is not None:
            previous.close()

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
                model_name=self.model_runner.model_config.model,
                server_url=f"{host}:{port}",
                source_order=(WeightSource.TRAINER,),
                worker_id=worker_id,
                staging_buffer_bytes=staging_buffer_bytes,
                staging_buffers_count=staging_buffers_count,
            )
        )
        atexit.register(self._generator.close)

    @torch.no_grad()
    def update_weights_from_modelexpress(self, version_uid: str) -> None:
        if not version_uid:
            raise ValueError("modelexpress requires version_uid")
        version = WeightVersionRef(version_uid)
        staged = self._generator.stage_weight(version=version)
        try:
            self._generator.apply_weight(staged)
        finally:
            staged.release()
