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

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker
else:
    Worker = object


class ModelExpressWeightUpdateWorker(Worker):
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
        from prime_rl.trainer.models import get_custom_causal_lm_cls
        from prime_rl.trainer.models.conversion_ops import apply_prime_to_hf

        hf_config = self.model_runner.model_config.hf_config
        chain = get_custom_causal_lm_cls(hf_config).conversion_chain(hf_config)
        worker_id = f"{session_id}:{rank_offset + self.rank}"
        self._generator = ModelExpressGeneratorClient.initialize(
            ModelExpressGeneratorConfig(
                engine_context=VllmGeneratorContext(
                    model=cast(nn.Module, self.model_runner.get_model()),
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
