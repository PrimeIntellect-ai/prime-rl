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
        """Push: the orchestrator installs every version on every engine at once, so only the trainer has it.
        ModelExpress identifies and places every worker itself, so the shared broadcast arguments carry nothing
        it needs."""
        del rank_offset, inference_world_size, timeout, session_id
        self._init_generator(
            model_name=self.model_runner.model_config.model,
            server_url=f"{host}:{port}",
            source_order=(WeightSource.TRAINER,),
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
        self._generator = ModelExpressGeneratorClient.initialize(
            ModelExpressGeneratorConfig(
                engine_context=VllmGeneratorContext(
                    model=cast(nn.Module, self.model_runner.get_model()),
                    vllm_config=self.vllm_config,
                    convert_native_to_hf=(lambda tensors: apply_prime_to_hf(tensors, chain)) if chain else None,
                    # Pull installs through vLLM's own loader, as NCCL does, so it refits what vLLM can reload.
                    install_through_loader=install_through_loader,
                ),
                model_name=model_name,
                server_url=server_url,
                source_order=source_order,
                reader_group=reader_group,
                staging_buffer_bytes=staging_buffer_bytes,
                staging_buffers_count=staging_buffers_count,
                # vLLM's loaders read host tensors, as from safetensors, and copy only this rank's slices.
                staging_device="cpu" if install_through_loader else None,
            )
        )
        atexit.register(self._generator.close)
        self._staged = None
