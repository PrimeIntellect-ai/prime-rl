from typing import TYPE_CHECKING, cast

import torch
from modelexpress_rl import (
    ModelExpressGeneratorClient,
    ModelExpressGeneratorConfig,
    VllmGeneratorContext,
    WeightSource,
    WeightVersionRef,
)
from torch.nn import Module

from prime_rl.transports.weights.mx_phases import timed_refit
from prime_rl.transports.weights.mx_rdma import apply_rdma_defaults


def _step_of(version_uid: str) -> int:
    """Parse the step suffix from a ModelExpress version ID."""
    _, _, suffix = version_uid.rpartition(":")
    return int(suffix) if suffix.isdigit() else -1


if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object


class MXRefitUpdateWorker(Worker):
    """vLLM worker extension for ModelExpress weight updates."""

    def init_broadcaster(self, mx_server_host: str, mx_server_port: int, *args) -> None:
        del args  # unused extras from the shared init_broadcaster route
        model = cast(Module, self.model_runner.get_model())
        # TODO: Move PrimeRL-to-HF conversion to the trainer.
        from prime_rl.trainer.models import get_custom_causal_lm_cls
        from prime_rl.trainer.models.conversion_ops import apply_prime_to_hf

        hf_config = self.model_runner.model_config.hf_config
        conversion_chain = get_custom_causal_lm_cls(hf_config).conversion_chain(hf_config)
        convert_native_to_hf = (lambda sd: apply_prime_to_hf(sd, conversion_chain)) if conversion_chain else None
        # The NIXL agent reads the rail configuration during initialization.
        apply_rdma_defaults()
        self._generator = ModelExpressGeneratorClient.initialize(
            ModelExpressGeneratorConfig(
                engine_context=VllmGeneratorContext(
                    model=model,
                    vllm_config=self.vllm_config,
                    convert_native_to_hf=convert_native_to_hf,
                ),
                model_name=self.model_runner.model_config.model,
                server_url=f"{mx_server_host}:{mx_server_port}",
                source_order=(WeightSource.TRAINER,),
            )
        )

    def liveness_probe(self) -> None:
        return None

    @torch.no_grad()
    def update_weights_from_path(self, weight_dir: str | None = None, version_uid: str | None = None) -> None:
        del weight_dir  # mx_refit pulls by version, not a path
        if version_uid is None:
            raise ValueError("mx_refit update_weights requires version_uid")
        with timed_refit("generator", _step_of(version_uid), version_uid) as timer:
            with timer.phase("wire"):
                staged = self._generator.stage_weight(version=WeightVersionRef(version_uid))
            try:
                with timer.phase("install"):
                    self._generator.apply_weight(staged)
            finally:
                with timer.phase("release"):
                    staged.release()
