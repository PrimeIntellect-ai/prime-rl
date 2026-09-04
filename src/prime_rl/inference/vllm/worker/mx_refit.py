from typing import TYPE_CHECKING, cast

import torch
from modelexpress_rl import (
    ModelExpressGeneratorClient,
    ModelExpressGeneratorConfig,
    VllmGeneratorContext,
    WeightVersionRef,
)
from torch.nn import Module

from prime_rl.transports.weights.mx_rdma import apply_rdma_defaults

# Type hints for the Worker class without extending it at runtime, as required by
# the vLLM worker-extension mechanism.
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object


class MXRefitUpdateWorker(Worker):
    """vLLM worker extension that pulls and installs weights through ModelExpress's
    generator client: it stages, applies, and releases the version_uid it is given."""

    def init_broadcaster(self, mx_server_host: str, mx_server_port: int, *args) -> None:
        del args  # unused extras from the shared init_broadcaster route
        model = cast(Module, self.model_runner.get_model())
        # TODO: check if this prime-native -> HF conversion can move to the trainer
        # side to decouple trainer and inference (avoids importing trainer models here).
        from prime_rl.trainer.models import get_custom_causal_lm_cls
        from prime_rl.trainer.models.conversion_ops import apply_prime_to_hf

        hf_config = self.model_runner.model_config.hf_config
        conversion_chain = get_custom_causal_lm_cls(hf_config).conversion_chain(hf_config)
        convert_native_to_hf = (lambda sd: apply_prime_to_hf(sd, conversion_chain)) if conversion_chain else None
        # The receiver side is where rail convergence actually bites: every
        # inference worker on a node is a separate process pulling concurrently,
        # and they will pick the same NIC unless something assigns them globally.
        # Must precede client construction - the pin is read when the NIXL agent
        # initialises.
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
            )
        )

    def liveness_probe(self) -> None:
        """No-op RPC used by the API server liveness endpoint."""
        return None

    @torch.no_grad()
    def update_weights_from_path(self, weight_dir: str | None = None, version_uid: str | None = None) -> None:
        del weight_dir  # mx_refit pulls by version, not a path
        if version_uid is None:
            raise ValueError("mx_refit update_weights requires version_uid")
        staged = self._generator.stage_weight(version=WeightVersionRef(version_uid))
        try:
            self._generator.apply_weight(staged)
        finally:
            staged.release()
