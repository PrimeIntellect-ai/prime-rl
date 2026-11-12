from typing import TYPE_CHECKING

from prime_rl.inference.vllm.worker.weight_transfer import (
    WeightRPCMixin,
    checkpoint_weights,
    get_weight_model,
    load_weights_checkpoint_layerwise,
)

# This is to get type hints for the Worker class but not actually extend it at runtime as this is required by vLLM worker extension
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object


class FileSystemWeightUpdateWorker(WeightRPCMixin, Worker):
    """vLLM worker extension for updating weights in-place using shared filesystem."""

    def liveness_probe(self) -> None:
        """No-op RPC used by the API server liveness endpoint."""
        return None

    def update_weights_from_path(self, weight_path: str) -> None:
        """Update weights from a specified path in shared filesystem containing a HF-compatible checkpoint."""
        model = get_weight_model(self.model_runner, "model")
        weights = checkpoint_weights(model, weight_path, self.load_config)
        load_weights_checkpoint_layerwise(model, weights, self.model_runner.model_config, self.vllm_config)
