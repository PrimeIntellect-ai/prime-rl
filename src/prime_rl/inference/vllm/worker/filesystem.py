from typing import TYPE_CHECKING

from vllm.model_executor.model_loader import get_model_loader

from prime_rl.inference.vllm.worker.fp8_refit import load_checkpoint_weights_layerwise

# This is to get type hints for the Worker class but not actually extend it at runtime as this is required by vLLM worker extension
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object


class FileSystemWeightUpdateWorker(Worker):
    """vLLM worker extension for updating weights in-place using shared filesystem."""

    def init_broadcaster(self) -> None:
        """Initialize the broadcaster."""
        ...

    def update_weights_from_path(self, weight_path: str) -> None:
        """Update weights from a specified path in shared filesystem containing a HF-compatible checkpoint."""
        model = self.model_runner.get_model()
        if hasattr(model, "runnable"):
            model = model.runnable
        load_config = self.model_runner.vllm_config.load_config

        model_loader = get_model_loader(load_config)
        model_config = self.model_runner.model_config
        original_model_path = model_config.model
        model_config.model = weight_path
        weights_iter = model_loader.get_all_weights(model_config, model)
        load_checkpoint_weights_layerwise(self.model_runner, model, weights_iter)
        model_config.model = original_model_path
