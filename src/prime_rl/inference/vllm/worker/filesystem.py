from typing import TYPE_CHECKING

from vllm.model_executor.model_loader import get_model_loader

from .fp8_refit import load_checkpoint_weights_layerwise, unwrap_worker_model

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
        model = unwrap_worker_model(self.model_runner.get_model())
        model_loader = get_model_loader(self.model_runner.load_config)
        if not hasattr(model_loader, "get_all_weights"):
            raise NotImplementedError(
                f"Model reloading with `{self.model_runner.load_config.load_format}` format"
            )

        self.model_runner.model_config.model = weight_path
        weights_iter = model_loader.get_all_weights(self.model_runner.model_config, model)
        load_checkpoint_weights_layerwise(self.model_runner, model, weights_iter)
