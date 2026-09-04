from typing import TYPE_CHECKING

from vllm.logger import init_logger

logger = init_logger(__name__)

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

    def liveness_probe(self) -> None:
        """No-op RPC used by the API server liveness endpoint."""
        return None

    def update_weights_from_path(self, weight_path: str) -> None:
        """Update weights from a specified path containing an HF-compatible checkpoint."""
        logger.info("Reloading filesystem weights from %s", weight_path)
        self.reload_weights(weights_path=weight_path)
        logger.info("Reloaded filesystem weights from %s", weight_path)
