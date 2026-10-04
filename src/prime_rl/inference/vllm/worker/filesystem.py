from pathlib import Path
from typing import TYPE_CHECKING

from torch.nn import Module
from vllm.model_executor.model_loader import DefaultModelLoader, get_model_loader

from prime_rl.inference.vllm.worker.weight_transfer import (
    load_sparse_delta_weights,
    load_weights_checkpoint_layerwise,
    update_mla_absorbed_weights,
)

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
        """Update weights from a specified path in shared filesystem containing a HF-compatible checkpoint."""
        model = self._model()

        # Get vLLM model loader
        model_loader = get_model_loader(self.load_config)
        assert isinstance(model_loader, DefaultModelLoader)
        local_source = DefaultModelLoader.Source(
            weight_path,
            revision=None,  # TODO: Check that this is correct or if we should use the default (model_config.revision)
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
            allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
        )
        weights_iterator = model_loader._get_weights_iterator(local_source)
        load_weights_checkpoint_layerwise(
            model,
            weights_iterator,
            self.model_runner.model_config,
            self.vllm_config,
        )

    def update_weights_from_delta_path(self, delta_path: str) -> None:
        """Apply a sparse delta artifact to the current vLLM model weights."""
        parallel = self.vllm_config.parallel_config
        if parallel.pipeline_parallel_size != 1 or parallel.enable_expert_parallel:
            raise ValueError("sparse deltas require PP=1 and no expert parallelism")
        if self.model_runner.model_config.quantization is not None:
            raise ValueError("sparse deltas do not support quantized inference weights")
        path = Path(delta_path)
        if path.is_dir():
            stream_path = path / "delta.stream"
            path = stream_path if stream_path.exists() else path / "delta.safetensors"
        model = self._model()
        load_sparse_delta_weights(model, path.as_posix())
        update_mla_absorbed_weights(model)

    def _model(self) -> Module:
        model_runner = self.model_runner
        if hasattr(model_runner.model, "runnable"):
            model = model_runner.model.runnable
        else:
            model = model_runner.model
        assert isinstance(model, Module)
        return model
