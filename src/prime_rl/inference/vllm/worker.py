from typing import TYPE_CHECKING

from torch.nn import Module
from vllm.model_executor.model_loader import DefaultModelLoader, get_model_loader
from vllm.model_executor.model_loader.utils import process_weights_after_loading

# This is to get type hints for the Worker class but not actually extend it at runtime as this is required by vLLM worker extension
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object


class CheckpointWorker(Worker):
    """
    This is an extension of a vLLM worker that allows for loading checkpoints
    from a specified directory via RPC calls from the AsyncLLMEngine class, exposed
    by the vLLM server. This is useful in RL training, where we want to load the
    recent policy model from a checkpoint directory.
    """

    def update_weights(self, weight_path: str) -> None:
        """Update weights from a specified path pointing to a .pt file."""
        # Get vLLM model runner and model
        model_runner = self.model_runner
        model = model_runner.model
        assert isinstance(model, Module)

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
        model.load_weights(weights_iterator)  # type: ignore

        # Process weights after loading (important for some models)
        device = next(model.parameters()).device
        process_weights_after_loading(model, self.model_runner.model_config, device)
