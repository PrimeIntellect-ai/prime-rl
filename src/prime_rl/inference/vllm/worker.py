from pathlib import Path

import torch


class CheckpointWorker:
    """
    This is an extension of a vLLM worker that allows for loading checkpoints
    from a specified directory via RPC calls from the AsyncLLMEngine class, exposed
    by the vLLM server. This is useful in RL training, where we want to load the
    recent policy model from a checkpoint directory.
    """

    def reload_weights(self, model_path: Path) -> None:
        """Reload the weights from a specified path."""
        state_dict = torch.load(model_path, map_location="cpu", mmap=True)

        def weights_iterator():
            for key, value in state_dict.items():
                if not key:
                    continue
                yield key, value

        self.model_runner.model.load_weights(weights_iterator())

        # Process weights after loading (important for some models)
        from vllm.model_executor.model_loader.utils import process_weights_after_loading

        device = next(self.model_runner.model.parameters()).device
        process_weights_after_loading(self.model_runner.model, self.model_runner.model_config, device)

    def reset_weights(self) -> None:
        self.load_model()
