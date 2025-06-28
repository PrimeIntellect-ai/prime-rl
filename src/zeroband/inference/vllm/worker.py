from pathlib import Path

from safetensors.torch import safe_open


class CheckpointWorker:
    """
    This is an extension of a vLLM worker that allows for loading checkpoints
    from a specified directory via RPC calls from the AsyncLLMEngine class, exposed
    by the vLLM server. This is useful in RL training, where we want to load the
    recent policy model from a checkpoint directory.
    """

    def test_rpc(self):
        """A simple method to test if the RPC call is working."""
        print("RPC call test successful")
        return True

    def load_checkpoint(self, ckpt_path: Path) -> None:
        """Load a checkpoint from a specified directory."""
        with safe_open(ckpt_path, framework="pt", device="cpu") as f:

            def weights_iterator():
                for key in f.keys():
                    if not key:  # Skip empty keys
                        continue
                    yield key, f.get_tensor(key)

            # Load weights from custom weight iterator
            self.model_runner.model.load_weights(weights_iterator())

        # Process weights after loading (important for some models)
        from vllm.model_executor.model_loader.utils import process_weights_after_loading

        device = next(self.model_runner.model.parameters()).device
        process_weights_after_loading(
            self.model_runner.model, self.model_runner.model_config, device
        )
