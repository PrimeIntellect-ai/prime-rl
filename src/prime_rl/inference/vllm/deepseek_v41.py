from collections.abc import Iterable

import torch
from vllm.models.deepseek_v41.nvidia.model import DeepseekV41LLMForCausalLM

# Checkpoint weights of the vision tower and the image tokens, which the text model has no place for.
VISION_PREFIXES = ("vision.", "aligner.", "image_start", "image_end", "image_newline")


class DeepseekV41TextForCausalLM(DeepseekV41LLMForCausalLM):
    """DeepSeek-V4.1 served as a text-only model, in place of vLLM's vision-language wrapper.

    The wrapper's `load_weights` sorts every weight before loading any, so a weight update streamed
    from the trainer would sit whole on the GPU. The text model loads weights as they arrive.
    """

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        self._weights_finalized = False
        loaded = super().load_weights(
            (name, weight) for name, weight in weights if not name.startswith(VISION_PREFIXES)
        )
        self._weights_finalized = True
        return loaded

    def process_weights_after_loading(self) -> None:
        # `load_weights` finalizes the weights itself; the loader's model-level hook would do it twice.
        if not getattr(self, "_weights_finalized", False):
            super().process_weights_after_loading()
