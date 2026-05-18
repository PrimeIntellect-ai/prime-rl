from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)

_INSTALL_LOCK = threading.Lock()
_INSTALLED = False


def monkey_patch_vllm_padded_input_scrub() -> None:
    """Zero vLLM's padded model inputs before graph replay.

    vLLM pads decode batches up to CUDA graph capture sizes. Some kernels read
    model-input tensors past the scheduled-token prefix, so stale values in the
    padded tail can affect replayed decode graphs. Keep this shim until the
    vLLM-side fix is available in the pinned runtime.
    """
    global _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED:
            return

        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        if getattr(GPUModelRunner, "_prime_rl_padded_input_scrub", False):
            _INSTALLED = True
            return

        original_preprocess = GPUModelRunner._preprocess

        def _patched_preprocess(
            self: Any,
            scheduler_output: Any,
            num_input_tokens: int,
            intermediate_tensors: Any | None = None,
        ) -> tuple[Any, ...]:
            result = original_preprocess(
                self,
                scheduler_output,
                num_input_tokens,
                intermediate_tensors,
            )

            num_scheduled_tokens = int(scheduler_output.total_num_scheduled_tokens)
            _zero_padded_model_inputs(
                result,
                num_scheduled_tokens,
                int(num_input_tokens),
            )
            return result

        GPUModelRunner._preprocess = _patched_preprocess
        GPUModelRunner._prime_rl_padded_input_scrub = True
        _INSTALLED = True

    logger.warning("Enabled vLLM padded model-input scrub.")


def _zero_padded_model_inputs(
    preprocess_result: Sequence[Any],
    num_scheduled_tokens: int,
    num_input_tokens: int,
) -> None:
    """Zero the model-input tail beyond scheduled tokens in-place."""
    if num_input_tokens <= num_scheduled_tokens:
        return

    input_ids = preprocess_result[0]
    inputs_embeds = preprocess_result[1]
    positions = preprocess_result[2]
    pad_slice = slice(num_scheduled_tokens, num_input_tokens)

    if input_ids is not None:
        input_ids[pad_slice].zero_()
    if inputs_embeds is not None:
        inputs_embeds[pad_slice].zero_()
    if positions is not None:
        if positions.ndim == 1:
            positions[pad_slice].zero_()
        else:
            positions[..., pad_slice].zero_()
