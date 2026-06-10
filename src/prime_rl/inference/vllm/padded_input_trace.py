from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from typing import Any

from prime_rl.utils.nan_trace import check_finite, tensor_summary, write_event

FALSE_VALUES = {"", "0", "false", "False", "no", "off"}

_INSTALL_LOCK = threading.Lock()
_INSTALLED = False
_EVENT_COUNT = 0


def _enabled() -> bool:
    return os.environ.get("PRIME_PADDED_INPUT_TRACE", "") not in FALSE_VALUES


def _event_limit() -> int:
    value = os.environ.get("PRIME_PADDED_INPUT_TRACE_LIMIT", "512")
    try:
        return max(int(value), 0)
    except ValueError:
        return 512


def monkey_patch_vllm_padded_input_trace() -> None:
    """Trace vLLM padded model-input tails without mutating them.

    This is intentionally not the scrub/fix from later prime-rl branches. It only
    records padded-tail summaries so exact historical runs can show whether the
    stale-padding failure class is present.
    """
    if not _enabled():
        return

    global _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED:
            return

        try:
            from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        except Exception as exc:
            write_event("vllm_padded_input_trace_install_error", error=repr(exc))
            _INSTALLED = True
            return

        if getattr(GPUModelRunner, "_prime_rl_padded_input_trace", False):
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
            _trace_padded_model_inputs(
                result,
                int(getattr(scheduler_output, "total_num_scheduled_tokens", num_input_tokens)),
                int(num_input_tokens),
            )
            return result

        GPUModelRunner._preprocess = _patched_preprocess
        GPUModelRunner._prime_rl_padded_input_trace = True
        _INSTALLED = True

    write_event("vllm_padded_input_trace_installed")


def _tail_summary(tensor: Any, pad_slice: slice) -> dict[str, Any] | None:
    if tensor is None:
        return None
    try:
        if getattr(tensor, "ndim", 0) == 1:
            tail = tensor[pad_slice]
        else:
            tail = tensor[..., pad_slice]
        return tensor_summary(tail)
    except Exception as exc:
        return {"error": repr(exc), "repr": repr(tensor)}


def _check_tail(name: str, tensor: Any, pad_slice: slice, **context: Any) -> None:
    if tensor is None:
        return
    try:
        tail = tensor[pad_slice] if getattr(tensor, "ndim", 0) == 1 else tensor[..., pad_slice]
    except Exception as exc:
        write_event("vllm_padded_input_tail_slice_error", name=name, error=repr(exc), context=context)
        return
    check_finite(name, tail, **context)


def _trace_padded_model_inputs(
    preprocess_result: Sequence[Any],
    num_scheduled_tokens: int,
    num_input_tokens: int,
) -> None:
    global _EVENT_COUNT
    if num_input_tokens <= num_scheduled_tokens:
        return
    if _EVENT_COUNT >= _event_limit():
        return
    _EVENT_COUNT += 1

    input_ids = preprocess_result[0] if len(preprocess_result) > 0 else None
    inputs_embeds = preprocess_result[1] if len(preprocess_result) > 1 else None
    positions = preprocess_result[2] if len(preprocess_result) > 2 else None
    pad_slice = slice(num_scheduled_tokens, num_input_tokens)
    context = {
        "num_scheduled_tokens": num_scheduled_tokens,
        "num_input_tokens": num_input_tokens,
        "tail_tokens": num_input_tokens - num_scheduled_tokens,
    }

    write_event(
        "vllm_padded_input_tail",
        **context,
        input_ids_tail=_tail_summary(input_ids, pad_slice),
        inputs_embeds_tail=_tail_summary(inputs_embeds, pad_slice),
        positions_tail=_tail_summary(positions, pad_slice),
    )
    _check_tail("vllm.padded_inputs.input_ids_tail", input_ids, pad_slice, **context)
    _check_tail("vllm.padded_inputs.inputs_embeds_tail", inputs_embeds, pad_slice, **context)
    _check_tail("vllm.padded_inputs.positions_tail", positions, pad_slice, **context)
