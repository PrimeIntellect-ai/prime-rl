from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_INSTALL_LOCK = threading.Lock()
_PATCHED = False
_WRITE_LOCK = threading.Lock()


def enabled() -> bool:
    return os.environ.get("PRIME_RL_VLLM_ZERO_PADDED_INPUTS", "0") != "0"


def install_padded_input_scrub() -> None:
    """Zero padded model inputs before FULL CUDA graph replay.

    vLLM already zeroes padded positions in GPUModelRunner._preprocess(), but
    text-only input_ids past scheduler_output.total_num_scheduled_tokens retain
    prior buffer contents. This diagnostic patch keeps default FULL graph
    dispatch enabled while making those padded token rows deterministic.
    """
    if not enabled():
        return

    with _INSTALL_LOCK:
        _patch_gpu_model_runner()


def _patch_gpu_model_runner() -> None:
    global _PATCHED
    if _PATCHED:
        return

    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception:
        logger.exception("Failed to import vLLM GPUModelRunner for padded input scrub")
        return

    original_preprocess = getattr(GPUModelRunner, "_preprocess", None)
    if original_preprocess is None:
        return

    def _patched_preprocess(self, scheduler_output, num_input_tokens: int, *args, **kwargs):
        result = original_preprocess(self, scheduler_output, num_input_tokens, *args, **kwargs)
        try:
            num_scheduled_tokens = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0))
            num_input_tokens_int = int(num_input_tokens)
            if num_input_tokens_int > num_scheduled_tokens:
                input_ids, inputs_embeds, positions, *_ = result
                if input_ids is not None:
                    input_ids[num_scheduled_tokens:num_input_tokens_int].zero_()
                if inputs_embeds is not None:
                    inputs_embeds[num_scheduled_tokens:num_input_tokens_int].zero_()
                if positions is not None:
                    positions[..., num_scheduled_tokens:num_input_tokens_int].zero_()
                _trace_scrub_record(num_scheduled_tokens, num_input_tokens_int, result)
        except Exception:
            logger.exception("Failed to scrub padded vLLM model inputs")
        return result

    GPUModelRunner._preprocess = _patched_preprocess
    _PATCHED = True
    logger.warning("Enabled padded model input scrub.")


def _trace_scrub_record(num_scheduled_tokens: int, num_input_tokens: int, result: Any) -> None:
    trace_dir = os.environ.get("PRIME_RL_VLLM_NAN_TRACE_DIR")
    if not trace_dir:
        return
    try:
        input_ids, inputs_embeds, positions, *_ = result
        record = {
            "schema": "prime_rl.vllm_padded_input_scrub.v1",
            "timestamp": time.time(),
            "num_scheduled_tokens": num_scheduled_tokens,
            "num_input_tokens": num_input_tokens,
            "num_padded_tokens": num_input_tokens - num_scheduled_tokens,
            "input_ids": _tensor_summary(input_ids),
            "inputs_embeds": _tensor_summary(inputs_embeds),
            "positions": _tensor_summary(positions),
        }
        _write_jsonl(Path(trace_dir), "padded_input_scrub", record)
    except Exception:
        logger.exception("Failed to trace padded input scrub")


def _tensor_summary(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "shape": list(getattr(value, "shape", ())),
        "dtype": str(getattr(value, "dtype", None)),
        "device": str(getattr(value, "device", None)),
    }


def _write_jsonl(trace_dir: Path, stem: str, record: dict[str, Any]) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"{stem}.{os.getpid()}.jsonl"
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")
