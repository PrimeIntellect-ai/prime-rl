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
_DISPATCHER_PATCHED = False
_MODEL_RUNNER_PATCHED = False
_WRITE_LOCK = threading.Lock()


def enabled() -> bool:
    return os.environ.get("PRIME_RL_VLLM_DISABLE_PADDED_FULL_FOR_MAMBA", "0") != "0"


def install_padded_full_mamba_guard() -> None:
    """Avoid padded FULL decode CUDA graph replay for Mamba/hybrid models.

    This is intentionally a narrow vLLM plugin patch. Exact-width FULL decode
    graphs still run as FULL; only Mamba batches padded upward to a larger FULL
    graph descriptor are re-dispatched with FULL excluded.
    """
    if not enabled():
        return

    with _INSTALL_LOCK:
        _patch_model_runner()
        _patch_cudagraph_dispatcher()


def _patch_model_runner() -> None:
    global _MODEL_RUNNER_PATCHED
    if _MODEL_RUNNER_PATCHED:
        return

    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception:
        logger.exception("Failed to import vLLM GPUModelRunner for padded FULL Mamba guard")
        return

    original_determine_batch = getattr(GPUModelRunner, "_determine_batch_execution_and_padding", None)
    if original_determine_batch is None:
        return

    def _patched_determine_batch_execution_and_padding(self, *args, **kwargs):
        try:
            dispatcher = getattr(self, "cudagraph_dispatcher", None)
            if dispatcher is not None:
                dispatcher._prime_rl_has_mamba_layers = _model_runner_has_mamba_layers(self)
        except Exception:
            logger.exception("Failed to mark vLLM dispatcher Mamba state")
        return original_determine_batch(self, *args, **kwargs)

    GPUModelRunner._determine_batch_execution_and_padding = _patched_determine_batch_execution_and_padding
    _MODEL_RUNNER_PATCHED = True


def _patch_cudagraph_dispatcher() -> None:
    global _DISPATCHER_PATCHED
    if _DISPATCHER_PATCHED:
        return

    try:
        from vllm.config import CUDAGraphMode
        from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
    except Exception:
        logger.exception("Failed to import vLLM CudagraphDispatcher for padded FULL Mamba guard")
        return

    original_dispatch = CudagraphDispatcher.dispatch

    def _patched_dispatch(self, *args, **kwargs):
        result = original_dispatch(self, *args, **kwargs)
        try:
            cudagraph_mode, batch_descriptor = result
            if cudagraph_mode != CUDAGraphMode.FULL:
                return result
            if not getattr(self, "_prime_rl_has_mamba_layers", False):
                return result

            num_tokens = _dispatch_num_tokens(args, kwargs)
            descriptor_tokens = getattr(batch_descriptor, "num_tokens", None)
            if num_tokens is None or descriptor_tokens is None:
                return result
            if int(descriptor_tokens) <= int(num_tokens):
                return result

            patched_args = list(args)
            patched_kwargs = dict(kwargs)
            if len(patched_args) >= 6:
                invalid_modes = set(patched_args[5] or ())
                patched_args[5] = invalid_modes
            else:
                invalid_modes = set(patched_kwargs.get("invalid_modes") or ())
                patched_kwargs["invalid_modes"] = invalid_modes
            invalid_modes.add(CUDAGraphMode.FULL)

            fallback = original_dispatch(self, *patched_args, **patched_kwargs)
            _trace_guard_record(
                requested_num_tokens=int(num_tokens),
                original_result=result,
                fallback_result=fallback,
                args=args,
                kwargs=kwargs,
            )
            return fallback
        except Exception:
            logger.exception("Failed to apply padded FULL Mamba CUDA graph guard")
            return result

    CudagraphDispatcher.dispatch = _patched_dispatch
    _DISPATCHER_PATCHED = True
    logger.warning("Enabled padded FULL Mamba CUDA graph guard.")


def _dispatch_num_tokens(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int | None:
    if "num_tokens" in kwargs:
        try:
            return int(kwargs["num_tokens"])
        except (TypeError, ValueError):
            return None
    if args:
        try:
            return int(args[0])
        except (TypeError, ValueError):
            return None
    return None


def _model_runner_has_mamba_layers(model_runner: Any) -> bool:
    kv_cache_config = getattr(model_runner, "kv_cache_config", None)
    if bool(getattr(kv_cache_config, "has_mamba_layers", False)):
        return True

    groups = getattr(kv_cache_config, "kv_cache_groups", None) or []
    for group in groups:
        spec = getattr(group, "kv_cache_spec", None)
        if spec is not None and spec.__class__.__name__ == "MambaSpec":
            return True
    return False


def _trace_guard_record(
    *,
    requested_num_tokens: int,
    original_result: Any,
    fallback_result: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    trace_dir = os.environ.get("PRIME_RL_VLLM_NAN_TRACE_DIR")
    if not trace_dir:
        return
    try:
        original_mode, original_desc = original_result
        fallback_mode, fallback_desc = fallback_result
    except Exception:
        return

    _write_jsonl(
        Path(trace_dir),
        "padded_full_mamba_guard",
        {
            "schema": "prime_rl.vllm_padded_full_mamba_guard.v1",
            "timestamp": time.time(),
            "requested_num_tokens": requested_num_tokens,
            "original_mode": _enum_summary(original_mode),
            "original_batch_descriptor": _object_attr_summary(
                original_desc,
                ("num_tokens", "num_reqs", "uniform", "has_lora", "num_active_loras"),
            ),
            "fallback_mode": _enum_summary(fallback_mode),
            "fallback_batch_descriptor": _object_attr_summary(
                fallback_desc,
                ("num_tokens", "num_reqs", "uniform", "has_lora", "num_active_loras"),
            ),
            "dispatch_inputs": {
                "uniform_decode": _json_safe(kwargs.get("uniform_decode")),
                "has_lora": _json_safe(kwargs.get("has_lora")),
                "num_active_loras": _json_safe(kwargs.get("num_active_loras")),
                "valid_modes": _json_safe(kwargs.get("valid_modes")),
                "invalid_modes": _json_safe(kwargs.get("invalid_modes")),
                "positional_arg_count": len(args),
            },
        },
    )


def _write_jsonl(trace_dir: Path, stem: str, record: dict[str, Any]) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"{stem}.{os.getpid()}.jsonl"
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(record), sort_keys=True))
            handle.write("\n")


def _enum_summary(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "name": getattr(value, "name", None),
        "value": getattr(value, "value", None),
        "string": str(value),
        "type": value.__class__.__name__,
    }


def _object_attr_summary(value: Any, attrs: tuple[str, ...]) -> dict[str, Any] | None:
    if value is None:
        return None
    return {attr: _json_safe(getattr(value, attr, None)) for attr in attrs}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        if value != value:
            return "NaN"
        if value == float("inf"):
            return "Infinity"
        if value == float("-inf"):
            return "-Infinity"
        return value
    if isinstance(value, dict):
        return {str(_json_safe(key)): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    try:
        return str(value)
    except Exception:
        return f"<{value.__class__.__name__}>"
