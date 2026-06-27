"""Durable capture for vLLM's post-warmup Triton JIT monitor."""

import datetime
import json
import logging
import os
import socket
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MAX_DETAIL_REPR_CHARS = 1000
_DETAIL_KEYS = ("key", "repr", "already_compiled", "is_manual_warmup", "compile")


@lru_cache(maxsize=1)
def _log_path() -> Path | None:
    explicit = os.environ.get("PRIME_RL_JIT_MONITOR_LOG")
    if explicit:
        return Path(explicit)

    log_dir = os.environ.get("PRIME_RL_JIT_MONITOR_LOG_DIR")
    if not log_dir:
        return None

    host = socket.gethostname().split(".", 1)[0]
    rank = os.environ.get("INFER_NODE_RANK") or os.environ.get("SLURM_PROCID") or os.environ.get("RANK") or "unknown"
    local_rank = os.environ.get("LOCAL_RANK") or os.environ.get("SLURM_LOCALID") or "unknown"
    return Path(log_dir) / f"jit_monitor_{host}_rank{rank}_local{local_rank}_pid{os.getpid()}.jsonl"


def _kernel_name(kwargs: dict[str, Any]) -> str:
    fn = kwargs.get("fn")
    return str(getattr(fn, "name", None) or getattr(fn, "__name__", None) or "<unknown>")


def _json_safe_detail(value: Any) -> bool | int | float | str | None:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    text = repr(value)
    if len(text) > _MAX_DETAIL_REPR_CHARS:
        return text[: _MAX_DETAIL_REPR_CHARS - 3] + "..."
    return text


def _compile_details(kwargs: dict[str, Any]) -> dict[str, bool | int | float | str | None]:
    return {key: _json_safe_detail(kwargs[key]) for key in _DETAIL_KEYS if key in kwargs}


def _append_jit_event(kwargs: dict[str, Any]) -> None:
    path = _log_path()
    if path is None:
        return

    record = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "event": "triton_jit_compile_during_inference",
        "kernel": _kernel_name(kwargs),
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "infer_node_rank": os.environ.get("INFER_NODE_RANK"),
        "rank": os.environ.get("RANK"),
        "local_rank": os.environ.get("LOCAL_RANK") or os.environ.get("SLURM_LOCALID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "kwargs": sorted(kwargs),
        "details": _compile_details(kwargs),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError as exc:
        logger.warning("Failed to write Triton JIT monitor event to %s: %s", path, exc)


def _wrap_current_hook() -> None:
    from vllm.triton_utils.importing import HAS_TRITON

    if not HAS_TRITON:
        return

    from triton import knobs  # type: ignore[import-untyped]

    existing_hook = knobs.runtime.jit_post_compile_hook
    if getattr(existing_hook, "_prime_rl_jit_file_logger", False):
        return

    def _prime_rl_jit_file_logger(**kwargs):
        _append_jit_event(kwargs)
        if existing_hook is not None:
            return existing_hook(**kwargs)
        return None

    _prime_rl_jit_file_logger._prime_rl_jit_file_logger = True
    knobs.runtime.jit_post_compile_hook = _prime_rl_jit_file_logger


def apply_jit_monitor_file_logging_patch() -> None:
    """Append vLLM Triton JIT monitor events to JSONL files when configured.

    vLLM's own monitor logs through stdlib logging. On multi-process Slurm
    launches those worker warnings can appear in live stdout but fail to land in
    the persisted rank log. This keeps the upstream warning behavior and adds a
    small file-side audit trail keyed by worker process.
    """

    if _log_path() is None:
        return

    import vllm.triton_utils.jit_monitor as jit_monitor

    if getattr(jit_monitor, "_prime_rl_jit_file_logging_patched", False):
        return

    original_setup = jit_monitor._setup_triton_jit_hook

    def _patched_setup_triton_jit_hook() -> None:
        original_setup()
        _wrap_current_hook()

    jit_monitor._setup_triton_jit_hook = _patched_setup_triton_jit_hook
    jit_monitor._prime_rl_jit_file_logging_patched = True

    if jit_monitor.is_active():
        _wrap_current_hook()
