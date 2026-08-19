import asyncio
import functools
import importlib
import math
import os
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
import wandb

from prime_rl.utils.logger import get_logger

# TODO: Change all imports to use utils.pathing
# ruff: noqa: F401
from prime_rl.utils.pathing import (
    get_all_ckpt_steps,
    get_broadcast_dir,
    get_ckpt_dir,
    get_eval_dir,
    get_log_dir,
    get_rollout_dir,
    get_step_path,
    get_weights_dir,
    resolve_latest_ckpt_step,
    sync_wait_for_path,
    wait_for_path,
)


def import_object(dotted_path: str) -> Any:
    """Import an object from a dotted path like 'my_module.submodule.MyClass'."""
    module_path, _, name = dotted_path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, name)


def clean_exit(func: Callable) -> Callable:
    """
    A decorator that ensures the a torch.distributed process group is properly
    cleaned up after the decorated function runs or raises an exception.
    """
    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                ret = await func(*args, **kwargs)
            except Exception:
                get_logger().opt(exception=True).error(f"Fatal error in {func.__name__}")
                wandb.finish(exit_code=1)
                # Do NOT destroy the process group on the fatal path: peer ranks are
                # typically still blocked in a collective this rank will never join, so
                # a graceful shutdown hangs in ProcessGroup.shutdown() and the job only
                # dies at the collective timeout — with the watchdog blaming a victim
                # collective on a healthy rank. Hard-exit instead so the launcher's
                # failure propagation (torchrun / srun --kill-on-bad-exit) tears the
                # world down immediately.
                os._exit(1)
            wandb.finish()
            if dist.is_initialized():
                dist.destroy_process_group()
            return ret

        return async_wrapper
    else:

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                ret = func(*args, **kwargs)
            except Exception:
                get_logger().opt(exception=True).error(f"Fatal error in {func.__name__}")
                wandb.finish(exit_code=1)
                # Do NOT destroy the process group on the fatal path: peer ranks are
                # typically still blocked in a collective this rank will never join, so
                # a graceful shutdown hangs in ProcessGroup.shutdown() and the job only
                # dies at the collective timeout — with the watchdog blaming a victim
                # collective on a healthy rank. Hard-exit instead so the launcher's
                # failure propagation (torchrun / srun --kill-on-bad-exit) tears the
                # world down immediately.
                os._exit(1)
            wandb.finish()
            if dist.is_initialized():
                dist.destroy_process_group()
            return ret

        return sync_wrapper


def format_time(time_s: float) -> str:
    """
    Format a time in seconds to a human-readable format:
    - >1d -> Xd Yh
    - >1h -> Xh Ym
    - >1m -> Xm Ys
    - <1s -> Xms
    - Else: Xs
    """
    if time_s >= 86400:
        d = time_s // 86400
        h = (time_s % 86400) // 3600
        return f"{d:.0f}d" + (f" {h:.0f}h" if h > 0 else "")
    elif time_s >= 3600:
        h = time_s // 3600
        m = (time_s % 3600) // 60
        return f"{h:.0f}h" + (f" {m:.0f}m" if m > 0 else "")
    elif time_s >= 60:
        m = time_s // 60
        s = (time_s % 60) // 1
        return f"{m:.0f}m" + (f" {s:.0f}s" if s > 0 else "")
    elif time_s < 1:
        ms = time_s * 1e3
        return f"{ms:.0f}ms"
    else:
        return f"{time_s:.0f}s"


def format_num(num: float | int, precision: int = 2) -> str:
    """
    Format a number in human-readable format with abbreviations.
    """
    sign = "-" if num < 0 else ""
    num = abs(num)
    if num < 1e3:
        return f"{sign}{num:.{precision}f}" if isinstance(num, float) else f"{sign}{num}"
    elif num < 1e6:
        return f"{sign}{num / 1e3:.{precision}f}K"
    elif num < 1e9:
        return f"{sign}{num / 1e6:.{precision}f}M"
    else:
        return f"{sign}{num / 1e9:.{precision}f}B"


def sanitize(obj: Any) -> tuple[Any, list[str]]:
    """Recursively drop non-finite floats (NaN/inf), which are not valid JSON.
    Returns the sanitized object and the dotted paths of the dropped values."""
    dropped_paths: list[str] = []

    def keep(item: Any, path: str) -> bool:
        if isinstance(item, float) and not math.isfinite(item):
            dropped_paths.append(path)
            return False
        return True

    def walk(value: Any, path: str) -> Any:
        if isinstance(value, dict):
            return {
                key: walk(item, child)
                for key, item in value.items()
                if keep(item, child := f"{path}.{key}" if path else key)
            }
        if isinstance(value, list):
            return [walk(item, child) for index, item in enumerate(value) if keep(item, child := f"{path}[{index}]")]
        return value

    return walk(obj, ""), dropped_paths


def get_free_port() -> int:
    """Find and return a free port"""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Bind to any available port
        s.listen(1)
        port = s.getsockname()[1]
    return port


def get_latest_ckpt_step(weights_dir: Path) -> int | None:
    step_dirs = list(weights_dir.glob("step_*"))
    if len(step_dirs) == 0:
        return None
    steps = sorted([int(step_dir.name.split("_")[-1]) for step_dir in step_dirs])
    for latest_step in steps[::-1]:
        if Path(weights_dir / f"step_{latest_step}" / "STABLE").exists():
            return latest_step
    return None


@contextmanager
def default_dtype(dtype):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)
