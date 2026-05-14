"""Lightweight GPU memory diagnostics for BlenderGym env workers.

Writes to its own ``gpu_mem.log`` file in the same directory as
``env_worker_N.log``, so it gets picked up by the S3 rsync automatically.
"""

from __future__ import annotations

import logging
import os
import subprocess
from datetime import datetime, timezone

import torch

_log_file = None
_initialized = False


def _init() -> None:
    global _log_file, _initialized
    if _initialized:
        return
    _initialized = True

    log_dir = None
    for name in (
        "verifiers.utils.env_utils",
        "verifiers.serve.server.env_worker.EnvWorker",
        "verifiers",
    ):
        for h in logging.getLogger(name).handlers:
            if isinstance(h, logging.FileHandler) and h.baseFilename:
                log_dir = os.path.dirname(h.baseFilename)
                break
        if log_dir:
            break

    if not log_dir:
        return

    try:
        _log_file = open(os.path.join(log_dir, "gpu_mem.log"), "a")
    except Exception:
        pass


def _write(msg: str) -> None:
    _init()
    if _log_file is None:
        return
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    _log_file.write(f"{ts} - {msg}\n")
    _log_file.flush()


def log_gpu_mem(gpu_id: int, label: str, *, empty_cache: bool = False) -> None:
    """Log GPU memory state.  Optionally call ``empty_cache`` first."""
    try:
        if not torch.cuda.is_available():
            return
        if empty_cache:
            before = torch.cuda.memory_reserved(gpu_id)
            torch.cuda.empty_cache()
            after = torch.cuda.memory_reserved(gpu_id)
            freed = (before - after) / 1048576
            _write(f"[GPU {gpu_id}] {label}: empty_cache freed {freed:.1f} MiB reserved")
        alloc = torch.cuda.memory_allocated(gpu_id) / 1048576
        reserved = torch.cuda.memory_reserved(gpu_id) / 1048576
        _write(f"[GPU {gpu_id}] {label}: torch alloc={alloc:.1f} MiB, reserved={reserved:.1f} MiB")
    except Exception:
        pass

    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free",
                "--format=csv,noheader,nounits",
                f"--id={gpu_id}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0:
            _write(f"[GPU {gpu_id}] {label}: nvidia-smi {r.stdout.strip()} (used,free MiB)")
    except Exception:
        pass
