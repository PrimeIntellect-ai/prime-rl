"""Opt-in per-rank diagnostics for distributed trainer bring-up."""

import json
import logging
import os
import socket
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("trainer_diagnostics_context", default={})
_LOGGER_NAME = "prime_rl.trainer.diagnostics"


def enabled() -> bool:
    return os.environ.get("PRL_CRASH_DIAGNOSTICS") == "1"


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _output_dir(output_dir: Path) -> Path:
    path = output_dir / "diagnostics"
    path.mkdir(parents=True, exist_ok=True)
    return path


def trainer_log_file(output_dir: Path) -> Path | None:
    if not enabled():
        return None
    filename = f"trainer-{socket.gethostname()}-rank{_rank()}-pid{os.getpid()}.log"
    return _output_dir(output_dir) / filename


def setup_diagnostics(output_dir: Path) -> None:
    if not enabled():
        return
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    filename = f"events-{socket.gethostname()}-rank{_rank()}-pid{os.getpid()}.jsonl"
    handler = logging.FileHandler(_output_dir(output_dir) / filename)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    # Torch's TORCH_LOGS output uses stdlib logging rather than prime-rl's Loguru sink.
    python_filename = f"python-{socket.gethostname()}-rank{_rank()}-pid{os.getpid()}.log"
    python_handler = logging.FileHandler(_output_dir(output_dir) / python_filename)
    python_handler.setLevel(logging.DEBUG)
    python_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(python_handler)


@contextmanager
def diagnostic_context(**fields: Any):
    if not enabled():
        yield
        return
    token = _CONTEXT.set({**_CONTEXT.get(), **fields})
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def log_event(event: str, **fields: Any) -> None:
    if not enabled():
        return
    record = {
        "timestamp": time.time(),
        "monotonic": time.monotonic(),
        "event": event,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "rank": _rank(),
        "local_rank": _local_rank(),
        **_CONTEXT.get(),
        **fields,
    }
    logging.getLogger(_LOGGER_NAME).info(json.dumps(record, separators=(",", ":")))


def _read_cgroup_value(path: str) -> int | str | None:
    try:
        value = Path(path).read_text().strip()
    except OSError:
        return None
    return int(value) if value.isdigit() else value


def _resource_snapshot(nvml: Any = None, gpu_handle: Any = None) -> dict[str, Any]:
    import psutil
    import torch

    memory = psutil.virtual_memory()
    snapshot: dict[str, Any] = {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "load_average": list(os.getloadavg()),
        "host_ram_total_bytes": memory.total,
        "host_ram_available_bytes": memory.available,
        "host_ram_used_bytes": memory.used,
        "host_ram_percent": memory.percent,
        "cgroup_memory_current_bytes": _read_cgroup_value("/sys/fs/cgroup/memory.current"),
        "cgroup_memory_max_bytes": _read_cgroup_value("/sys/fs/cgroup/memory.max"),
        "torch_cuda_allocated_bytes": torch.cuda.memory_allocated(),
        "torch_cuda_reserved_bytes": torch.cuda.memory_reserved(),
        "torch_cuda_max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch_cuda_max_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    if nvml is not None and gpu_handle is not None:
        gpu_memory = nvml.nvmlDeviceGetMemoryInfo(gpu_handle)
        utilization = nvml.nvmlDeviceGetUtilizationRates(gpu_handle)
        snapshot.update(
            gpu_memory_total_bytes=gpu_memory.total,
            gpu_memory_used_bytes=gpu_memory.used,
            gpu_memory_free_bytes=gpu_memory.free,
            gpu_utilization_percent=utilization.gpu,
            gpu_memory_utilization_percent=utilization.memory,
        )
    return snapshot


def start_resource_sampler() -> None:
    if not enabled():
        return
    interval = float(os.environ.get("PRL_DIAGNOSTICS_SAMPLE_INTERVAL", "30"))

    def sample() -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(_local_rank())
            nvml = pynvml
        except Exception as error:
            nvml = None
            gpu_handle = None
            log_event("resource_sampler_nvml_unavailable", error_type=type(error).__name__, error=str(error))
        while True:
            try:
                log_event("resources", **_resource_snapshot(nvml, gpu_handle))
            except Exception as error:
                log_event("resource_sample_error", error_type=type(error).__name__, error=str(error))
            time.sleep(interval)

    threading.Thread(target=sample, name="trainer-diagnostics", daemon=True).start()
