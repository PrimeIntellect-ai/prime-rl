import contextlib
import os
import signal
import subprocess
from subprocess import Popen
from threading import Event, Thread

import psutil
import setproctitle

from prime_rl.utils.logger import get_logger

PRIME_RL_PROC_PREFIX = "PRIME-RL"


# Applied to every launched component (trainer, orchestrator, inference).
DEFAULT_COMMON_ENV_VARS: dict[str, str] = {
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "PYTHONUNBUFFERED": "1",
    "OMP_NUM_THREADS": "1",
    "GIT_LFS_SKIP_SMUDGE": "1",
}

DEFAULT_TRAINER_ENV_VARS: dict[str, str] = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}

DEFAULT_INFERENCE_ENV_VARS: dict[str, str] = {
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
    "VLLM_ENGINE_READY_TIMEOUT_S": "4200",
    "UCX_TLS": "all",
    # Grace before vLLM SIGKILLs engine/worker procs at shutdown (read by
    # both the MPClient engine manager and the multiproc executor). The 5s
    # default truncates HiSparse decode ranks mid host-pool unpin, pushing
    # the remainder into uninterruptible kernel exit (node drain).
    "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS": "600",
}


def set_proc_title(name: str) -> None:
    """Set the process title for visibility in tools like ``ps`` and ``htop``.

    Args:
        name: A short, descriptive label (e.g. ``Trainer``, ``Orchestrator``).
              The process title is set to ``{PRIME_RL_PROC_PREFIX}::{name}``.
    """
    title = f"{PRIME_RL_PROC_PREFIX}::{name}"
    setproctitle.setproctitle(title)


def cleanup_threads(threads: list[Thread]):
    """Cleanup a list of threads"""
    for thread in threads:
        thread.join(timeout=5)


def cleanup_process(pid: int, sig: int = signal.SIGTERM):
    """Kill a process and all its descendants.

    Walks the process tree via ``psutil`` so that grandchildren spawned by
    intermediate wrappers (e.g. ``uv``, ``torchrun``) are reliably reached
    regardless of process-group boundaries.
    """
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    for child in children:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child.pid, sig)
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, sig)


def cleanup_processes(processes: list[Popen]):
    """Cleanup a list of subprocesses by killing their entire process trees."""
    for process in processes:
        if process.poll() is not None:
            continue
        cleanup_process(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            cleanup_process(process.pid, signal.SIGKILL)
        get_logger().debug(f"Cleaned up process {process.pid}")


def monitor_process(process: Popen, stop_event: Event, error_queue: list, process_name: str):
    """Monitor a subprocess and signal errors via shared queue."""
    process.wait()

    if process.returncode != 0:
        err_msg = f"{process_name.capitalize()} failed with exit code {process.returncode}"
        if process.stderr:
            err_msg += f"\n{process.stderr.read().decode('utf-8')}"
        error_queue.append(RuntimeError(err_msg))
    stop_event.set()
