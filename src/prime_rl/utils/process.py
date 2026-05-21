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
TORCHRUN_LOG_PREFIX_PATTERN = r"s/^\[[a-zA-Z]*[0-9]*\]://"


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


def start_tail_processes(
    log_path: str | os.PathLike[str],
    *,
    strip_torchrun_prefix: bool = False,
) -> list[Popen]:
    """Start subprocesses that stream a log file without invoking a shell."""
    tail_process = Popen(
        ["tail", "-F", os.fspath(log_path)],
        stdout=subprocess.PIPE if strip_torchrun_prefix else None,
    )
    if not strip_torchrun_prefix:
        return [tail_process]

    if tail_process.stdout is None:
        raise RuntimeError("tail process stdout was not captured")

    sed_process = Popen(
        ["sed", "-u", TORCHRUN_LOG_PREFIX_PATTERN],
        stdin=tail_process.stdout,
    )
    tail_process.stdout.close()
    return [tail_process, sed_process]


def monitor_process(process: Popen, stop_event: Event, error_queue: list, process_name: str):
    """Monitor a subprocess and signal errors via shared queue."""
    process.wait()

    if process.returncode != 0:
        err_msg = f"{process_name.capitalize()} failed with exit code {process.returncode}"
        if process.stderr:
            err_msg += f"\n{process.stderr.read().decode('utf-8')}"
        error_queue.append(RuntimeError(err_msg))
    stop_event.set()
