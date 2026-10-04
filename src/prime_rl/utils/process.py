import contextlib
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path
from subprocess import Popen
from threading import Event, Thread

import psutil
import setproctitle

from prime_rl.utils.logger import get_logger

PRIME_RL_PROC_PREFIX = "PRL"


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
}


def get_physical_gpu_ids() -> list[int]:
    """Return physical GPU IDs visible to the launcher."""
    raw_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw_visible is None:
        import pynvml

        pynvml.nvmlInit()
        return list(range(pynvml.nvmlDeviceGetCount()))
    return [int(token.strip()) for token in raw_visible.split(",") if token.strip()]


def partition_gpus(num_infer: int, num_train: int) -> tuple[list[int], list[int]]:
    """Split the launcher's physical GPUs into inference GPUs followed by trainer GPUs."""
    physical_gpu_ids = get_physical_gpu_ids()
    if num_infer + num_train > len(physical_gpu_ids):
        raise ValueError(
            f"Requested {num_infer + num_train} GPUs via deployment settings, but only "
            f"{len(physical_gpu_ids)} physical GPU(s) are available: {physical_gpu_ids}"
        )
    return physical_gpu_ids[:num_infer], physical_gpu_ids[num_infer : num_infer + num_train]


def torchrun_cmd(
    module: str, config_path: Path, nproc_per_node: int, log_dir: Path, ranks_filter: list[int]
) -> list[str]:
    """Single-node torchrun command for a trainer module."""
    from prime_rl.utils.utils import get_free_port

    return [
        "torchrun",
        "--role=trainer",
        f"--rdzv-endpoint=localhost:{get_free_port()}",
        f"--rdzv-id={uuid.uuid4().hex}",
        # Pipe all logs to file, and only master rank logs to stdout
        f"--log-dir={log_dir / 'trainer' / 'torchrun'}",
        f"--local-ranks-filter={','.join(map(str, ranks_filter))}",
        "--redirect=3",
        "--tee=3",
        f"--nproc-per-node={nproc_per_node}",
        "-m",
        module,
        "@",
        config_path.as_posix(),
    ]


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


class ProcessGroup:
    """Launcher subprocesses, each logging to its own file and watched by a monitor thread.

    As a context manager, SIGTERM, Ctrl-C, errors and normal exit all terminate every
    started process."""

    def __init__(self):
        self.processes: list[Popen] = []
        self.monitor_threads: list[Thread] = []
        self.error_queue: list[Exception] = []
        self.stop_events: dict[str, Event] = {}

    def __enter__(self) -> "ProcessGroup":
        signal.signal(signal.SIGTERM, self._sigterm_handler)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is KeyboardInterrupt:
            get_logger().warning("Received interrupt signal, terminating all processes...")
        elif isinstance(exc, Exception):
            get_logger().error(f"Error occurred: {exc}")
        self.cleanup()
        if exc_type is KeyboardInterrupt:
            sys.exit(1)

    def _sigterm_handler(self, signum, frame):
        get_logger().warning("Received SIGTERM, terminating all processes...")
        # Clean up here too: a SIGTERM during __exit__'s cleanup would otherwise leave orphans.
        self.cleanup()
        sys.exit(1)

    def cleanup(self) -> None:
        cleanup_processes(self.processes)
        cleanup_threads(self.monitor_threads)

    def start(self, name: str, cmd: list[str], env: dict[str, str], log_path: Path) -> None:
        get_logger().debug(f"{name[:1].upper() + name[1:]} command: {' '.join(cmd)}")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # If we don't log stdout, the inference server hangs
        with open(log_path, "w") as log_file:
            process = Popen(cmd, env=env, stdout=log_file, stderr=log_file)
        self.processes.append(process)
        stop_event = Event()
        self.stop_events[name] = stop_event
        monitor_thread = Thread(target=monitor_process, args=(process, stop_event, self.error_queue, name), daemon=True)
        monitor_thread.start()
        self.monitor_threads.append(monitor_thread)

    def wait(self, *names: str) -> None:
        """Block until the named processes exit; exit with code 1 as soon as any started process fails."""
        events = [self.stop_events[name] for name in names]
        while True:
            pending = [event for event in events if not event.is_set()]
            if self.error_queue:
                get_logger().error(f"Error: {self.error_queue[0]}")
                get_logger().error("Terminating all processes...")
                sys.exit(1)
            if not pending:
                return
            pending[0].wait(timeout=1)
