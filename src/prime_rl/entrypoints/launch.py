"""Shared subprocess-launch primitives for RL entrypoints.

``rl_local`` and the multi-run shared-trainer entrypoint both spin up an
inference server, optional teacher inference server, a trainer torchrun
process, and one or more orchestrators. The startup, monitor-thread, and
supervision loop are identical across them; this module factors that
boilerplate out so the two entrypoints differ only in the orchestrator
plumbing.

All helpers operate on a ``LaunchSupervisor`` that bundles the shared
process / monitor / stop-event lists. Callers create one supervisor and
pass it to each ``start_*`` helper, then call ``wait_for_completion`` with
the labels whose termination signals end-of-run.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from subprocess import PIPE, Popen
from threading import Event, Thread
from typing import TYPE_CHECKING, Any

from prime_rl.utils.process import cleanup_processes, cleanup_threads, monitor_process
from prime_rl.utils.utils import get_free_port

if TYPE_CHECKING:  # pragma: no cover
    from prime_rl.configs.rl import RLConfig


@dataclass
class GpuMapping:
    """Resolved local→physical GPU assignments for an RL deployment."""

    infer: list[int]
    trainer: list[int]
    teacher: list[int]
    physical: dict[int, int]


@dataclass
class LaunchSupervisor:
    """Shared bookkeeping for spawned subprocesses + their monitor threads."""

    logger: Any
    log_dir: Path
    processes: list[Popen] = field(default_factory=list)
    monitor_threads: list[Thread] = field(default_factory=list)
    stop_events: dict[str, Event] = field(default_factory=dict)
    error_queue: list[Exception] = field(default_factory=list)


def compute_gpu_mapping(config: RLConfig, get_physical_gpu_ids: Any) -> GpuMapping:
    """Resolve the launcher-local GPU layout into physical GPU IDs.

    ``get_physical_gpu_ids`` is injected so this helper does not depend on
    pynvml directly and stays trivially testable.
    """
    gpu_offset = 0
    num_infer_gpus = config.deployment.num_infer_gpus if config.inference is not None else 0
    infer_local_gpu_ids = list(range(gpu_offset, gpu_offset + num_infer_gpus))
    gpu_offset += num_infer_gpus
    trainer_local_gpu_ids = list(range(gpu_offset, gpu_offset + config.deployment.num_train_gpus))
    gpu_offset += config.deployment.num_train_gpus
    num_teacher_gpus = config.deployment.num_teacher_gpus or 0
    teacher_local_gpu_ids = (
        list(range(gpu_offset, gpu_offset + num_teacher_gpus)) if num_teacher_gpus > 0 else []
    )

    total_requested_gpus = num_infer_gpus + config.deployment.num_train_gpus + num_teacher_gpus
    physical_gpu_ids = get_physical_gpu_ids()
    if total_requested_gpus > len(physical_gpu_ids):
        raise ValueError(
            f"Requested {total_requested_gpus} GPUs via deployment settings, but only "
            f"{len(physical_gpu_ids)} physical GPU(s) are available: {physical_gpu_ids}"
        )
    physical = {local_id: physical_gpu_ids[local_id] for local_id in range(total_requested_gpus)}
    return GpuMapping(
        infer=[physical[i] for i in infer_local_gpu_ids],
        trainer=[physical[i] for i in trainer_local_gpu_ids],
        teacher=[physical[i] for i in teacher_local_gpu_ids],
        physical=physical,
    )


def build_wandb_shared_env(config: RLConfig) -> dict[str, str]:
    """Compose the WANDB_SHARED_* env that subprocesses inherit when shared mode is on."""
    env: dict[str, str] = {}
    if config.wandb and config.wandb.shared:
        env["WANDB_SHARED_MODE"] = "1"
        env["WANDB_SHARED_RUN_ID"] = os.environ.get("WANDB_SHARED_RUN_ID", uuid.uuid4().hex)
    return env


def init_wandb_shared_primary(
    config: RLConfig,
    wandb_shared_env: dict[str, str],
    logger: Any | None = None,
) -> Any | None:
    """Open the shared W&B run from the launcher process as primary.

    The launcher outlives every subprocess (trainer + orchestrators), so
    binding ``x_update_finish_state`` to its lifetime is the only way to
    guarantee the shared run finishes after every late metric has flushed.
    A primary trainer would mark the run finished at ``max_steps`` while
    orchestrators are still emitting final-eval logs; a primary orchestrator
    would do the same if pruned. Returns the wandb ``Run`` so the caller
    can ``finish()`` it; returns ``None`` when shared mode is off.
    """
    if wandb_shared_env.get("WANDB_SHARED_MODE") != "1":
        return None
    if config.wandb is None:
        return None

    import wandb
    from wandb.errors import CommError

    run_id = wandb_shared_env["WANDB_SHARED_RUN_ID"]
    settings = wandb.Settings(
        mode="shared",
        x_label="launcher",
        x_primary=True,
        x_update_finish_state=True,
    )
    for attempt in range(5):
        try:
            return wandb.init(
                id=run_id,
                project=config.wandb.project,
                entity=config.wandb.entity,
                name=config.wandb.name,
                group=config.wandb.group,
                tags=config.wandb.tags,
                settings=settings,
            )
        except CommError as e:
            if attempt == 4:
                raise
            if logger is not None:
                logger.info(f"Transient W&B init error ({e}) - retrying in 10s ({attempt + 1}/5)")
            time.sleep(10)

    raise RuntimeError("unreachable")


def _start_supervised(
    label: str,
    cmd: list[str],
    env: dict[str, str],
    log_path: Path,
    supervisor: LaunchSupervisor,
) -> Popen:
    """Start ``cmd`` with stdout/stderr to ``log_path`` and a monitor thread.

    Records the process, monitor thread, and stop event under ``label`` on
    the supervisor so ``wait_for_completion`` can decide when to return.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w")
    try:
        process = Popen(cmd, env=env, stdout=log_file, stderr=log_file)
    finally:
        log_file.close()
    supervisor.processes.append(process)
    stop_event = Event()
    supervisor.stop_events[label] = stop_event
    monitor_thread = Thread(
        target=monitor_process,
        args=(process, stop_event, supervisor.error_queue, label),
        daemon=True,
    )
    monitor_thread.start()
    supervisor.monitor_threads.append(monitor_thread)
    return process


def start_inference(
    *,
    cmd: list[str],
    gpu_ids: list[int],
    label: str,
    log_path: Path,
    supervisor: LaunchSupervisor,
) -> Popen:
    """Spawn an inference (or teacher inference) server pinned to ``gpu_ids``."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpu_ids))}
    supervisor.logger.info(f"Starting {label} on GPU(s) {' '.join(map(str, gpu_ids))}")
    supervisor.logger.debug(f"{label} start command: {' '.join(cmd)}")
    return _start_supervised(label, cmd, env, log_path, supervisor)


def start_orchestrator(
    *,
    config_path: Path,
    label: str,
    log_path: Path,
    start_command: list[str],
    wandb_shared_env: dict[str, str],
    wandb_program: str,
    supervisor: LaunchSupervisor,
    extra_env: dict[str, str] | None = None,
) -> Popen:
    """Spawn an orchestrator subprocess pointed at ``config_path``.

    Single-run mode passes ``label="orchestrator"``. Multi-run mode passes a
    per-run label like ``"orchestrator-0000-abc"`` so the supervisor's
    ``stop_events`` and the W&B label can be told apart. ``extra_env`` is
    layered on top of the default env (after WANDB_*) so callers can scope
    per-orchestrator env vars like ``PRIME_RL_SWEEP_METRICS_JSONL`` without
    leaking them into sibling orchestrators.
    """
    cmd = ["orchestrator", "@", config_path.as_posix()]
    env = {
        **os.environ,
        **wandb_shared_env,
        "WANDB_SHARED_LABEL": label,
        "WANDB_SHARED_PRIMARY": "0",
        "LOGURU_FORCE_COLORS": "1",
        "WANDB_PROGRAM": wandb_program,
        "WANDB_ARGS": json.dumps(start_command),
    }
    if extra_env:
        env.update(extra_env)
    supervisor.logger.info(f"Starting {label} process")
    supervisor.logger.debug(f"{label} start command: {' '.join(cmd)}")
    return _start_supervised(label, cmd, env, log_path, supervisor)


def start_trainer(
    *,
    config_path: Path,
    gpu_ids: list[int],
    ranks_filter: list[int],
    log_path: Path,
    torchrun_log_dir: Path,
    start_command: list[str],
    wandb_shared_env: dict[str, str],
    wandb_program: str,
    supervisor: LaunchSupervisor,
) -> Popen:
    """Spawn the torchrun-driven trainer subprocess."""
    cmd = [
        "torchrun",
        "--role=trainer",
        f"--rdzv-endpoint=localhost:{get_free_port()}",
        f"--rdzv-id={uuid.uuid4().hex}",
        f"--log-dir={torchrun_log_dir}",
        f"--local-ranks-filter={','.join(map(str, ranks_filter))}",
        "--redirect=3",
        "--tee=3",
        f"--nproc-per-node={len(gpu_ids)}",
        "-m",
        "prime_rl.trainer.rl.train",
        "@",
        config_path.as_posix(),
    ]
    env = {
        **os.environ,
        **wandb_shared_env,
        "WANDB_SHARED_LABEL": "trainer",
        "WANDB_SHARED_PRIMARY": "0",
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpu_ids)),
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "LOGURU_FORCE_COLORS": "1",
        "WANDB_PROGRAM": wandb_program,
        "WANDB_ARGS": json.dumps(start_command),
    }
    supervisor.logger.info(f"Starting trainer on GPU(s) {' '.join(map(str, gpu_ids))}")
    supervisor.logger.debug(f"Training start command: {' '.join(cmd)}")
    return _start_supervised("trainer", cmd, env, log_path, supervisor)


def tail_trainer_log(supervisor: LaunchSupervisor, trainer_log: Path) -> Popen:
    """Mirror the trainer log to stdout so the user sees live training output."""
    tail = Popen(
        ["tail", "-F", trainer_log.as_posix()],
        stdout=PIPE,
        stderr=sys.stderr,
        text=True,
        bufsize=1,
    )

    def print_trainer_lines() -> None:
        assert tail.stdout is not None
        rank_prefix = re.compile(r"^\[[a-zA-Z]*[0-9]*\]:")
        for line in tail.stdout:
            print(rank_prefix.sub("", line), end="", flush=True)

    Thread(target=print_trainer_lines, daemon=True).start()
    supervisor.processes.append(tail)
    return tail


def wait_for_completion(
    primary_labels: list[str],
    supervisor: LaunchSupervisor,
) -> None:
    """Block until every primary label's stop event fires.

    A failure surfaced through ``supervisor.error_queue`` (typically a
    crashed monitor thread) tears down all processes and exits 1. Successful
    completion returns; the caller is responsible for inspecting individual
    return codes and final cleanup.
    """
    while not all(supervisor.stop_events[label].is_set() for label in primary_labels):
        if supervisor.error_queue:
            error = supervisor.error_queue[0]
            supervisor.logger.error(f"Error: {error}")
            supervisor.logger.error("Terminating all processes...")
            cleanup_threads(supervisor.monitor_threads)
            cleanup_processes(supervisor.processes)
            sys.exit(1)
        time.sleep(1)
