"""Shared-trainer LoRA launcher.

Boots one trainer + one inference server + N orchestrators against a shared
output directory, one orchestrator per pre-materialized ``run_*`` directory.
The trainer's ``MultiRunManager`` discovers the run dirs, allocates per-run
LoRA adapter slots, and routes each orchestrator's training samples to the
right adapter.

The sweep controller is the expected caller: it materializes
``<output_dir>/run_<trial_id>/control/orch.toml`` for every trial up front
and then invokes::

    rl-multi-run @ shared.toml --runs-dir run_a:run_b:run_c

``shared.toml`` is an ordinary ``RLConfig`` whose trainer block has
``max_concurrent_runs = N`` and whose orchestrator block carries the shared
defaults trial overrides inherit from. The orchestrator block in the shared
file is *not* used to launch a process here; only the per-run ``orch.toml``
files referenced via ``--runs-dir`` produce orchestrators.
"""

import os
import signal
import sys
import time
from pathlib import Path

from prime_rl.configs.rl import RLConfig
from prime_rl.entrypoints.launch import (
    LaunchSupervisor,
    build_wandb_shared_env,
    compute_gpu_mapping,
    init_wandb_shared_primary,
    start_inference,
    start_orchestrator,
    start_trainer,
    tail_trainer_log,
)
from prime_rl.entrypoints.rl import (
    INFERENCE_TOML,
    TEACHER_INFERENCE_TOML,
    TRAINER_TOML,
    check_gpus_available,
    get_physical_gpu_ids,
    write_subconfigs,
)
from prime_rl.entrypoints.rl_multi_run_args import RUNS_DIR_FLAG, parse_runs_dirs
from prime_rl.sweep.run_control import (
    finished_orchestrator_failures,
    record_finished_orchestrator_exit_codes,
    record_orchestrator_exit_codes,
)
from prime_rl.utils.config import cli
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.monitor import SWEEP_METRICS_JSONL_ENV
from prime_rl.utils.process import cleanup_processes, cleanup_threads, set_proc_title
from prime_rl.utils.utils import get_log_dir


def _validate_run_layout(run_dirs: list[Path]) -> None:
    """Each run dir must already contain ``control/orch.toml``.

    The trainer's ``MultiRunManager`` will reject a run whose config file is
    missing; failing loudly here gives a much better error than waiting for
    the trainer to silently skip the run.
    """
    missing = [d for d in run_dirs if not (d / "control" / "orch.toml").exists()]
    if missing:
        raise SystemExit(
            f"{RUNS_DIR_FLAG} entries missing control/orch.toml: {[d.as_posix() for d in missing]}. "
            "The sweep launcher must pre-materialize each run before invoking rl-multi-run."
        )


def _validate_concurrency(config: RLConfig, run_dirs: list[Path]) -> None:
    """Trainer must be sized for at least len(run_dirs) concurrent runs."""
    max_runs = getattr(config.trainer, "max_concurrent_runs", None)
    if max_runs is None or max_runs < 1:
        raise SystemExit(
            "rl-multi-run requires trainer.max_concurrent_runs >= 1 in the shared config "
            "(use multi-run-LoRA training)."
        )
    if max_runs < len(run_dirs):
        raise SystemExit(
            f"trainer.max_concurrent_runs={max_runs} but {len(run_dirs)} run dirs were passed; "
            "set max_concurrent_runs to at least the number of concurrent trials."
        )


def rl_multi_run(config: RLConfig, run_dirs: list[Path]) -> None:
    assert config.deployment.type == "single_node", "rl-multi-run is single-node only"
    _validate_concurrency(config, run_dirs)
    _validate_run_layout(run_dirs)

    logger = setup_logger(
        config.log.level or os.environ.get("PRIME_LOG_LEVEL", "info"),
        json_logging=config.log.json_logging,
    )

    config_dir = config.output_dir / "configs"
    write_subconfigs(config, config_dir)
    logger.info(f"Wrote subconfigs to {config_dir}")

    if config.dry_run:
        logger.success(
            "Dry run complete. To start a multi-run RL launch, remove --dry-run from your command."
        )
        return

    mapping = compute_gpu_mapping(config, get_physical_gpu_ids)
    logger.info(f"Using local->physical GPU mapping: {mapping.physical}")

    start_command = sys.argv
    logger.info(f"Starting multi-run RL launch with {len(run_dirs)} orchestrator(s)")
    logger.debug(f"Multi-run RL start command: {' '.join(start_command)}")

    wandb_shared_env = build_wandb_shared_env(config)

    all_gpu_ids = list(set(mapping.infer + mapping.trainer + mapping.teacher))
    check_gpus_available(all_gpu_ids)

    log_dir = get_log_dir(config.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    supervisor = LaunchSupervisor(logger=logger, log_dir=log_dir)

    orchestrator_labels: list[str] = []
    orchestrator_processes = []
    recorded_exit_code_dirs: set[Path] = set()

    def sigterm_handler(signum, frame):
        logger.warning("Received SIGTERM, terminating all processes...")
        cleanup_threads(supervisor.monitor_threads)
        cleanup_processes(supervisor.processes)
        record_orchestrator_exit_codes(orchestrator_processes, run_dirs)
        sys.exit(1)

    signal.signal(signal.SIGTERM, sigterm_handler)

    launcher_wandb_run = None
    try:
        # The launcher process owns the shared W&B run so its lifetime matches
        # the supervisor that waits on every subprocess. Trainer and
        # orchestrators stay non-primary so neither a trainer that exits at
        # max_steps nor a pruned orchestrator can mark the run finished while
        # siblings still have unflushed metrics (e.g. final-eval logs).
        launcher_wandb_run = init_wandb_shared_primary(config, wandb_shared_env, logger)

        if config.inference:
            start_inference(
                cmd=["inference", "@", (config_dir / INFERENCE_TOML).as_posix()],
                gpu_ids=mapping.infer,
                label="inference",
                log_path=log_dir / "inference.log",
                supervisor=supervisor,
            )
        else:
            logger.warning(
                "No inference config specified, skipping starting inference server. "
                "Make sure your inference server is running."
            )

        if config.teacher_inference:
            if not mapping.teacher:
                raise ValueError(
                    "teacher_inference is configured but deployment.num_teacher_gpus is not set."
                )
            start_inference(
                cmd=["inference", "@", (config_dir / TEACHER_INFERENCE_TOML).as_posix()],
                gpu_ids=mapping.teacher,
                label="teacher_inference",
                log_path=log_dir / "teacher_inference.log",
                supervisor=supervisor,
            )

        for run_dir in run_dirs:
            run_id = run_dir.name
            label = f"orchestrator-{run_id}"
            orchestrator_labels.append(label)
            orchestrator_processes.append(
                start_orchestrator(
                    config_path=run_dir / "control" / "orch.toml",
                    label=label,
                    log_path=log_dir / f"orchestrator-{run_id}.log",
                    start_command=start_command,
                    wandb_shared_env=wandb_shared_env,
                    wandb_program="uv run rl-multi-run",
                    supervisor=supervisor,
                    # Per-run sweep sidecar metrics, so the controller can
                    # read this trial's objective from <run_dir>/metrics.jsonl
                    # without colliding with sibling orchestrators.
                    extra_env={SWEEP_METRICS_JSONL_ENV: (run_dir / "metrics.jsonl").as_posix()},
                )
            )

        trainer_process = start_trainer(
            config_path=config_dir / TRAINER_TOML,
            gpu_ids=mapping.trainer,
            ranks_filter=config.trainer.log.ranks_filter,
            log_path=log_dir / "trainer.log",
            torchrun_log_dir=log_dir / "trainer" / "torchrun",
            start_command=start_command,
            wandb_shared_env=wandb_shared_env,
            wandb_program="uv run rl-multi-run",
            supervisor=supervisor,
        )

        logger.success("Startup complete. Showing trainer logs...")
        tail_trainer_log(supervisor, log_dir / "trainer.log")

        # Wait for every orchestrator + the trainer. Unlike single-run, we
        # cannot fail-fast on the first non-zero exit: a pruned or failed
        # orchestrator (Optuna writes evicted.txt, the orchestrator raises)
        # is expected, and the trainer's MultiRunManager keeps driving
        # surviving runs. The sweep controller attributes failure per-run
        # via each control/exit_code file, so we just wait for everything
        # to finish and let the post-wait code report status.
        #
        # Trainer / inference / teacher_inference crashes are still
        # terminal — survivors would block forever (trainer waiting for
        # batches, orchestrators waiting for weights), so we tear the wave
        # down and exit non-zero, but only after recording per-run exit
        # codes so the controller can attribute failures correctly.
        primary_label_set = {*orchestrator_labels, "trainer"}
        infra_labels = [label for label in supervisor.stop_events if label not in primary_label_set]
        all_labels = orchestrator_labels + ["trainer"]

        def _terminate_with_exit_codes(reason: str) -> None:
            logger.error(f"{reason}; tearing down remaining processes")
            record_orchestrator_exit_codes(orchestrator_processes, run_dirs)
            cleanup_threads(supervisor.monitor_threads)
            cleanup_processes(supervisor.processes)
            sys.exit(1)

        while not all(supervisor.stop_events[label].is_set() for label in all_labels):
            record_finished_orchestrator_exit_codes(
                orchestrator_processes,
                run_dirs,
                recorded_exit_code_dirs,
            )
            failed_finished_orchestrators = finished_orchestrator_failures(
                orchestrator_labels,
                orchestrator_processes,
                supervisor.stop_events,
            )
            if failed_finished_orchestrators:
                _terminate_with_exit_codes(
                    "All orchestrators have exited and at least one failed or was pruned: "
                    f"{failed_finished_orchestrators}"
                )
            # An infra subprocess (inference / teacher_inference) is meant
            # to run for the whole wave; if its stop_event fires here, it
            # exited unexpectedly and the trainer/orchestrators will hang.
            failed_infra = [label for label in infra_labels if supervisor.stop_events[label].is_set()]
            if failed_infra:
                _terminate_with_exit_codes(
                    f"Infrastructure process(es) exited unexpectedly: {failed_infra}"
                )
            if supervisor.stop_events["trainer"].is_set() and trainer_process.returncode != 0:
                _terminate_with_exit_codes(
                    f"Trainer failed with exit code {trainer_process.returncode}"
                )
            time.sleep(1)

        # Per-orchestrator exit_code is the sweep controller's source of
        # truth for failure attribution; write it as soon as we've waited
        # for every orchestrator, before any cleanup that might mask codes.
        record_orchestrator_exit_codes(orchestrator_processes, run_dirs)

        failed_orchestrators = [
            (label, proc.returncode)
            for label, proc in zip(orchestrator_labels, orchestrator_processes)
            if proc.returncode != 0
        ]
        if failed_orchestrators:
            for label, code in failed_orchestrators:
                logger.error(f"{label} failed with exit code {code}")
            cleanup_threads(supervisor.monitor_threads)
            cleanup_processes(supervisor.processes)
            sys.exit(1)

        if trainer_process.returncode != 0:
            logger.error(f"Trainer failed with exit code {trainer_process.returncode}")
            cleanup_threads(supervisor.monitor_threads)
            cleanup_processes(supervisor.processes)
            sys.exit(1)

        logger.success("Multi-run RL training finished!")
        cleanup_threads(supervisor.monitor_threads)
        cleanup_processes(supervisor.processes)

    except KeyboardInterrupt:
        logger.warning("Received interrupt signal, terminating all processes...")
        cleanup_threads(supervisor.monitor_threads)
        cleanup_processes(supervisor.processes)
        record_orchestrator_exit_codes(orchestrator_processes, run_dirs)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        cleanup_threads(supervisor.monitor_threads)
        cleanup_processes(supervisor.processes)
        record_orchestrator_exit_codes(orchestrator_processes, run_dirs)
        raise
    finally:
        # Run finish on every exit path (success, sys.exit, raise, SIGTERM)
        # so x_update_finish_state fires only after the supervisor has
        # waited on every subprocess.
        if launcher_wandb_run is not None:
            launcher_wandb_run.finish()


def main():
    set_proc_title("MultiRunLauncher")
    run_dirs, remaining = parse_runs_dirs(sys.argv[1:])
    config = cli(RLConfig, args=remaining)
    rl_multi_run(config, run_dirs)


if __name__ == "__main__":
    main()
