import asyncio
import shutil
import time
from pathlib import Path

from prime_rl.utils.logger import get_logger


def get_log_dir(output_dir: Path) -> Path:
    return output_dir / "logs"


def format_log_message(
    log_dir: Path,
    trainer: bool = False,
    orchestrator: bool = False,
    inference: bool = False,
    job_log: bool = False,
    train_env_names: list[str] | None = None,
    eval_env_names: list[str] | None = None,
    num_train_nodes: int = 1,
    num_infer_nodes: int = 0,
) -> str:
    """Format a log message showing where to find all log files."""
    col = 18
    i1 = " " * 2
    i2 = " " * 3
    i3 = " " * 4
    max_name = col - 4

    log_lines: list[str] = []
    if job_log:
        log_lines.append(f"{i1}{'Job:':<{col}}tail -F {log_dir.parent}/job_*.log")
    if trainer:
        log_lines.append(f"{i1}{'Trainer:':<{col}}tail -F {log_dir}/trainer.log")
        if num_train_nodes > 1:
            log_lines.append(f"{i2}{'All nodes:':<{col - 1}}tail -F {log_dir}/trainer/node_*.log")
        log_lines.append(f"{i2}{'All ranks:':<{col - 1}}tail -F {log_dir}/trainer/torchrun/*/*/*/*.log")
    if orchestrator:
        log_lines.append(f"{i1}{'Orchestrator:':<{col}}tail -F {log_dir}/orchestrator.log")
    if inference:
        log_lines.append(f"{i1}{'Inference:':<{col}}tail -F {log_dir}/inference.log")
        if num_infer_nodes > 1:
            log_lines.append(f"{i2}{'All nodes:':<{col - 1}}tail -F {log_dir}/inference/node_*.log")
    if train_env_names:
        env_log_dir = log_dir / "envs"
        log_lines.append(f"{i1}{'Envs:':<{col}}tail -F {env_log_dir}/*/*/*.log")
        log_lines.append(f"{i2}{'Train:':<{col - 1}}tail -F {env_log_dir}/train/*/*.log")
        for name in train_env_names:
            short = name if len(name) <= max_name else name[: max_name - 3] + "..."
            log_lines.append(f"{i3}{f'{short}:':<{col - 2}}tail -F {env_log_dir}/train/{name}/*.log")
        if eval_env_names:
            log_lines.append(f"{i2}{'Eval:':<{col - 1}}tail -F {env_log_dir}/eval/*/*.log")
            for name in eval_env_names:
                short = name if len(name) <= max_name else name[: max_name - 3] + "..."
                log_lines.append(f"{i3}{f'{short}:':<{col - 2}}tail -F {env_log_dir}/eval/{name}/*.log")
    return "Logs:\n" + "\n".join(log_lines)


def get_config_dir(output_dir: Path) -> Path:
    return output_dir / "configs"


def get_ckpt_dir(output_dir: Path) -> Path:
    return output_dir / "checkpoints"


def get_weights_dir(output_dir: Path) -> Path:
    return output_dir / "weights"


def get_rollout_dir(output_dir: Path) -> Path:
    return output_dir / "rollouts"


def get_eval_dir(output_dir: Path) -> Path:
    return output_dir / "evals"


def get_broadcast_dir(output_dir: Path) -> Path:
    return output_dir / "broadcasts"


def get_step_path(path: Path, step: int) -> Path:
    return path / f"step_{step}"


def get_all_ckpt_steps(ckpt_dir: Path) -> list[int]:
    """Gets all checkpoint steps from the checkpoint directory, sorted in ascending order."""
    step_dirs = list(ckpt_dir.glob("step_*"))
    return sorted([int(step_dir.name.split("_")[-1]) for step_dir in step_dirs])


def get_stable_ckpt_steps(ckpt_dir: Path) -> list[int]:
    """Gets checkpoint steps that have STABLE file, sorted in ascending order."""
    steps = get_all_ckpt_steps(ckpt_dir)
    return [s for s in steps if (ckpt_dir / f"step_{s}" / "STABLE").exists()]


def resolve_latest_ckpt_step(ckpt_dir: Path) -> int | None:
    """Gets the latest checkpoint step from the checkpoint directory. Returns None if no checkpoints are found."""
    steps = get_all_ckpt_steps(ckpt_dir)
    if len(steps) == 0:
        logger = get_logger()
        logger.warning(f"No checkpoints found in {ckpt_dir}. Starting from scratch.")
        return None
    latest_step = steps[-1]
    logger = get_logger()
    logger.info(f"Found latest checkpoint in {ckpt_dir}: {latest_step}")
    return latest_step


def has_checkpoints(output_dir: Path) -> bool:
    """Check if the output directory contains any checkpoints."""
    ckpt_dir = get_ckpt_dir(output_dir)
    return ckpt_dir.exists() and any(ckpt_dir.iterdir())


def validate_output_dir(output_dir: Path, *, resuming: bool, clean: bool, ckpt_output_dir: Path | None = None) -> None:
    """Validate the output directory before training starts.

    Raises if the directory contains checkpoints from a previous run, unless
    explicitly resuming or opting into cleaning. Other artifacts (logs,
    rollouts, configs) are fine and don't trigger the error.

    When ckpt_output_dir is set, checkpoints live there instead of under
    output_dir, so the guard and clean logic check both locations.
    """
    dirs_to_check = [output_dir]
    if ckpt_output_dir is not None and ckpt_output_dir != output_dir:
        dirs_to_check.append(ckpt_output_dir)

    if resuming:
        return
    if clean:
        logger = get_logger()
        for d in dirs_to_check:
            if d.exists():
                logger.warning(f"Cleaning existing directory: {d}")
                shutil.rmtree(d)
        return
    for d in dirs_to_check:
        if has_checkpoints(d):
            raise FileExistsError(
                f"Directory '{d}' already contains checkpoints from a previous run. "
                f"To resume the latest step of the previous run, set ckpt.resume_step=-1 or --ckpt.resume-step -1 via CLI. "
                f"To delete the existing directory and start fresh, set clean_output_dir=true or --clean-output-dir via CLI. "
                f"Otherwise use a unique output_dir for this experiment."
            )


def clean_future_steps(output_dir: Path, resume_step: int) -> None:
    """Remove stale rollouts and broadcasts past ``resume_step``.

    Pass ``resume_step=-1`` to wipe every step directory (fresh runs).
    """
    run_default = output_dir / "run_default"
    dirs = [
        get_rollout_dir(output_dir),
        get_rollout_dir(run_default),
        get_broadcast_dir(run_default),
    ]

    for directory in dirs:
        steps_to_delete = [step for step in get_all_ckpt_steps(directory) if step > resume_step]
        if not steps_to_delete:
            continue
        get_logger().info(
            f"Deleting {len(steps_to_delete)} step directories in {directory} ({','.join(map(str, steps_to_delete))})"
        )
        for step in steps_to_delete:
            shutil.rmtree(get_step_path(directory, step))


def sync_wait_for_path(path: Path, interval: int = 1, log_interval: int = 10) -> None:
    logger = get_logger()
    wait_time = 0
    logger.debug(f"Waiting for path `{path}`")
    while True:
        if path.exists():
            logger.debug(f"Found path `{path}`")
            break
        if wait_time % log_interval == 0 and wait_time > 0:  # Every log_interval seconds
            logger.debug(f"Waiting for path `{path}` for {wait_time} seconds")
        time.sleep(interval)
        wait_time += interval


async def wait_for_path(path: Path, interval: int = 1, log_interval: int = 10) -> None:
    logger = get_logger()
    wait_time = 0
    logger.debug(f"Waiting for path `{path}`")
    while True:
        if path.exists():
            logger.debug(f"Found path `{path}`")
            break
        if wait_time % log_interval == 0 and wait_time > 0:  # Every log_interval seconds
            logger.debug(f"Waiting for path `{path}` for {wait_time} seconds")
        await asyncio.sleep(interval)
        wait_time += interval
