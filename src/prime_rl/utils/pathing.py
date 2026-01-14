import asyncio
import time
from pathlib import Path

from prime_rl.utils.logger import get_logger


def get_log_dir(output_dir: Path) -> Path:
    return output_dir / "logs"


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


def get_env_worker_log_dir(output_dir: Path, env_name: str) -> Path:
    return output_dir / "logs" / "env_workers" / env_name


def get_step_path(path: Path, step: int) -> Path:
    return path / f"step_{step}"


def get_all_ckpt_steps(ckpt_dir: Path) -> list[int]:
    """Gets all checkpoint steps from the checkpoint directory, sorted in ascending order."""
    step_dirs = list(ckpt_dir.glob("step_*"))
    return sorted([int(step_dir.name.split("_")[-1]) for step_dir in step_dirs])


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


def get_common_ckpt_steps(dirs: list[Path]) -> list[int]:
    """Returns sorted intersection of checkpoint steps across directories."""
    sets = [set(get_all_ckpt_steps(d)) for d in dirs if d.exists()]
    if not sets:
        return []
    return sorted(set.intersection(*sets))


def warn_if_ckpts_inconsistent(output_dir: Path, resume_step: int) -> None:
    """Warns if resume_step is not safe given checkpoint state across directories."""
    logger = get_logger()
    orch_dirs = list(output_dir.glob("run_*"))
    if len(orch_dirs) > 1:
        return  # Multi-tenant: orchestrators may legitimately differ

    all_dirs_and_steps = {
        get_ckpt_dir(output_dir): get_all_ckpt_steps(get_ckpt_dir(output_dir)),
        get_weights_dir(output_dir): get_all_ckpt_steps(get_weights_dir(output_dir)),
    }
    if orch_dirs:
        all_dirs_and_steps[get_ckpt_dir(orch_dirs[0])] = get_all_ckpt_steps(get_ckpt_dir(orch_dirs[0]))

    if not all(all_dirs_and_steps.values()):  # no checkpoints found
        return

    common_steps = get_common_ckpt_steps(all_dirs_and_steps.keys())
    if not common_steps:
        logger.error(f"No common checkpoint steps across dirs: {all_dirs_and_steps}. Cannot safely resume.")
        return
    latest_common_step = max(common_steps)
    latest_steps_all_equal = all(
        all_dirs_and_steps[_dir][-1] == latest_common_step for _dir in all_dirs_and_steps.keys()
    )

    if resume_step == -1 and not latest_steps_all_equal:
        logger.warning(
            f"Checkpoint mismatch detected with resume_step=-1. Check: {all_dirs_and_steps.keys()}. "
            f"Consider setting resume_step={latest_common_step} explicitly."
        )
    elif resume_step >= 0 and resume_step not in common_steps:
        logger.warning(
            f"resume_step={resume_step} not found in all checkpoint dirs: {all_dirs_and_steps.keys()}. "
            f"Latest common step: {latest_common_step}."
        )


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
