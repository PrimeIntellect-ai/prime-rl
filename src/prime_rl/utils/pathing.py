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


def get_env_worker_log_file(output_dir: Path, env_name: str) -> Path:
    return output_dir / "logs" / "env_workers" / f"{env_name}.log"


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


def wait_for_stable_checkpoint(ckpt_dir: Path, step: int, timeout: int) -> bool:
    """Wait up to timeout seconds for a checkpoint to become stable. Returns True if stable."""
    logger = get_logger()
    stable_file = ckpt_dir / f"step_{step}" / "STABLE"

    if stable_file.exists():
        return True

    logger.info(f"Checkpoint step_{step} not yet stable. Waiting up to {timeout}s...")
    wait_time = 0
    while wait_time < timeout:
        time.sleep(1)
        wait_time += 1
        if stable_file.exists():
            logger.info(f"Checkpoint step_{step} is now stable.")
            return True
        if wait_time % 60 == 0:
            logger.info(f"Still waiting for step_{step} to become stable ({wait_time}s elapsed)")

    logger.warning(f"Checkpoint step_{step} did not become stable within {timeout}s.")
    return False


def resolve_latest_ckpt_step(ckpt_dir: Path, wait_for_stable: int | None = None) -> int | None:
    """Gets the latest checkpoint step from the checkpoint directory.

    Args:
        ckpt_dir: Path to the checkpoint directory.
        wait_for_stable: If set, wait up to this many seconds for the checkpoint to become stable.

    Returns:
        The latest checkpoint step, or None if no checkpoints found (or stability timeout exceeded).
    """
    logger = get_logger()
    steps = get_all_ckpt_steps(ckpt_dir)
    if len(steps) == 0:
        logger.warning(f"No checkpoints found in {ckpt_dir}. Starting from scratch.")
        return None

    latest_step = steps[-1]

    if wait_for_stable:
        if not wait_for_stable_checkpoint(ckpt_dir, latest_step, wait_for_stable):
            return None

    logger.info(f"Found latest checkpoint in {ckpt_dir}: {latest_step}")
    return latest_step


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
