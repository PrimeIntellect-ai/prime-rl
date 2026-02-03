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


def resolve_latest_ckpt_step(ckpt_dir: Path, stability_timeout: int = 1000) -> int | None:
    """Gets the latest checkpoint step from the checkpoint directory.

    If the latest checkpoint is not yet stable (missing STABLE file), waits up to
    stability_timeout seconds for it to become stable. Returns None if no checkpoints
    are found or if the timeout is exceeded.

    Args:
        ckpt_dir: Path to the checkpoint directory.
        stability_timeout: Maximum seconds to wait for checkpoint to become stable.
    """
    logger = get_logger()
    steps = get_all_ckpt_steps(ckpt_dir)
    if len(steps) == 0:
        logger.warning(f"No checkpoints found in {ckpt_dir}. Starting from scratch.")
        return None

    latest_step = steps[-1]
    stable_file = ckpt_dir / f"step_{latest_step}" / "STABLE"

    if not stable_file.exists():
        logger.info(f"Checkpoint step_{latest_step} not yet stable. Waiting up to {stability_timeout}s...")
        wait_time = 0
        while wait_time < stability_timeout:
            time.sleep(1)
            wait_time += 1
            if stable_file.exists():
                break
            if wait_time % 30 == 0:
                logger.info(f"Still waiting for checkpoint step_{latest_step} to become stable ({wait_time}s elapsed)")

        if not stable_file.exists():
            logger.warning(
                f"Checkpoint step_{latest_step} did not become stable within {stability_timeout}s. Starting from scratch."
            )
            return None

    logger.info(f"Found latest stable checkpoint in {ckpt_dir}: {latest_step}")
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
