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


def get_step_path(path: Path, step: int) -> Path:
    return path / f"step_{step}"


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
