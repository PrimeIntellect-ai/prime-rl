import time
from pathlib import Path
from typing import Any

import numpy as np

from zeroband.training.orchestrator.genesys import get_reward_function
from zeroband.utils.logger import get_logger


def wait_for_weight_checkpoint(path: Path, step: int, interval: int = 1, log_interval: int = 10) -> None:
    logger = get_logger()
    wait_time = 0
    model_path = Path(path) / f"step_{step}" / "model.pt"
    logger.info(f"Waiting for checkpoint for step {step} at {model_path}")
    while True:
        if model_path.exists():
            logger.info(f"Found checkpoint for step {step} at {model_path}")
            break
        if wait_time % log_interval == 0 and wait_time > 0:  # Every log_interval seconds
            logger.info(f"Waiting for checkpoint for step {step} at {model_path} for {wait_time} seconds")
        time.sleep(interval)
        wait_time += interval


def compute_rewards(
    completions: list[str],
    task_types: list[str],
    verification_infos: list[dict[str, Any]],
) -> list[float]:
    rewards = []
    for completion, task_type, verification_info in zip(completions, task_types, verification_infos):
        compute_reward = get_reward_function(task_type)
        reward = compute_reward(completion, verification_info)
        rewards.append(reward)
    return rewards


def compute_advantages(rewards: list[float], samples_per_problem: int) -> list[float]:
    per_problem_rewards = [rewards[i : i + samples_per_problem] for i in range(0, len(rewards), samples_per_problem)]
    advantages = []
    for problem_rewards in per_problem_rewards:
        reward_array = np.array(problem_rewards)
        problem_advantages = reward_array - reward_array.mean()
        advantages.extend(problem_advantages.tolist())
    return advantages
