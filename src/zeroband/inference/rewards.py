from typing import Literal

import torch
from pydantic_config import BaseConfig
from vllm import RequestOutput, CompletionOutput
from concurrent.futures import ThreadPoolExecutor

from zeroband.inference.genesys import get_reward_function, TaskType
from zeroband.utils.logger import get_logger

# Global logger
logger = get_logger("INFER")


class RewardsConfig(BaseConfig):
    reward_type: Literal["exact", "max", "clip"] = "max"
    target_length_sampling: Literal["discrete", "range"] = "discrete"
    length_prompt_location: Literal["system_prompt", "instruction"] = "system_prompt"

    # applicable if target_length_sampling == "range"
    min_length: int = 1000
    max_length: int = 24000

    # applicable if target_length_sampling == "discrete"
    target_lengths: list[float] = [500, 1000, 2000, 3000]

    # applicable for reward_type max and exact
    reward_coef: float = 0.0003

    # only applicable for reward_type == "max"
    max_reward_delta: float = 0.5


def compute_reward(
    completion_output: CompletionOutput,
    verification_info: dict,
    task_type: TaskType,
    config: RewardsConfig | None,
) -> dict[str, float]:
    # Compute task reward
    compute_reward = get_reward_function(task_type)
    task_reward = compute_reward(completion_output.text, verification_info)

    # Compute length penalty
    reward = task_reward
    target_length = verification_info.get("target_length", None)
    length_penalty = 0
    if target_length is not None:
        output_length = len(completion_output.token_ids)
        # Penalizes absolute deviation from target length
        if config.reward_type == "exact":
            length_penalty = abs(target_length - output_length) * config.reward_coef
            reward -= length_penalty
        # Rewards for being close to target length with a maximum reward
        elif config.reward_type == "max":
            raw_value = config.reward_coef * (target_length - output_length) + config.max_reward_delta
            length_penalty = max(0, min(1, raw_value))
            reward *= length_penalty
        # Zero reward if output exceeds target length
        elif config.reward_type == "clip":
            length_penalty = int(output_length > target_length)

            if length_penalty == 1:
                reward = 0
        else:
            raise ValueError(f"Invalid reward type: {config.reward_type}")
    logger.debug(f"Computed reward: {reward} (task_reward: {task_reward}, length_penalty: {length_penalty})")

    return {"reward": reward, "task_reward": task_reward, "length_penalty": length_penalty}


def compute_rewards(
    request_outputs: list[RequestOutput],
    verification_infos: list[dict],
    task_types: list[str],
    config: RewardsConfig | None,
) -> tuple[dict[int, torch.FloatTensor], dict[int, torch.FloatTensor], dict[int, torch.FloatTensor]]:
    futures, mapping = [], []
    with ThreadPoolExecutor(max_workers=32) as executor:
        for request_id, (request, verification_info, task_type) in enumerate(zip(request_outputs, verification_infos, task_types)):
            for output in request.outputs:
                args = (output, verification_info, task_type, config)
                futures.append(executor.submit(compute_reward, *args))
                mapping.append(request_id)

        results = [future.result() for future in futures]

    # Group rewards by request index
    grouped_total_rewards = {}
    grouped_task_rewards = {}
    grouped_length_penalties = {}

    for request_id in set(mapping):
        grouped_total_rewards[request_id] = []
        grouped_task_rewards[request_id] = []
        grouped_length_penalties[request_id] = []

    for request_id, result in zip(mapping, results):
        grouped_total_rewards[request_id].append(result["reward"])
        grouped_task_rewards[request_id].append(result["task_reward"])
        grouped_length_penalties[request_id].append(result["length_penalty"])

    for request_id in grouped_total_rewards:
        grouped_total_rewards[request_id] = torch.FloatTensor(grouped_total_rewards[request_id])
        grouped_task_rewards[request_id] = torch.FloatTensor(grouped_task_rewards[request_id])
        grouped_length_penalties[request_id] = torch.FloatTensor(grouped_length_penalties[request_id])

    return grouped_total_rewards, grouped_task_rewards, grouped_length_penalties


def compute_advantages(grouped_rewards: dict[int, dict[str, torch.FloatTensor]], epsilon: float = 1e-6) -> dict[int, list[float]]:
    advantages = {}
    for req_idx, rewards_tensor in grouped_rewards.items():
        mean_reward = rewards_tensor.mean().item()
        sd_reward = rewards_tensor.std().item()
        normalized_reward = ((rewards_tensor - mean_reward) / (sd_reward + epsilon)).tolist()
        advantages[req_idx] = normalized_reward
    return advantages
