import json
from typing import Literal

import torch
from pydantic_config import BaseConfig
from vllm import RequestOutput, CompletionOutput
from concurrent.futures import ThreadPoolExecutor

from zeroband.inference.rewards.registry import REWARD_FUNCTIONS
from zeroband.utils.logger import get_logger

# Global logger
logger = get_logger("INFER")


class LenRewardConfig(BaseConfig):
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
    len_reward_config: LenRewardConfig,
    task_type: str,
) -> dict[str, float]:
    # Compute task reward
    reward_fn = REWARD_FUNCTIONS[task_type]
    task_reward = reward_fn(completion_output.text, verification_info)

    # Compute length penalty
    total_reward = task_reward
    length_penalty = 0
    if verification_info["target_length"] > 0:
        output_length = len(completion_output.token_ids)
        target_length = verification_info["target_length"]

        if len_reward_config.reward_type == "exact":
            length_penalty = abs(output_length - target_length)
            length_penalty = length_penalty * len_reward_config.reward_coef  # Scale factor to balance with math reward
            total_reward -= length_penalty

        elif len_reward_config.reward_type == "max":
            diff = target_length - output_length
            length_penalty = torch.clip(
                torch.tensor(len_reward_config.reward_coef * diff + len_reward_config.max_reward_delta), 0, 1
            ).item()
            total_reward *= length_penalty

        elif len_reward_config.reward_type == "clip":
            length_penalty = int(output_length > target_length)

            if length_penalty == 1:
                total_reward = 0

    return {"total_reward": total_reward, "task_reward": task_reward, "length_penalty": length_penalty}


def compute_rewards(
    generated_tokens: list[RequestOutput],
    verification_infos: list[str],
    target_lengths: list[int],
    task_types: list[str],
    len_reward_config: LenRewardConfig,
) -> tuple[dict[int, torch.FloatTensor], dict[int, torch.FloatTensor], dict[int, torch.FloatTensor]]:
    parsed_infos = [json.loads(ver) for ver in verification_infos]

    for info, target_len in zip(parsed_infos, target_lengths):
        info["target_length"] = target_len

    futures, mapping = [], []
    with ThreadPoolExecutor(max_workers=32) as executor:
        for request_id, (request, verification_info, task_type) in enumerate(zip(generated_tokens, parsed_infos, task_types)):
            for output in request.outputs:
                futures.append(executor.submit(compute_reward, output, verification_info, len_reward_config, task_type))
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
        grouped_total_rewards[request_id].append(result["total_reward"])
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
