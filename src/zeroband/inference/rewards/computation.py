import asyncio
import torch
import vllm
import json
from typing import Literal

from .registry import REWARD_FUNCTIONS
from pydantic_config import BaseConfig

import concurrent.futures
from functools import lru_cache


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


@lru_cache(maxsize=1)
def get_process_executor():
    return concurrent.futures.ProcessPoolExecutor(max_workers=32)


async def compute_reward_for_output(output, verification_info, len_reward_config, task_type):
    loop = asyncio.get_running_loop()
    reward_fn = REWARD_FUNCTIONS[task_type]
    task_reward = await loop.run_in_executor(get_process_executor(), reward_fn, output.text, verification_info)

    total_reward = task_reward
    length_penalty = 0
    if verification_info["target_length"] > 0:
        output_length = len(output.token_ids)
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

    return dict(total_reward=total_reward, task_reward=task_reward, length_penalty=length_penalty)


async def compute_rewards_async(
    generated_tokens: list[vllm.RequestOutput],
    verification_infos: list[str],
    target_lengths: list[int],
    task_types: list[str],
    len_reward_config: LenRewardConfig,
) -> tuple[dict[int, torch.FloatTensor], dict[int, torch.FloatTensor], dict[int, torch.FloatTensor]]:
    parsed_infos = [json.loads(ver) for ver in verification_infos]

    for info, target_len in zip(parsed_infos, target_lengths):
        info["target_length"] = target_len

    tasks = []
    mapping = []

    for req_idx, (request, verification_info, task_type) in enumerate(zip(generated_tokens, parsed_infos, task_types)):
        for output in request.outputs:
            tasks.append(asyncio.create_task(compute_reward_for_output(output, verification_info, len_reward_config, task_type)))
            mapping.append(req_idx)

    all_results = await asyncio.gather(*tasks)

    grouped_total_rewards = {}
    grouped_task_rewards = {}
    grouped_length_penalties = {}

    for req_idx in set(mapping):
        grouped_total_rewards[req_idx] = []
        grouped_task_rewards[req_idx] = []
        grouped_length_penalties[req_idx] = []

    for req_idx, result in zip(mapping, all_results):
        grouped_total_rewards[req_idx].append(result["total_reward"])
        grouped_task_rewards[req_idx].append(result["task_reward"])
        grouped_length_penalties[req_idx].append(result["length_penalty"])

    for req_idx in grouped_total_rewards:
        grouped_total_rewards[req_idx] = torch.FloatTensor(grouped_total_rewards[req_idx])
        grouped_task_rewards[req_idx] = torch.FloatTensor(grouped_task_rewards[req_idx])
        grouped_length_penalties[req_idx] = torch.FloatTensor(grouped_length_penalties[req_idx])

    return grouped_total_rewards, grouped_task_rewards, grouped_length_penalties


def compute_advantages_grpo(grouped_rewards: dict[int, dict[str, torch.FloatTensor]], epsilon: float = 1e-6) -> dict[int, list[float]]:
    advantages = {}
    for req_idx, rewards_tensor in grouped_rewards.items():
        mean = torch.mean(rewards_tensor).item()
        std_dev = torch.std(rewards_tensor).item()
        normalized = ((rewards_tensor - mean) / (std_dev + epsilon)).tolist()
        advantages[req_idx] = normalized
    return advantages
