from typing import Literal
from dataclasses import dataclass

import numpy as np
from pydantic_config import BaseConfig
from vllm import RequestOutput, CompletionOutput
from concurrent.futures import ThreadPoolExecutor

from zeroband.inference.genesys import get_reward_function, TaskType
from zeroband.utils.logger import get_logger

# Global logger
logger = get_logger("INFER")


class LenRewardsConfig(BaseConfig):
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


@dataclass
class CompletionReward:
    completion_id: int  # type(CompletionOutput.index)
    reward: float
    task_reward: float
    length_penalty: float
    advantage: float | None = None


@dataclass
class RequestRewards:
    request_id: str  # type(RequestOutput.request_id)
    rewards: list[CompletionReward]


def _compute_completion_reward(
    completion_output: CompletionOutput,
    verification_info: dict,
    task_type: TaskType,
    config: LenRewardsConfig | None,
) -> dict[str, float]:
    """
    Computes the reward from a single vLLM completion output given the
    task type (e.g. math, code, etc.) and information on how to verify
    the output. Also supports an optional length penalty.

    Args:
        completion_output: The completion output to compute the reward for.
        verification_info: The verification info for the completion output.
        task_type: The task type for the completion output.
        config: The config for the rewards.

    Returns:
        A dictionary containing the reward, task reward, and length penalty.
    """
    # Compute task reward
    compute_reward = get_reward_function(task_type)
    task_reward = compute_reward(completion_output.text, verification_info)

    # Compute length penalty
    reward = task_reward
    length_penalty = 0
    target_length = verification_info["target_length"]
    if target_length > 0:
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

    return CompletionReward(completion_id=completion_output.index, reward=reward, task_reward=task_reward, length_penalty=length_penalty)


def _compute_request_rewards(
    request_output: RequestOutput,
    verification_info: dict,
    task_type: TaskType,
    config: LenRewardsConfig | None,
) -> RequestRewards:
    """
    Computes the rewards and advantages from a single vLLM request output given
    the task type (e.g. math, code, etc.) and information on how to verify all
    completions in the request output.

    Args:
        request_output: The request output to compute the rewards for.
        verification_info: The verification info for the request output.
        task_type: The task type for the request output.
        config: The config for the rewards.

    Returns:
        A dictionary containing the rewards, task rewards, and length penalties
        for each completion in the request output.
    """
    completion_rewards = []
    for output in request_output.outputs:
        args = (output, verification_info, task_type, config)
        completion_rewards.append(_compute_completion_reward(*args))

    # Compute advantage (normalized rewards)
    reward_array = np.array([reward.reward for reward in completion_rewards], dtype=np.float32)
    advantage_array = (reward_array - reward_array.mean()) / (reward_array.std(ddof=1) + 1e-6)
    for completion_reward, advantage in zip(completion_rewards, advantage_array):
        completion_reward.advantage = float(advantage)

    return RequestRewards(request_id=request_output.request_id, rewards=completion_rewards)


def compute_rewards(
    request_outputs: list[RequestOutput],
    verification_infos: list[dict],
    task_types: list[TaskType],
    config: LenRewardsConfig | None,
) -> list[RequestRewards]:
    """
    Computes the rewards and advantages for a list of vLLM request outputs
    given their task types and verification infos.

    Args:
        request_outputs: The request outputs to compute the rewards for.
        verification_infos: The verification infos for the request outputs.
        task_types: The task types for the request outputs.
        config: The config for the rewards.

    Returns:
        A tuple containing dictionaries mapping request IDs to lists of rewards,
        task rewards, length penalties, and advantages.
    """

    max_workers = min(32, len(request_outputs))
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for request, info, task_type in zip(request_outputs, verification_infos, task_types):
            args = (request, info, task_type, config)
            futures.append(executor.submit(_compute_request_rewards, *args))

    return list(future.result() for future in futures)
