import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterator, Literal, Sequence

import numpy as np
import requests
from pydantic import BaseModel
from pydantic_config import BaseConfig
from vllm import RequestOutput

from zeroband.inference.genesys import TaskType, get_reward_function
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


class RewardsConfig(BaseConfig):
    len_reward: LenRewardsConfig | None = None


class ModelCompletion(BaseModel):
    index: int
    text: str
    token_ids: Sequence[int]


class ModelOutput(BaseModel):
    request_id: str
    outputs: list[ModelCompletion]


class RewardRequest(BaseModel):
    model_outputs: list[ModelOutput]
    verification_infos: list[dict[str, Any]]
    task_types: list[TaskType]
    config: RewardsConfig | None = None

    def __len__(self) -> int:
        return len(self.model_outputs)

    def __iter__(self) -> Iterator[tuple[ModelOutput, dict[str, Any], TaskType]]:
        for request_output, verification_info, task_type in zip(self.model_outputs, self.verification_infos, self.task_types):
            yield request_output, verification_info, task_type


def unwrap_request_output(request_output: RequestOutput) -> ModelOutput:
    outputs = [ModelCompletion(index=o.index, text=o.text, token_ids=o.token_ids) for o in request_output.outputs]
    return ModelOutput(request_id=request_output.request_id, outputs=outputs)


def vllm_output_to_serializable(
    request_outputs: list[RequestOutput],
    verification_infos: list[dict],
    task_types: list[TaskType],
    config: RewardsConfig | None = None,
) -> RewardRequest:
    model_outputs = [unwrap_request_output(request_output) for request_output in request_outputs]
    return RewardRequest(
        model_outputs=model_outputs,
        verification_infos=verification_infos,
        task_types=task_types,
        config=config,
    )


class CompletionReward(BaseModel):
    completion_id: int  # type(CompletionOutput.index)
    reward: float
    task_reward: float
    advantage: float | None = None


class RequestRewards(BaseModel):
    request_id: str  # type(RequestOutput.request_id)
    rewards: list[CompletionReward]
    task_type: str


class RewardsResponse(BaseModel):
    rewards: list[RequestRewards]


def _compute_completion_reward(
    completion_output: ModelCompletion,
    verification_info: dict,
    task_type: TaskType,
    config: RewardsConfig | None,
) -> CompletionReward:
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
    reward = task_reward

    return CompletionReward(completion_id=completion_output.index, reward=reward, task_reward=task_reward)


def _compute_request_rewards(
    request_output: ModelOutput,
    verification_info: dict,
    task_type: TaskType,
    config: RewardsConfig | None,
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

    return RequestRewards(request_id=request_output.request_id, rewards=completion_rewards, task_type=str(task_type))


def compute_rewards(
    reward_request: RewardRequest,
) -> RewardsResponse:
    remote_url = os.getenv("REWARD_URL", None)
    if remote_url is None:
        max_workers = min(32, len(reward_request))
        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for request, info, task_type in reward_request:
                args = (request, info, task_type, reward_request.config)
                futures.append(executor.submit(_compute_request_rewards, *args))

        return RewardsResponse(rewards=list(future.result() for future in futures))
    else:
        port = os.getenv("REWARD_PORT", 8000)
        remote_auth = os.getenv("REWARD_AUTH", None)
        if remote_auth is None:
            raise ValueError("Remote URL is set but no authentication token provided. Set the REWARD_AUTH environment variable.")

        # TODO: SSL/HTTPS?
        response = requests.post(
            f"http://{remote_url}:{port}/compute_rewards",
            data=json.dumps(reward_request.model_dump()),
            headers={"Authorization": f"Bearer {remote_auth}"},
        )

        if response.status_code != 200:
            logger.error(f"Failed to compute rewards: {response.status_code} - {response.text}")
            raise RuntimeError(f"Failed to compute rewards: {response.status_code} - {response.text}")
        response = RewardsResponse.model_validate(json.loads(response.text))
        return response


def compute_vllm_rewards(
    request_outputs: list[RequestOutput],
    verification_infos: list[dict],
    task_types: list[TaskType],
    config: RewardsConfig | None = None,
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

    reward_request = vllm_output_to_serializable(
        request_outputs=request_outputs,
        verification_infos=verification_infos,
        task_types=task_types,
        config=config,
    )
    return compute_rewards(reward_request).rewards
