import asyncio
import time
from pathlib import Path
from typing import Any

import numpy as np
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion

from zeroband.training.config import ModelConfig
from zeroband.training.orchestrator.config import ClientConfig, SamplingConfig
from zeroband.training.orchestrator.genesys import get_reward_function
from zeroband.utils.logger import get_logger


def setup_client(client_config: ClientConfig) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=client_config.base_url, api_key=client_config.api_key)


async def health_check(client: AsyncOpenAI, timeout: int = 60, interval: int = 10) -> None:
    logger = get_logger()
    logger.info("Checking health of inference pool")
    num_attempts = 0
    while num_attempts * interval < timeout:
        try:
            await client.models.list()
            logger.success("Inference pool is ready")
            return
        except Exception as e:
            num_attempts += 1
            logger.warning(f"Inference pool cannot be reached after {num_attempts} attempt(s) (Error: {e})")
            await asyncio.sleep(interval)
    msg = f"Inference pool is not ready after {num_attempts} attempt(s). Aborting..."
    logger.error(msg)
    raise TimeoutError(msg)


async def reload_weights(client: AsyncOpenAI, path: Path, step: int) -> None:
    """Make a HTTP post request to the vLLM server to reload the weights."""
    logger = get_logger()
    url = str(client.base_url) + "reload_weights"
    model_path = path / f"step_{step}" / "model.pt"
    logger.info(f"Sending request to {url} to reload weights from {model_path}")
    await client._client.post(url=url, json={"model_path": model_path.as_posix()})


async def generate_completion(
    client: AsyncOpenAI,
    model_config: ModelConfig,
    sampling_config: SamplingConfig,
    messages: list[dict[str, str]],
) -> ChatCompletion:
    response = await client.chat.completions.create(
        messages=messages,
        model=model_config.name,
        temperature=sampling_config.temperature,
        top_p=sampling_config.top_p,
        max_tokens=sampling_config.max_tokens,
        logprobs=sampling_config.logprobs,
        seed=sampling_config.seed,
        extra_body={
            "top_k": sampling_config.top_k,
            "min_p": sampling_config.min_p,
            "min_tokens": sampling_config.min_tokens,
        },
    )
    assert len(response.choices) == 1, "Response should always have one choice"
    return response


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
