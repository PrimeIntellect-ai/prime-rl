import asyncio
from pathlib import Path
from typing import Any

import numpy as np
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from pyarrow import Table
from transformers import AutoTokenizer

from zeroband.training.orchestrator.config import CompletionConfig
from zeroband.training.orchestrator.genesys import get_reward_function
from zeroband.training.parquet import SCHEMA
from zeroband.utils.logger import get_logger


async def health_check(
    client: AsyncOpenAI, timeout: int = 60, interval: int = 10
) -> None:
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
            logger.warning(
                f"Inference pool cannot be reached after {num_attempts} attempt(s) (Error: {e})"
            )
            await asyncio.sleep(interval)
    msg = f"Inference pool is not ready after {num_attempts} attempt(s). Aborting..."
    logger.error(msg)
    raise TimeoutError(msg)


async def load_checkpoint(client: AsyncOpenAI, ckpt_path: Path, step: int) -> None:
    """Make a HTTP post request to the vLLM server to load a checkpoint."""
    logger = get_logger()
    url = str(client.base_url) + "load_checkpoint"
    logger.info(f"Sending load checkpoint request to {url} with ckpt_path {ckpt_path}")
    await client._client.post(url=url, json={"ckpt_path": ckpt_path.as_posix()})


async def generate_completion(
    client: AsyncOpenAI,
    completion_config: CompletionConfig,
    messages: list[dict[str, str]],
) -> ChatCompletion:
    response = await client.chat.completions.create(
        **completion_config.model_dump(),
        messages=messages,
    )
    assert len(response.choices) == 1, "Response should always have one choice"
    return response


def get_parquet(
    prompts: list[str],
    completions: list[str],
    rewards: list[float],
    advantages: list[float],
    temperature: float,
    tokenizer: AutoTokenizer,
) -> Table:
    rows = []
    for prompt, completion, reward, advantage in zip(
        prompts, completions, rewards, advantages
    ):
        input_tokens = tokenizer.encode(prompt)
        output_tokens = tokenizer.encode(completion)
        rows.append(
            {
                "input_tokens": tokenizer.encode(prompt),
                "output_tokens": tokenizer.encode(completion),
                "advantages": advantage,
                "input_logprobs": [0] * len(input_tokens),
                "output_logprobs": [0] * len(output_tokens),
                "temperature": temperature,
            }
        )
    return Table.from_pylist(rows, schema=SCHEMA)


def compute_rewards(
    completions: list[str],
    task_types: list[str],
    verification_infos: list[dict[str, Any]],
) -> list[float]:
    rewards = []
    for completion, task_type, verification_info in zip(
        completions, task_types, verification_infos
    ):
        compute_reward = get_reward_function(task_type)
        reward = compute_reward(completion, verification_info)
        rewards.append(reward)
    return rewards


def compute_advantages(rewards: list[float], samples_per_problem: int) -> list[float]:
    per_problem_rewards = [
        rewards[i : i + samples_per_problem]
        for i in range(0, len(rewards), samples_per_problem)
    ]
    advantages = []
    for problem_rewards in per_problem_rewards:
        reward_array = np.array(problem_rewards)
        problem_advantages = reward_array - reward_array.mean()
        advantages.extend(problem_advantages.tolist())
    return advantages
