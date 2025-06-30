import asyncio
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from transformers import AutoTokenizer

from zeroband.training.config import ModelConfig
from zeroband.training.orchestrator.config import ClientConfig, SamplingConfig
from zeroband.training.orchestrator.genesys import get_reward_function
from zeroband.utils.logger import get_logger
from zeroband.training.data import BatchOutput


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


async def load_checkpoint(client: AsyncOpenAI, ckpt_path: Path, step: int) -> None:
    """Make a HTTP post request to the vLLM server to load a checkpoint."""
    logger = get_logger()
    url = str(client.base_url) + "load_checkpoint"
    ckpt_path = ckpt_path / f"step_{step}" / "model.pt"
    logger.info(f"Sending load checkpoint request to {url} with ckpt_path {ckpt_path}")
    await client._client.post(url=url, json={"ckpt_path": ckpt_path.as_posix()})


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


def wait_for_checkpoint(ckpt_path: Path, step: int, interval: int = 1, log_interval: int = 10) -> None:
    logger = get_logger()
    wait_time = 0
    ckpt_path = Path(ckpt_path) / f"step_{step}" / "model.pt"
    logger.info(f"Waiting for checkpoint for step {step} at {ckpt_path}")
    while True:
        if ckpt_path.exists():
            logger.info(f"Found checkpoint for step {step} at {ckpt_path}")
            break
        if wait_time % log_interval == 0 and wait_time > 0:  # Every log_interval seconds
            logger.info(f"Waiting for checkpoint for step {step} at {ckpt_path} for {wait_time} seconds")
        time.sleep(interval)
        wait_time += interval


def prepare_sample(prompt: str, completion: str, advantage: float, max_seq_len: int, tokenizer: AutoTokenizer):
    """
    Prepare a problem and pad it for training.
    Tokenize and
    """

    input_tokens = torch.tensor(tokenizer.encode(prompt))
    output_tokens = torch.tensor(tokenizer.encode(completion))

    inputs_ids = torch.cat([input_tokens, output_tokens], dim=0)
    total_tokens = inputs_ids.shape[0]

    loss_mask = torch.cat([torch.zeros(len(input_tokens)), torch.ones(len(output_tokens))], dim=0).int()

    if inputs_ids.shape[0] > max_seq_len:
        inputs_ids = inputs_ids[:max_seq_len]
        loss_mask = loss_mask[:max_seq_len].int()
        advantages = torch.tensor(advantage).repeat(max_seq_len).float()
    else:
        padding_len = max_seq_len - inputs_ids.shape[0]
        inputs_ids = torch.cat([inputs_ids, torch.full((padding_len,), tokenizer.pad_token_id)])
        loss_mask = torch.cat([loss_mask, torch.zeros(padding_len)]).int()
        advantages = torch.tensor(advantage).repeat(inputs_ids.shape[0]).float()
        advantages = torch.cat([advantages, torch.zeros(padding_len)])

    advantages = torch.tensor(advantage).repeat(max_seq_len).float()

    logprobs = torch.ones_like(inputs_ids).float()
    position_ids = torch.arange(max_seq_len)

    return {
        "input_ids": inputs_ids,
        "advantages": advantages,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "logprobs": logprobs,
        "total_tokens": total_tokens,
    }


def prepare_micro_batch(samples: list[BatchOutput], temperature: float):
    micro_batch = {}

    for key in ["input_ids", "advantages", "loss_mask", "position_ids", "logprobs"]:
        micro_batch[key] = torch.stack([sample[key] for sample in samples], dim=0)

    micro_batch["temperature"] = temperature
    micro_batch["total_tokens"] = sum([sample["total_tokens"] for sample in samples])

    return micro_batch


def prepare_batch(
    prompts: list[str],
    completions: list[str],
    advantages: list[float],
    temperature: float,
    tokenizer: AutoTokenizer,
    micro_bs: int,
    max_seq_len: int,
    n_data_ranks: int,
) -> list[list[BatchOutput]]:
    """
    Prepare a batch of problems for each GPU. Each batch is a list of micro batches.
    """

    assert len(prompts) == len(completions) == len(advantages), (
        "Prompts, completions, and advantages must have the same length"
    )
    batch_size = len(prompts)

    assert batch_size % (micro_bs * n_data_ranks) == 0, "Batch size must be divisible by micro batch size"
    per_gpu_micro_batches = batch_size // (n_data_ranks * micro_bs)

    batches_per_gpu = []
    for _ in range(n_data_ranks):
        batches = []
        for _ in range(per_gpu_micro_batches):
            micro_batches = []
            for _ in range(micro_bs):
                sample = prepare_sample(prompts.pop(), completions.pop(), advantages.pop(), max_seq_len, tokenizer)
                micro_batches.append(sample)
            batches.append(prepare_micro_batch(micro_batches, temperature))

        batches_per_gpu.append(batches)

    return batches_per_gpu


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
