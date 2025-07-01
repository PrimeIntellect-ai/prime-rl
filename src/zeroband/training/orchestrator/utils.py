import time
from pathlib import Path
from typing import Any

import numpy as np
from openai import ChatCompletion

from zeroband.training.orchestrator.genesys import get_reward_function
from zeroband.utils.logger import get_logger


def parse_logprobs(chat_completions: list[ChatCompletion]) -> list[list[float]]:
    """Parses the logprobs from a list of chat completions returned by vLLM OAI server."""
    logprobs = []
    for chat_completion in chat_completions:
        assert len(chat_completion.choices) == 1, "Response should always have one choice"
        assert chat_completion.choices[0].logprobs is not None, (
            "Logprobs should not be None. Make sure to set logprobs=True when making the request to /v1/chat/completions"
        )
        logprobs.append([logprob.logprob for logprob in chat_completion.choices[0].logprobs.content])
    return logprobs


def parse_output_tokens(chat_completions: list[ChatCompletion]) -> list[list[int]]:
    """Parses the output token ids from a list of chat completions returned by vLLM OAI server."""
    tokens = []
    for chat_completion in chat_completions:
        assert len(chat_completion.choices) == 1, "Response should always have one choice"
        assert chat_completion.choices[0].logprobs is not None, (
            "Logprobs should not be None. Make sure to set return_tokens_as_token_ids=True in the extra body when making the request to /v1/chat/completions"
        )
        tokens.append([int(token.token.split(":")[-1]) for token in chat_completion.choices[0].logprobs.content])
    return tokens


def parse_completions(chat_completions: list[ChatCompletion]) -> list[str]:
    """Parses the completions from a list of chat completions returned by vLLM OAI server."""
    completions = []
    for chat_completion in chat_completions:
        assert len(chat_completion.choices) == 1, "Response should always have one choice"
        completions.append(chat_completion.choices[0].message.content)
    return completions


def wait_for_weight_checkpoint(path: Path, step: int, interval: int = 1, log_interval: int = 10) -> None:
    logger = get_logger()
    wait_time = 0
    model_path = Path(path) / f"step_{step}" / "model.pt"
    logger.info(f"Waiting for checkpoint for step {step} at {model_path}")
    while True:
        if model_path.exists():
            logger.debug(f"Found checkpoint for step {step} at {model_path}")
            break
        if wait_time % log_interval == 0 and wait_time > 0:  # Every log_interval seconds
            logger.debug(f"Waiting for checkpoint for step {step} at {model_path} for {wait_time} seconds")
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
