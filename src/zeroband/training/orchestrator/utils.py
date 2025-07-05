from pathlib import Path
from typing import Any

import numpy as np
from openai.types.chat import ChatCompletion

from zeroband.training.orchestrator.genesys import TaskType, get_reward_function
from zeroband.utils.utils import wait_for_path


def parse_logprobs(chat_completions: list[ChatCompletion]) -> list[list[float]]:
    """Parses the logprobs from a list of chat completions returned by vLLM OAI server."""
    logprobs = []
    for chat_completion in chat_completions:
        assert len(chat_completion.choices) == 1, "Response should always have one choice"
        assert chat_completion.choices[0].logprobs is not None, (
            "Logprobs should not be None. Make sure to set logprobs=True in the extra body when making the request to /v1/chat/completions"
        )
        assert chat_completion.choices[0].logprobs.content is not None, (
            "Logprob content should not be None. Make sure to set logprobs=True in the extra body when making the request to /v1/chat/completions"
        )
        logprobs.append([logprob.logprob for logprob in chat_completion.choices[0].logprobs.content])
    return logprobs


def parse_output_tokens(chat_completions: list[ChatCompletion]) -> list[list[int]]:
    """Parses the output token ids from a list of chat completions returned by vLLM OAI server."""
    tokens = []
    for chat_completion in chat_completions:
        assert len(chat_completion.choices) == 1, "Response should always have one choice"
        assert chat_completion.choices[0].logprobs is not None, (
            "Logprobs should not be None. Make sure to set logprobs=True in the extra body when making the request to /v1/chat/completions"
        )
        assert chat_completion.choices[0].logprobs.content is not None, (
            "Logprob content should not be None. Make sure to set logprobs=True in the extra body when making the request to /v1/chat/completions"
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
    model_path = Path(path) / f"step_{step}" / "model.pt"
    wait_for_path(model_path, interval, log_interval)


def compute_rewards(
    completions: list[str],
    task_types: list[TaskType],
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
