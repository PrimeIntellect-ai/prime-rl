from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from openai.types.chat import ChatCompletion
from rich.console import Console
from rich.table import Table

from prime_rl.orchestrator.client import tokenize
from prime_rl.orchestrator.data import GeneratedSample
from prime_rl.orchestrator.genesys import TaskType, get_reward_function
from prime_rl.utils.utils import format_num, format_time, get_weight_ckpt_model_path, wait_for_path


def parse_completion_logprobs(chat_completion: ChatCompletion) -> list[float]:
    """Parses the completion logprobs from a vLLM chat completion"""
    assert len(chat_completion.choices) == 1, "Response should always have one choice"
    assert chat_completion.choices[0].logprobs is not None, (
        "Logprobs should not be None. Make sure to set logprobs=True in the extra body when making the request to /v1/chat/completions"
    )
    assert chat_completion.choices[0].logprobs.content is not None, (
        "Logprob content should not be None. Make sure to set logprobs=True in the extra body when making the request to /v1/chat/completions"
    )
    logprobs = [logprob.logprob for logprob in chat_completion.choices[0].logprobs.content]
    return logprobs


def parse_completion_tokens(chat_completion: ChatCompletion) -> list[int]:
    """Parses the output token ids from a list of chat completions returned by vLLM OAI server."""
    assert len(chat_completion.choices) == 1, "Response should always have one choice"
    assert chat_completion.choices[0].logprobs is not None, (
        "Logprobs should not be None. Make sure to set logprobs=True in the extra body when making the request to /v1/chat/completions"
    )
    assert chat_completion.choices[0].logprobs.content is not None, (
        "Logprob content should not be None. Make sure to set logprobs=True in the extra body when making the request to /v1/chat/completions"
    )
    tokens = [int(token.token.split(":")[-1]) for token in chat_completion.choices[0].logprobs.content]
    return tokens


async def process_env_results(outputs, client, config):
    """Hotfix `process_env_results` for using vLLM prompt and completion tokens/ logprobs"""

    all_prompt_tokens = []
    all_completion_tokens = []
    all_completion_logprobs = []
    prompt_masks = []
    completion_masks = []

    assert all(len(s["responses"]) == 1 for s in outputs["state"])
    chat_completions = [s["responses"][0] for s in outputs["state"]]
    for prompt, chat_completion in zip(outputs["prompt"], chat_completions):
        # Tokenize prompt using vLLM server
        prompt_tokens = await tokenize(client, config.model, prompt)

        # Parse vLLM output tokens and logprobs
        completion_tokens = parse_completion_tokens(chat_completion)
        completion_logprobs = parse_completion_logprobs(chat_completion)

        # Truncate sequences
        if len(prompt_tokens) + len(completion_tokens) > config.seq_len:
            if len(prompt_tokens) > config.seq_len:
                prompt_tokens = prompt_tokens[: config.seq_len]
            completion_tokens = completion_tokens[: config.seq_len - len(prompt_tokens)]
            completion_logprobs = completion_logprobs[: config.seq_len - len(prompt_tokens)]

        prompt_mask = [0] * len(prompt_tokens)
        completion_mask = [1] * len(completion_tokens)

        all_prompt_tokens.append(prompt_tokens)
        all_completion_tokens.append(completion_tokens)
        all_completion_logprobs.append(completion_logprobs)
        prompt_masks.append(prompt_mask)
        completion_masks.append(completion_mask)

    return {
        "prompt_tokens": all_prompt_tokens,
        "completion_tokens": all_completion_tokens,
        "completion_logprobs": all_completion_logprobs,
        "prompt_masks": prompt_masks,
        "completion_masks": completion_masks,
    }


def parse_completions(chat_completions: list[ChatCompletion]) -> list[str]:
    """Parses the completions from a list of chat completions returned by vLLM OAI server."""
    completions = []
    for chat_completion in chat_completions:
        assert len(chat_completion.choices) == 1, "Response should always have one choice"
        completions.append(chat_completion.choices[0].message.content)
    return completions


def wait_for_weight_checkpoint(path: Path, step: int, interval: int = 1, log_interval: int = 10) -> None:
    model_path = get_weight_ckpt_model_path(path, step)
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


def compute_advantages(per_problem_rewards: list[list[float]]) -> list[float]:
    advantages = []
    for problem_rewards in per_problem_rewards:
        reward_array = np.array(problem_rewards)
        problem_advantages = reward_array - reward_array.mean()
        advantages.extend(problem_advantages.tolist())
    return advantages


def print_benchmark(history: dict[str, list[Any]]) -> None:
    """
    Print benchmark results as rich table. Shows formatted values for the
    inference throughput and overall step time. First first N rows show the
    per-step values, and the last row shows the mean, std, min, and max values.
    """
    history.pop("step")
    assert all(len(v) for v in history.values()), "All metrics must have logged the same number of steps"

    # Turn metric history into pd.DataFrame
    df = pd.DataFrame(dict(history.items()))
    columns = {
        "perf/infer/throughput": "Throughput",
        "time/orchestrator": "Step Time",
    }
    df = df[columns.keys()].rename(columns=columns)
    df = df.iloc[1:]  # Exclude first row

    # Setup console
    console = Console()
    table = Table(title="Benchmark")

    # Add columns
    table.add_column("Step", justify="right")
    for col in df.columns:
        table.add_column(col, justify="center", style="magenta")

    # Add formatted rows
    formatted_df = pd.DataFrame(columns=df.columns)
    formatted_df["Step Time"] = df["Step Time"].apply(format_time)
    formatted_df["Throughput"] = df["Throughput"].apply(format_num, precision=2)
    for step, row in formatted_df.iterrows():
        table.add_row(*([str(step)] + [str(x) for x in row]))

    # Separator
    table.add_row(*([""] * len(row)))

    # Add row for formatted, aggregated statistics
    mean_df = df.describe().loc[["mean", "std", "min", "max"], :]
    formatted_mean_df = pd.DataFrame(columns=mean_df.columns)
    formatted_mean_df["Step Time"] = mean_df["Step Time"].apply(format_time)
    formatted_mean_df["Throughput"] = mean_df["Throughput"].apply(format_num, precision=2)
    mean_row = ["Overall"] + formatted_mean_df.T.apply(
        lambda row: f"{row['mean']} ± {row['std']} [{row['min']}, {row['max']}]", axis=1
    ).tolist()
    table.add_row(*mean_row)

    # Display table
    console.print(table)


def flatten_keep(lst: list, keep_indices: list[int], group_size: int):
    return [item for i in keep_indices for item in lst[i * group_size : (i + 1) * group_size]]


def create_generated_samples(
    prompt_tokens: List[List[int]],
    completion_tokens: List[List[int]],
    completion_logprobs: List[List[float]],
    prompt_masks: List[List[int]],
    completion_masks: List[List[int]],
    rewards: List[float],
    advantages: List[float],
) -> List[GeneratedSample]:
    return [
        GeneratedSample(
            prompt_tokens=pt,
            completion_tokens=ct,
            completion_logprobs=cl,
            prompt_masks=pm,
            completion_masks=cm,
            reward=r,
            advantages=a,
        )
        for (pt, ct, cl, pm, cm, r, a) in zip(
            prompt_tokens, completion_tokens, completion_logprobs, prompt_masks, completion_masks, rewards, advantages
        )
    ]


def unpack_generated_samples(
    generated_samples: List[GeneratedSample],
) -> Dict[str, List[Any]]:
    return {
        "prompt_tokens": [gs.prompt_tokens for gs in generated_samples],
        "completion_tokens": [gs.completion_tokens for gs in generated_samples],
        "completion_logprobs": [gs.completion_logprobs for gs in generated_samples],
        "prompt_masks": [gs.prompt_masks for gs in generated_samples],
        "completion_masks": [gs.completion_masks for gs in generated_samples],
        "reward": [gs.reward for gs in generated_samples],
        "advantages": [gs.advantages for gs in generated_samples],
    }
