from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openai.types.chat import ChatCompletion
from rich.console import Console
from rich.table import Table

from zeroband.training.orchestrator.genesys import TaskType, get_reward_function
from zeroband.training.weights import WeightCheckpointManager
from zeroband.utils.utils import format_num, format_time, wait_for_path


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
    model_path = WeightCheckpointManager.get_model_path(path, step)
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
        "time/infer": "Step Time",
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
