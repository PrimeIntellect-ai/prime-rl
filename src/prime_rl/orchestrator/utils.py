from pathlib import Path
from typing import Any

import pandas as pd
from openai.types.chat import ChatCompletion
from rich.console import Console
from rich.table import Table
from verifiers.types import State

from prime_rl.utils.utils import (
    format_num,
    format_time,
    get_weight_ckpt_model_path,
    wait_for_path,
)


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


def parse_truncated_completions(states: list[State]) -> list[bool]:
    is_truncated = []
    for state in states:
        assert "responses" in state, "Responses should be present in the state"
        assert all(isinstance(r, ChatCompletion) for r in state["responses"]), (
            "Responses should be ChatCompletion objects"
        )
        for chat_completion in state["responses"]:
            assert len(chat_completion.choices) == 1, "Response should always have one choice"
            if chat_completion.choices[0].finish_reason == "length":
                is_truncated.append(True)
            else:
                is_truncated.append(False)
    return is_truncated


def wait_for_weight_checkpoint(path: Path, step: int, interval: int = 1, log_interval: int = 10) -> None:
    model_path = get_weight_ckpt_model_path(path, step)
    wait_for_path(model_path, interval, log_interval)


def apply_length_bonus(
    rewards: list[float], completion_lengths: list[int], rollouts_per_prompt: int, length_bonus: float
) -> list[float]:
    """Return a new reward list where the shortest *correct* rollout(s) in each
    fully correct group receive a bonus."""

    assert len(rewards) == len(completion_lengths), "Rewards and lengths must align"

    new_rewards = list(rewards)
    for start in range(0, len(rewards), rollouts_per_prompt):
        group_rewards = new_rewards[start : start + rollouts_per_prompt]
        if sum(group_rewards) == rollouts_per_prompt:
            group_lengths = completion_lengths[start : start + rollouts_per_prompt]
            min_len = min(group_lengths)
            for idx, length in enumerate(group_lengths):
                if length == min_len:
                    new_rewards[start + idx] += length_bonus
    return new_rewards


def process_rewards(
    rewards: list[float], completion_lengths: list[int], rollouts_per_prompt: int, length_bonus: float
) -> list[float]:
    return apply_length_bonus(rewards, completion_lengths, rollouts_per_prompt, length_bonus)


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
        "perf/throughput": "Throughput",
        "time/step": "Step Time",
    }
    df = df.rename(columns=columns)
    df = df[list(columns.values())]
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
    num_table_columns = 1 + len(df.columns)
    table.add_row(*([""] * num_table_columns))

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
