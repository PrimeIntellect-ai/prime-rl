from typing import Any, AsyncContextManager

import pandas as pd
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.completion_usage import CompletionUsage
from rich.console import Console
from rich.table import Table
from verifiers.utils.async_utils import maybe_semaphore

from prime_rl.orchestrator.config import SamplingConfig
from prime_rl.utils.utils import (
    format_num,
    format_time,
)

SEMAPHORE: AsyncContextManager | None = None


async def set_semaphore(limit: int):
    global SEMAPHORE
    SEMAPHORE = await maybe_semaphore(limit)


async def get_semaphore() -> AsyncContextManager:
    global SEMAPHORE
    assert SEMAPHORE is not None, "Semaphore not set"
    return SEMAPHORE


def get_sampling_args(sampling_config: SamplingConfig) -> dict:
    # Convert SamplingConfig to vLLM OAI sampling args
    # https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters_2
    sampling_args = dict(sampling_config)
    sampling_args["top_p"] = 1.0
    sampling_args["logprobs"] = True
    sampling_args["extra_body"] = {
        **sampling_config.extra_body,
        "return_token_ids": True,  # Always return token IDs
        "top_k": -1,
        "min_p": 0.0,
    }
    sampling_args["extra_body"]["min_tokens"] = sampling_args.pop("min_tokens")
    sampling_args["extra_body"]["repetition_penalty"] = sampling_args.pop("repetition_penalty")
    return sampling_args


def parse_num_completion_tokens(responses: list[list[ChatCompletion]]) -> list[int]:
    """Parses the number of tokens from a list of chat completions returned by OAI API."""
    all_num_completion_tokens = []
    for response in responses:
        num_completion_tokens = 0
        for chat_completion in response:
            assert isinstance(chat_completion, ChatCompletion)
            assert chat_completion.usage is not None, "Usage should be present in the response"
            usage = chat_completion.usage
            assert isinstance(usage, CompletionUsage)
            num_completion_tokens += usage.completion_tokens
        all_num_completion_tokens.append(num_completion_tokens)
    assert len(all_num_completion_tokens) == len(responses), (
        "Number of completion tokens should be the same as the number of responses"
    )
    return all_num_completion_tokens


def parse_is_truncated_completions(responses: list[list[ChatCompletion]]) -> list[bool]:
    """Parses whether the completions were truncated from a list of (multi-turn) OAI chat completions"""
    all_is_truncated = []
    for response in responses:
        is_truncated = False
        for chat_completion in response:
            assert isinstance(chat_completion, ChatCompletion)
            assert len(chat_completion.choices) == 1, "Response should always have one choice"
            choice = chat_completion.choices[0]
            assert isinstance(choice, Choice)
            if choice.finish_reason == "length":
                is_truncated = True
        all_is_truncated.append(is_truncated)
    return all_is_truncated


def build_step_metrics(
    *,
    results_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    val_results_df: pd.DataFrame | None,
    num_tokens: int,
    batch_size: int,
    rollouts_per_example: int,
    progress_total_tokens: int,
    progress_total_samples: int,
    progress_total_problems: int,
    progress_step: int,
    ckpt_step: int,
    throughput: float,
    step_time: float,
    generate_completions_time: float,
    save_ckpt_time: float,
    scheduler_metrics: dict[str, Any],
    buffer_metrics: dict[str, Any],
    event_loop_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build the metrics dict for logging a training step."""
    # Compute solve all and none
    solve_all = (
        results_df.groupby("example_id")
        .apply(lambda x: x.reward.sum() == rollouts_per_example, include_groups=False)
        .mean()
    )
    solve_none = results_df.groupby("example_id").apply(lambda x: x.reward.sum() == 0, include_groups=False).mean()
    effective_batch_size = 1 - solve_none - solve_all

    to_log = {
        # Progress metrics
        "progress/tokens": num_tokens,
        "progress/samples": batch_size,
        "progress/problems": batch_size // rollouts_per_example,
        "progress/total_tokens": progress_total_tokens,
        "progress/total_samples": progress_total_samples,
        "progress/total_problems": progress_total_problems,
        "progress/ckpt_step": ckpt_step,
        # Sequence length metrics
        "seq_len/mean": results_df.groupby("example_id").seq_len.mean().mean(),
        "seq_len/max": results_df.groupby("example_id").seq_len.mean().max(),
        "seq_len/min": results_df.groupby("example_id").seq_len.mean().min(),
        "prompt_len/mean": results_df.groupby("example_id").prompt_len.mean().mean(),
        "prompt_len/max": results_df.groupby("example_id").prompt_len.mean().max(),
        "prompt_len/min": results_df.groupby("example_id").prompt_len.mean().min(),
        "completion_len/mean": results_df.groupby("example_id").completion_len.mean().mean(),
        "completion_len/max": results_df.groupby("example_id").completion_len.mean().max(),
        "completion_len/min": results_df.groupby("example_id").completion_len.mean().min(),
        "is_truncated/mean": results_df.groupby("example_id").is_truncated.mean().mean(),
        "is_truncated/max": results_df.groupby("example_id").is_truncated.mean().max(),
        "is_truncated/min": results_df.groupby("example_id").is_truncated.mean().min(),
        # Turn metrics
        "num_turns/mean": results_df.groupby("example_id").num_turns.mean().mean(),
        "num_turns/max": results_df.groupby("example_id").num_turns.mean().max(),
        "num_turns/min": results_df.groupby("example_id").num_turns.mean().min(),
        # Verifier timing metrics
        "generation_ms/mean": results_df.groupby("example_id").generation_ms.mean().mean(),
        "generation_ms/max": results_df.groupby("example_id").generation_ms.mean().max(),
        "generation_ms/min": results_df.groupby("example_id").generation_ms.mean().min(),
        "scoring_ms/mean": results_df.groupby("example_id").scoring_ms.mean().mean(),
        "scoring_ms/max": results_df.groupby("example_id").scoring_ms.mean().max(),
        "scoring_ms/min": results_df.groupby("example_id").scoring_ms.mean().min(),
        # Performance metrics
        "perf/throughput": throughput,
        # Train reward
        "reward/mean": results_df.reward.mean(),
        # Batch metrics
        "batch/solve_none": solve_none,
        "batch/solve_all": solve_all,
        "batch/effective_batch_size": effective_batch_size,
        # Error metrics
        "error/mean": (~results_df.error.isna()).mean(),
        **{f"error/{error}": error_rate for error, error_rate in results_df.error.dropna().value_counts(normalize=True).items()},
        # Env metrics
        **{f"metrics/{metric}": metrics_df[metric].mean() for metric in metrics_df.columns},
        # Time metrics
        "time/step": step_time,
        "time/generate_completions": generate_completions_time,
        "time/save_ckpt": save_ckpt_time,
        # Scheduler metrics
        **scheduler_metrics,
        # Buffer metrics
        **buffer_metrics,
        # Event loop lag metrics
        **event_loop_metrics,
        # W&B axis
        "step": progress_step,
    }

    # If more than one env, add per-env metrics
    if results_df.task.nunique() > 1:
        per_env_reward = results_df.groupby("task").reward.mean().to_dict()
        to_log.update({f"reward/{env}": reward for env, reward in per_env_reward.items()})

        per_env_ratio = results_df.task.value_counts(normalize=True).to_dict()
        to_log.update({f"batch/{env}": ratio for env, ratio in per_env_ratio.items()})

    # Optionally, add val metrics
    if val_results_df is not None:
        to_log.update({"val_reward/mean": val_results_df.reward.mean()})

        if val_results_df.task.nunique() > 1:
            per_env_reward = val_results_df.groupby("task").reward.mean().to_dict()
            to_log.update({f"val_reward/{env}": reward for env, reward in per_env_reward.items()})

            per_env_ratio = val_results_df.task.value_counts(normalize=True).to_dict()
            to_log.update({f"val_batch/{env}": ratio for env, ratio in per_env_ratio.items()})

    return to_log


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
