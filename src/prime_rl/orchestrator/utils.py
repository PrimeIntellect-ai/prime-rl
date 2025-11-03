from typing import Any, List, Optional

import openai.types.chat
import pandas as pd
import verifiers as vf
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.completion_usage import CompletionUsage
from rich.console import Console
from rich.table import Table

from prime_rl.orchestrator.config import SamplingConfig
from prime_rl.utils.utils import (
    format_num,
    format_time,
)


def get_train_sampling_args(sampling_config: SamplingConfig) -> dict:
    # Convert SamplingConfig to vLLM OAI sampling args
    # https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters_2
    sampling_args = dict(sampling_config)
    sampling_args["top_p"] = 1.0
    sampling_args["logprobs"] = True
    sampling_args["extra_body"] = {
        "return_tokens_as_token_ids": True,
        "top_k": -1,
        "min_p": 0.0,
    }
    sampling_args["extra_body"]["min_tokens"] = sampling_args.pop("min_tokens")
    sampling_args["extra_body"]["repetition_penalty"] = sampling_args.pop("repetition_penalty")
    return sampling_args


def monkey_patch_chat_completion_logprobs():
    """
    At large batch sizes and context, constructing OAI's Pydantic model
    ChatCompletion with logprobs is causing heavy CPU overhead (~200ms per
    object at 32K context, which translates to >10min overhead at 4K batch
    size). This function monkey-patches the OAI type and verifiers'
    post-processing utils to avoid validating the complex logprobs field.
    """

    class ChoiceAny(Choice):
        """Same as openai.types.chat.chat_completion.Choice, but without type validation for logprobs field."""

        logprobs: Optional[Any] = None

    class ChatCompletionAny(ChatCompletion):
        """Same as openai.types.chat.chat_completion.ChatCompletion, but but using ChoiceAny instead of Choice."""

        choices: List[ChoiceAny]  # type: ignore

    # Patch OAI types
    openai.types.chat.chat_completion.Choice = ChoiceAny
    openai.types.chat.chat_completion.ChatCompletion = ChatCompletionAny

    # Patch verifiers parse_chat_completion_logprobs
    def patched_parse_chat_completion_logprobs(chat_completion: ChatCompletionAny) -> list[float]:
        """Same as verifiers.utils.processing_utils.parse_chat_completion_logprobs, but using arbitrary logprobs type."""
        assert len(chat_completion.choices) == 1, "Response should always have one choice"
        assert chat_completion.choices[0].logprobs is not None, (
            "Logprobs should not be None. Make sure to set logprobs=True in the extra body when making the request to /v1/chat/completions"
        )
        assert chat_completion.choices[0].logprobs["content"] is not None, (
            "Logprob content should not be None. Make sure to set logprobs=True in the extra body when making the request to /v1/chat/completions"
        )
        logprobs = [logprob["logprob"] for logprob in chat_completion.choices[0].logprobs["content"]]
        return logprobs

    # Patch verifiers parse_chat_completion_logprobs
    def patched_parse_chat_completion_tokens(chat_completion: ChatCompletionAny) -> list[int]:
        """Same as verifiers.utils.processing_utils.parse_chat_completion_tokens, but using arbitrary logprobs type."""
        assert len(chat_completion.choices) == 1, "Response should always have one choice"
        assert chat_completion.choices[0].logprobs is not None, (
            "Logprobs should not be None. Make sure to set logprobs=True in the extra body when making the request to /v1/chat/completions"
        )
        assert chat_completion.choices[0].logprobs["content"] is not None, (
            "Logprob content should not be None. Make sure to set logprobs=True in the extra body when making the request to /v1/chat/completions"
        )
        tokens = [
            # tokens are token_id:<int> because we request `return_tokens_as_token_ids` from vllm in GRPOTrainer
            int(token["token"].split(":")[-1])
            for token in chat_completion.choices[0].logprobs["content"]
        ]
        return tokens

    vf.utils.processing_utils.parse_chat_completion_logprobs = patched_parse_chat_completion_logprobs
    vf.utils.processing_utils.parse_chat_completion_tokens = patched_parse_chat_completion_tokens


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
