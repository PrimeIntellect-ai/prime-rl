from pathlib import Path
from typing import Any

import pandas as pd
import torch
from openai.types.chat import ChatCompletion
from rich.console import Console
from rich.table import Table
from transformers import AutoTokenizer
from vllm import LLM
from vllm.model_executor.model_loader.utils import process_weights_after_loading

from prime_rl.orchestrator.client import tokenize
from prime_rl.orchestrator.genesys import TaskType, get_reward_function
from prime_rl.utils.logger import get_logger
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


def reload_model_weights(llm: LLM, ckpt_path: Path):
    # Access the internal model from vLLM
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    # Load state dict
    logger = get_logger()
    logger.info(f"Reloading model weights from {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")

    # Create a better weight iterator that filters out empty keys and handles prefixes
    def weights_iterator():
        for key, value in state_dict.items():
            # Skip empty keys
            if not key:
                continue
            yield key, value

    model.load_weights(weights_iterator())

    # Process weights after loading (important for some models)
    model_config = llm.llm_engine.model_config
    device = next(model.parameters()).device
    process_weights_after_loading(model, model_config, device)

    return llm


def format_prompts(
    prompts: list[str],
    tokenizer: AutoTokenizer,
    enable_thinking: bool = True,
    tokenize: bool = False,
) -> list[str] | list[list[int]]:
    """
    Formats a batch of raw prompts. Relies on the default chat template of the
    LLM's tokenizer to call `apply_chat_template`. We call with
    `add_generation_prompt=True` to add the generation prompt to the beginning
    of the prompt. We also call with `enable_thinking=True` to enable thinking
    for models that support it. For example, for `Qwen/QwQ-32B` this will add an
    unclosed `</think>` tag to the beginning of the system response.

    Args:
        prompts: A list of raw prompts.
        target_lengths: A list of target lengths (will be [-1, -1, ...] if no length rewards are configured).
        len_rewards_config: A configuration for length rewards. If `None`, no length rewards are configured.
        tokenizer: Any HF tokenizer instance
        enable_thinking: Whether to enable thinking for the model. Used by the `apply_chat_template` to prepend a thinking prompt (for some models)
        tokenize: Whether to tokenize the formatted prompts. If True, returns BatchEncoding; if False (default), returns list[str].

    Returns:
        A list of formatted prompts if tokenize=False, or a BatchEncoding if tokenize=True.
    """
    # No length prompt additions, just use the prompts as is
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]

    # Apply chat template
    formatted_prompts = tokenizer.apply_chat_template(
        messages, tokenize=tokenize, enable_thinking=enable_thinking, add_generation_prompt=True
    )

    if not tokenize:
        for i, _formatted_prompt in enumerate(formatted_prompts):
            if tokenizer.bos_token and _formatted_prompt.startswith(tokenizer.bos_token):
                formatted_prompts[i] = _formatted_prompt[len(tokenizer.bos_token) :]

    return formatted_prompts
