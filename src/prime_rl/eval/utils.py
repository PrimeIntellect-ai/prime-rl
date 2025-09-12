import asyncio
import json
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from openai import AsyncOpenAI
from verifiers import load_environment
from verifiers.types import GenerateOutputs, Messages

from prime_rl.eval.config import OfflineEvalConfig
from prime_rl.orchestrator.config import ClientConfig, EvalConfig, EvalSamplingConfig, ModelConfig
from prime_rl.orchestrator.utils import parse_is_truncated_completions, parse_num_completion_tokens
from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor import get_monitor
from prime_rl.utils.utils import capitalize, get_eval_dir, get_step_path


def compute_pass_at_k(rewards: list[int]) -> dict[str, float]:
    total_attempts = len(rewards)
    k = total_attempts // 2

    if k == 0:
        return {"pass@1": float(any(reward == 1.0 for reward in rewards))}

    num_trials = 100
    pass_rates = []

    for _ in range(num_trials):
        sampled_rewards = np.random.choice(rewards, size=k, replace=False)
        pass_rate = float(any(reward == 1.0 for reward in sampled_rewards))
        pass_rates.append(pass_rate)

    return {f"pass@{k}": float(np.mean(pass_rates))}


def prepare_sampling_args(sampling_config: EvalSamplingConfig, client_config: ClientConfig) -> dict[str, Any]:
    """Prepare sampling args for the client."""
    # Initialize sampling args
    sampling_args: dict[str, Any] = {}

    # Apply sampling arguments, if specified
    if sampling_config.temperature is not None:
        sampling_args["temperature"] = sampling_config.temperature
    if sampling_config.max_tokens is not None:
        sampling_args["max_tokens"] = sampling_config.max_tokens
    if sampling_config.top_p is not None:
        sampling_args["top_p"] = sampling_config.top_p
    if sampling_config.reasoning_effort is not None:
        sampling_args["reasoning_effort"] = sampling_config.reasoning_effort

    if client_config.server_type == "vllm":
        # Always return logprobs and token IDs from vLLM server
        sampling_args["logprobs"] = True
        extra_body: dict[str, Any] = {"return_tokens_as_token_ids": True}

        # Apply vLLM-specific sampling arguments, if specified
        if sampling_config.top_k is not None:
            extra_body["top_k"] = sampling_config.top_k
        if sampling_config.min_p is not None:
            extra_body["min_p"] = sampling_config.min_p
        if sampling_config.min_tokens is not None:
            extra_body["min_tokens"] = sampling_config.min_tokens
        if sampling_config.repetition_penalty is not None:
            extra_body["repetition_penalty"] = sampling_config.repetition_penalty

        sampling_args["extra_body"] = extra_body

    return sampling_args


def normalize_prompt(messages: Messages):
    if not isinstance(messages, list):
        return messages
    sanitized_messages = []
    for m in messages:
        if isinstance(m, str):
            sanitized_messages.append({"role": m["role"], "content": [{"type": "text", "text": m}]})
        elif "content" in m and isinstance(m["content"], str):
            sanitized_messages.append({"role": m["role"], "content": [{"type": "text", "text": m["content"]}]})
        else:
            sanitized_messages.append(m)
    return sanitized_messages


def normalize_completion(messages: Messages):
    if not isinstance(messages, list):
        return messages
    sanitized_messages = []
    for m in messages:
        tool_calls = [
            json.dumps(tc.model_dump())  # type: ignore
            for tc in m.get("tool_calls", [])
        ]
        # Ensure tool_calls is always a list of strings, never empty with null inference
        if not tool_calls:
            tool_calls = [""]  # Add empty string to maintain list<string> type

        new_m = {
            "role": m["role"],
            "content": m.get("content", ""),
            "tool_calls": tool_calls,
            "tool_call_id": m.get("tool_call_id", ""),
        }
        sanitized_messages.append(new_m)
    return sanitized_messages


# Adapted from https://github.com/willccbb/verifiers/blob/b4d851db42cebbab2358b827fd0ed19773631937/verifiers/envs/environment.py#L523
def make_dataset(results: GenerateOutputs) -> Dataset:
    """
    Make a dataset from the evaluation results.
    """
    results_dict = {
        "prompt": [normalize_prompt(prompt) for prompt in results.prompt],
        "completion": [normalize_completion(completion) for completion in results.completion],
        "answer": results.answer,
        "task": results.task,
        "reward": results.reward,
        "info": [json.dumps(info) for info in results.info],
    }

    return Dataset.from_dict(results_dict)


async def run_eval(
    client: AsyncOpenAI,
    eval_id: str,
    env_args: dict,
    num_examples: int,
    rollouts_per_example: int,
    max_concurrent: int,
    save_to_disk: bool,
    output_dir: Path,
    ckpt_step: int,
    model_config: ModelConfig,
    sampling_config: EvalSamplingConfig,
    client_config: ClientConfig,
    step: int | None = None,
) -> None:
    # Get the logger
    logger = get_logger()
    monitor = get_monitor()
    assert logger is not None
    eval_start_time = time.time()

    # Load the eval environment
    load_eval_start_time = time.time()
    vf_eval = load_environment(eval_id, **env_args)
    load_eval_time = time.time() - load_eval_start_time
    logger.debug(f"Loaded eval environment in {load_eval_time:.2f}s")

    # Build inputs dataset (mirror Environment.evaluate but async)
    if vf_eval.eval_dataset is None:
        logger.warning(f"Did not find eval dataset for {eval_id}, falling back to train dataset")
        dataset = vf_eval.get_dataset(n=num_examples)
    else:
        dataset = vf_eval.get_eval_dataset(n=num_examples)

    # Convert to list of examples
    assert dataset is not None
    examples = dataset.to_list()
    example_ids = list(range(len(examples)))

    # Duplicate examples `rollouts_per_example` times
    if rollouts_per_example > 1:
        example_ids = [example_id for example_id in example_ids for _ in range(rollouts_per_example)]
        examples = [example for example in examples for _ in range(rollouts_per_example)]

    # Prepare sampling arguments
    sampling_args = prepare_sampling_args(sampling_config, client_config)

    logger.debug(
        f"Evaluating {eval_id} (num_examples={len(examples)}, rollouts_per_example={rollouts_per_example}) with args {env_args}"
    )

    # Run async generation and scoring
    run_eval_start_time = time.time()
    generate_outputs: GenerateOutputs = await vf_eval.a_generate(
        inputs=Dataset.from_list(examples),
        client=client,
        model=model_config.name,
        sampling_args=sampling_args,
        score_rollouts=True,
        max_concurrent=max_concurrent,
    )
    run_eval_time = time.time() - run_eval_start_time
    logger.debug(f"Generated and scored rollouts in {run_eval_time:.2f}s")

    rewards = torch.tensor(generate_outputs.reward).reshape(-1, rollouts_per_example).float()
    responses = [state["responses"] for state in generate_outputs.state]
    completion_lens = torch.tensor(parse_num_completion_tokens(responses)).reshape(-1, rollouts_per_example).float()
    is_truncated = torch.tensor(parse_is_truncated_completions(responses)).reshape(-1, rollouts_per_example).float()

    k = rollouts_per_example
    sample_stats = pd.DataFrame({"example_id": example_ids, "reward": rewards.flatten().tolist()})
    unique_rewards = sample_stats.reward.unique()
    could_be_binary = set(unique_rewards).issubset({0.0, 1.0})
    if could_be_binary:
        pass_at_k = (
            sample_stats.groupby("example_id")
            .apply(lambda x: compute_pass_at_k(x.reward), include_groups=False)
            .apply(pd.Series)
        )
    else:
        pass_at_k = None
        logger.warning("Skipping computing pass@k rates because the task rewards appear to be non-binary")

    # Log statistics
    eval_time = time.time() - eval_start_time
    message = f"Evaluated {eval_id} in {eval_time:.2f}s (Avg@{k}={sample_stats.reward.mean():.4f}"
    if could_be_binary:
        assert pass_at_k is not None
        for pass_rate, pass_rate_score in pd.Series(pass_at_k.mean()).items():
            message += f", {capitalize(str(pass_rate))}: {pass_rate_score:.4f}"
    message += f", Completion Length: {completion_lens.mean():.2f} (±{completion_lens.std():.2f}, ∈[{completion_lens.min():.2f}, {completion_lens.max():.2f}]), Truncated: {is_truncated.mean() * 100:.1f}%)"
    logger.success(message)

    # Log statistics to monitor
    eval_metrics = {
        f"avg@{k}": rewards.mean().item(),
    }

    eval_completion_len_metrics = {
        "avg": completion_lens.mean().item(),
        "max": completion_lens.max().item(),
        "min": completion_lens.min().item(),
    }
    eval_completion_len_metrics = {
        **{f"eval_completion_len/{eval_id}/{k}": v for k, v in eval_completion_len_metrics.items()}
    }
    if step is None:
        step = ckpt_step
    eval_completion_len_metrics.update({"progress/ckpt_step": ckpt_step, "step": step})
    monitor.log(eval_completion_len_metrics)

    if could_be_binary:
        assert pass_at_k is not None
        eval_metrics.update(pd.Series(pass_at_k.mean()).to_dict())
    eval_metrics = {**{f"eval/{eval_id}/{k}": v for k, v in eval_metrics.items()}}
    if step is None:
        step = ckpt_step
    eval_metrics.update({"progress/ckpt_step": ckpt_step, "step": step})

    monitor.log(eval_metrics)

    # Log timing metrics to monitor
    time_metrics = {
        "step": step,
        f"time/eval/{eval_id}": eval_time,
        f"time/eval/{eval_id}/load_environment": load_eval_time,
        f"time/eval/{eval_id}/generate_and_score_rollouts": run_eval_time,
    }
    monitor.log(time_metrics)

    # If specified, save eval artifacts
    if save_to_disk:
        # Save samples as dataset
        eval_dir = get_step_path(get_eval_dir(output_dir), ckpt_step) / eval_id
        dataset = make_dataset(generate_outputs)
        dataset.save_to_disk(eval_dir)
        logger.info(f"Saved eval results for {eval_id} to disk ({eval_dir})")


async def run_evals(
    client: AsyncOpenAI,
    eval_config: EvalConfig | OfflineEvalConfig,
    model_config: ModelConfig,
    sampling_config: EvalSamplingConfig,
    client_config: ClientConfig,
    output_dir: Path,
    ckpt_step: int,
    step: int | None = None,
):
    logger = get_logger()
    await asyncio.gather(
        *[
            run_eval(
                client=client,
                eval_id=eval_id,
                env_args=eval_config.environment_args.get(eval_id, {}),
                num_examples=num_examples,
                rollouts_per_example=rollouts_per_example,
                max_concurrent=max_concurrent,
                output_dir=output_dir,
                save_to_disk=eval_config.save_to_disk,
                model_config=model_config,
                sampling_config=sampling_config,
                client_config=client_config,
                ckpt_step=ckpt_step,
                step=step,
            )
            for eval_id, num_examples, rollouts_per_example, max_concurrent in zip(
                eval_config.environment_ids,
                eval_config.num_examples,
                eval_config.rollouts_per_example,
                eval_config.max_concurrent,
            )
        ]
    )

    if eval_config.save_to_hf is not None:
        logger.info(f"Pushing eval results for {', '.join(eval_config.environment_ids)} to HF Hub")
        eval_dirs = [
            get_step_path(get_eval_dir(output_dir), ckpt_step) / eval_id for eval_id in eval_config.environment_ids
        ]
        dataset_dict = DatasetDict(
            {path.name.replace("-", "_"): cast(Dataset, load_from_disk(path)) for path in eval_dirs}
        )
        dataset_dict.push_to_hub(eval_config.save_to_hf)
        logger.info(f"Pushed eval results to HF Hub (https://huggingface.co/datasets/{eval_config.save_to_hf})")
