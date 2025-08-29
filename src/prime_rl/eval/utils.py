import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from openai import AsyncOpenAI
from verifiers import load_environment
from verifiers.types import GenerateOutputs

from prime_rl.orchestrator.config import EvalSamplingConfig, ModelConfig
from prime_rl.orchestrator.utils import parse_completion_tokens, parse_truncated_completions
from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor import get_monitor
from prime_rl.utils.utils import capitalize, get_eval_dir


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

    avg_pass_rate = np.mean(pass_rates)

    return {f"pass@{k}": avg_pass_rate}


def create_accuracy_dataset(examples: list[dict], rewards: torch.Tensor, rollouts_per_example: int) -> Dataset:
    """
    Create a HuggingFace dataset with average accuracy per question.

    Args:
        examples: List of original examples with prompts, answers, etc.
        rewards: Tensor of shape (num_examples, rollouts_per_example) with rewards
        rollouts_per_example: Number of rollouts per example

    Returns:
        HuggingFace Dataset with original data plus average accuracy and fully correct indicator
    """
    # Reshape rewards to (num_examples, rollouts_per_example)
    num_examples = len(examples) // rollouts_per_example
    reshaped_rewards = rewards.reshape(num_examples, rollouts_per_example)

    # Compute average accuracy per example
    avg_accuracy = reshaped_rewards.mean(dim=1).tolist()

    # Compute fully correct indicator (all rollouts for this example were correct)
    fully_correct = (reshaped_rewards.sum(dim=1) == rollouts_per_example).tolist()

    # Extract unique examples (since examples are duplicated for rollouts)
    unique_examples = examples[::rollouts_per_example]

    # Create dataset
    dataset_dict = {
        "prompt": [ex["prompt"] for ex in unique_examples],
        "answer": [ex.get("answer", "") for ex in unique_examples],
        "task": [ex.get("task", "") for ex in unique_examples],
        "info": [ex.get("info", {}) for ex in unique_examples],
        "average_accuracy": avg_accuracy,
        "fully_correct": fully_correct,
        "rollouts_per_example": [rollouts_per_example] * num_examples,
    }

    return Dataset.from_dict(dataset_dict)


async def run_eval(
    client: AsyncOpenAI,
    eval_id: str,
    env_args: dict,
    model_config: ModelConfig,
    sampling_config: EvalSamplingConfig,
    num_examples: int,
    rollouts_per_example: int,
    save: bool,
    output_dir: Path,
    ckpt_step: int,
    step: int | None = None,
    eval_batch_size: int | None = None,
    save_accuracy_dataset: bool = True,
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

    # Prepare inputs
    inputs = {
        "prompt": [example["prompt"] for example in examples],
        "info": [example.get("info", {}) for example in examples],
        "task": [example.get("task", "") for example in examples],
        "answer": [example.get("answer", "") for example in examples],
    }

    logger.debug(
        f"Evaluating {eval_id} (num_examples={len(examples)}, rollouts_per_example={rollouts_per_example}) with args {env_args}"
    )

    # Always return logprobs to parser response length
    sampling_args: dict[str, Any] = {
        "logprobs": True,
        "extra_body": {
            "return_tokens_as_token_ids": True,
        },
    }

    # Apply sampling config only if specified
    if sampling_config.temperature is not None:
        sampling_args["temperature"] = sampling_config.temperature
    if sampling_config.max_tokens is not None:
        sampling_args["max_tokens"] = sampling_config.max_tokens
    if sampling_config.top_p is not None:
        sampling_args["top_p"] = sampling_config.top_p
    if sampling_config.top_k is not None:
        sampling_args["extra_body"]["top_k"] = sampling_config.top_k
    if sampling_config.min_p is not None:
        sampling_args["extra_body"]["min_p"] = sampling_config.min_p
    if sampling_config.min_tokens is not None:
        sampling_args["extra_body"]["min_tokens"] = sampling_config.min_tokens

    # Run async generation and scoring
    run_eval_start_time = time.time()

    # Process examples in batches if batch_size is specified
    if eval_batch_size is not None and len(examples) > eval_batch_size:
        logger.debug(f"Processing {len(examples)} examples in batches of {eval_batch_size}")

        # Initialize lists to collect batch results
        all_rewards = []
        all_states = []

        # Process examples in batches
        for batch_start in range(0, len(examples), eval_batch_size):
            batch_end = min(batch_start + eval_batch_size, len(examples))

            # Create batch inputs
            batch_inputs = {
                "prompt": inputs["prompt"][batch_start:batch_end],
                "info": inputs["info"][batch_start:batch_end],
                "task": inputs["task"][batch_start:batch_end],
                "answer": inputs["answer"][batch_start:batch_end],
            }

            logger.debug(
                f"Processing batch {batch_start // eval_batch_size + 1}/{(len(examples) + eval_batch_size - 1) // eval_batch_size}: examples {batch_start}-{batch_end - 1}"
            )

            # Generate and score batch
            batch_outputs: GenerateOutputs = await vf_eval.a_generate(
                inputs=batch_inputs,
                client=client,
                model=model_config.name,
                sampling_args=sampling_args,
                score_rollouts=True,
            )

            # Log batch statistics
            batch_rewards = torch.tensor(batch_outputs.reward).reshape(-1, rollouts_per_example).float()
            batch_avg_accuracy = batch_rewards.mean().item()
            batch_fully_correct = (batch_rewards.sum(dim=1) == rollouts_per_example).sum().item()
            batch_num_problems = batch_rewards.shape[0]

            logger.info(
                f"Batch {batch_start // eval_batch_size + 1}: "
                f"Avg accuracy: {batch_avg_accuracy:.3f}, "
                f"Fully correct problems: {batch_fully_correct}/{batch_num_problems} "
                f"({100 * batch_fully_correct / batch_num_problems:.1f}%)"
            )

            # Collect batch results
            all_rewards.extend(batch_outputs.reward)
            all_states.extend(batch_outputs.state)

        # Reconstruct complete generate_outputs
        generate_outputs = GenerateOutputs(reward=all_rewards, state=all_states)
    else:
        # Process all examples in a single batch (original behavior)
        generate_outputs: GenerateOutputs = await vf_eval.a_generate(
            inputs=inputs,
            client=client,
            model=model_config.name,
            sampling_args=sampling_args,
            score_rollouts=True,
        )

    run_eval_time = time.time() - run_eval_start_time
    logger.debug(f"Generated and scored rollouts in {run_eval_time:.2f}s")

    rewards = torch.tensor(generate_outputs.reward).reshape(-1, rollouts_per_example).float()
    completion_lens = (
        torch.tensor(list(map(len, parse_completion_tokens(states=generate_outputs.state))))
        .reshape(-1, rollouts_per_example)
        .float()
    )
    is_truncated = (
        torch.tensor(parse_truncated_completions(states=generate_outputs.state))
        .reshape(-1, rollouts_per_example)
        .float()
    )

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
    if save:
        # Save samples as dataset
        eval_dir = get_eval_dir(output_dir) / f"step_{ckpt_step}" / eval_id
        dataset = vf_eval.make_dataset(generate_outputs)
        dataset.save_to_disk(eval_dir)

        # Save "report"
        # TODO: Make this into an actually nice report, for now just JSON-dump eval metrics
        report_path = eval_dir / "report.json"
        report = {
            "metrics": eval_metrics,
            "truncated": {
                "mean": is_truncated.mean().item(),
            },
            "completion_len": {
                "avg": completion_lens.mean().item(),
                "max": completion_lens.max().item(),
                "min": completion_lens.min().item(),
            },
        }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # Save accuracy dataset if requested
        if save_accuracy_dataset:
            accuracy_dataset = create_accuracy_dataset(examples, rewards, rollouts_per_example)
            accuracy_dataset_path = eval_dir / "accuracy_dataset"
            accuracy_dataset.save_to_disk(accuracy_dataset_path)
            logger.info(f"Saved accuracy dataset to {accuracy_dataset_path}")

        logger.info(f"Saved samples and report to {eval_dir}")
