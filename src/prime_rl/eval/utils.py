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
    Creates a HuggingFace dataset with average accuracy per question alongside original question and answer.
    
    Args:
        examples: List of original examples (not duplicated by rollouts_per_example)
        rewards: Tensor of shape [num_examples, rollouts_per_example] with rewards for each rollout
        rollouts_per_example: Number of rollouts per example
        
    Returns:
        Dataset with columns: prompt, answer, task, info, average_accuracy
    """
    num_examples = len(examples)
    assert rewards.shape == (num_examples, rollouts_per_example), (
        f"Expected rewards shape ({num_examples}, {rollouts_per_example}), got {rewards.shape}"
    )
    
    # Calculate average accuracy for each example
    average_accuracies = rewards.mean(dim=1).tolist()
    
    # Create dataset dictionary
    dataset_dict = {
        "prompt": [example["prompt"] for example in examples],
        "answer": [example.get("answer", "") for example in examples],
        "task": [example.get("task", "") for example in examples],
        "info": [example.get("info", {}) for example in examples],
        "average_accuracy": average_accuracies,
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

    # Convert to list of examples (original examples, not duplicated yet)
    assert dataset is not None
    original_examples = dataset.to_list()
    example_ids = list(range(len(original_examples)))

    # Duplicate examples `rollouts_per_example` times
    if rollouts_per_example > 1:
        duplicated_example_ids = [example_id for example_id in example_ids for _ in range(rollouts_per_example)]
        duplicated_examples = [example for example in original_examples for _ in range(rollouts_per_example)]
    else:
        duplicated_example_ids = example_ids
        duplicated_examples = original_examples

    # Prepare inputs for all examples (duplicated)
    inputs = {
        "prompt": [example["prompt"] for example in duplicated_examples],
        "info": [example.get("info", {}) for example in duplicated_examples],
        "task": [example.get("task", "") for example in duplicated_examples],
        "answer": [example.get("answer", "") for example in duplicated_examples],
    }

    logger.debug(
        f"Evaluating {eval_id} (num_examples={len(original_examples)}, rollouts_per_example={rollouts_per_example}) with args {env_args}"
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

    # Initialize cumulative tracking variables
    all_rewards = []
    all_completion_lens = []
    all_is_truncated = []
    all_states = []
    
    # Cumulative metrics tracking
    cumulative_correct_count = 0
    cumulative_problem_count = 0  
    cumulative_reward_sum = 0.0
    cumulative_rollout_count = 0

    # Run evaluation (either batched or all at once)
    run_eval_start_time = time.time()
    
    if eval_batch_size is not None and eval_batch_size < len(duplicated_examples):
        logger.debug(f"Running batched evaluation with batch size {eval_batch_size}")
        
        # Process examples in batches
        for batch_start in range(0, len(duplicated_examples), eval_batch_size):
            batch_end = min(batch_start + eval_batch_size, len(duplicated_examples))
            batch_num = batch_start // eval_batch_size + 1
            total_batches = (len(duplicated_examples) + eval_batch_size - 1) // eval_batch_size

            # Create batch inputs
            batch_inputs = {
                "prompt": inputs["prompt"][batch_start:batch_end],
                "info": inputs["info"][batch_start:batch_end],
                "task": inputs["task"][batch_start:batch_end],
                "answer": inputs["answer"][batch_start:batch_end],
            }

            batch_problems = (batch_end - batch_start) // rollouts_per_example
            logger.debug(
                f"Processing batch {batch_num}/{total_batches}: examples {batch_start}-{batch_end - 1} ({batch_problems} problems)"
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
            
            # Update cumulative tracking
            cumulative_correct_count += batch_fully_correct
            cumulative_problem_count += batch_num_problems
            cumulative_reward_sum += batch_rewards.sum().item()
            cumulative_rollout_count += batch_rewards.numel()
            
            # Calculate cumulative statistics
            cumulative_avg_accuracy = cumulative_reward_sum / cumulative_rollout_count
            cumulative_fully_correct_pct = (cumulative_correct_count / cumulative_problem_count) * 100
            
            # Log batch statistics
            logger.info(
                f"Batch {batch_num}/{total_batches}: Avg Accuracy: {batch_avg_accuracy:.4f}, "
                f"Perfect Samples: {batch_fully_correct}/{batch_num_problems} ({(batch_fully_correct/batch_num_problems)*100:.1f}%)"
            )
            logger.info(
                f"Cumulative: Avg Accuracy: {cumulative_avg_accuracy:.4f}, "
                f"Perfect Samples: {cumulative_correct_count}/{cumulative_problem_count} ({cumulative_fully_correct_pct:.1f}%)"
            )

            # Store batch results
            all_rewards.extend(batch_outputs.reward)
            all_completion_lens.extend(list(map(len, parse_completion_tokens(states=batch_outputs.state))))
            all_is_truncated.extend(parse_truncated_completions(states=batch_outputs.state))
            all_states.extend(batch_outputs.state)

        # Create final generate_outputs object
        generate_outputs = GenerateOutputs(
            reward=all_rewards,
            state=all_states,
        )
        
    else:
        # Run all examples at once (original behavior)
        logger.debug("Running evaluation on all examples at once")
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
    completion_lens = torch.tensor(list(map(len, parse_completion_tokens(states=generate_outputs.state)))).reshape(-1, rollouts_per_example).float()
    is_truncated = torch.tensor(parse_truncated_completions(states=generate_outputs.state)).reshape(-1, rollouts_per_example).float()

    k = rollouts_per_example
    sample_stats = pd.DataFrame({"example_id": duplicated_example_ids, "reward": rewards.flatten().tolist()})
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
    message += (
        f", Completion Length: {completion_lens.mean():.2f} (±{completion_lens.std():.2f}, ∈[{completion_lens.min():.2f}, {completion_lens.max():.2f}]), Truncated: {is_truncated.mean() * 100:.1f}%)"
    )
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

        # Save accuracy dataset if requested
        if save_accuracy_dataset:
            accuracy_dataset = create_accuracy_dataset(original_examples, rewards, rollouts_per_example)
            accuracy_dir = eval_dir / "accuracy_dataset"
            accuracy_dataset.push_to_hub("Fareso/math_filtered")
            accuracy_dataset.save_to_disk(accuracy_dir)
            logger.info(f"Saved accuracy dataset to {accuracy_dir}")

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

        logger.info(f"Saved samples and report to {eval_dir}")
