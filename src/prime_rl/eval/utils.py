import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
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
    push_to_hub: bool = False,
    hub_name: str | None = None,
    eval_batch_size: int | None = None,
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

    # Helper function to generate outputs
    async def generate_outputs_for_inputs(input_data):
        return await vf_eval.a_generate(
            inputs=input_data,
            client=client,
            model=model_config.name,
            sampling_args=sampling_args,
            score_rollouts=True,
        )
    
    # Run async generation and scoring
    run_eval_start_time = time.time()
    
    if eval_batch_size is not None and eval_batch_size < len(examples):
        logger.debug(f"Running batched evaluation with batch size {eval_batch_size}")
        
        # Process examples in batches
        all_outputs = []
        total_batches = (len(examples) + eval_batch_size - 1) // eval_batch_size
        
        for batch_idx in range(total_batches):
            batch_start = batch_idx * eval_batch_size
            batch_end = min(batch_start + eval_batch_size, len(examples))
            
            # Create batch inputs
            batch_inputs = {k: v[batch_start:batch_end] for k, v in inputs.items()}
            
            logger.debug(f"Processing batch {batch_idx + 1}/{total_batches}")
            batch_outputs = await generate_outputs_for_inputs(batch_inputs)
            all_outputs.append(batch_outputs)
        
        # Combine all batch outputs
        generate_outputs = GenerateOutputs(**{
            field: [item for batch in all_outputs for item in getattr(batch, field)]
            for field in ['prompt', 'completion', 'answer', 'state', 'info', 'task', 'reward']
        })
    else:
        # Run all examples at once
        logger.debug("Running evaluation on all examples at once")
        generate_outputs = await generate_outputs_for_inputs(inputs)
    run_eval_time = time.time() - run_eval_start_time
    logger.debug(f"Generated and scored rollouts in {run_eval_time:.2f}s")

    # Process results into tensors
    rewards = torch.tensor(generate_outputs.reward, dtype=torch.float32).reshape(-1, rollouts_per_example)
    completion_lens = torch.tensor([len(tokens) for tokens in parse_completion_tokens(states=generate_outputs.state)], dtype=torch.float32).reshape(-1, rollouts_per_example)
    is_truncated = torch.tensor(parse_truncated_completions(states=generate_outputs.state), dtype=torch.float32).reshape(-1, rollouts_per_example)

    num_rollouts = rollouts_per_example
    reward_list = rewards.flatten().tolist()
    sample_stats = pd.DataFrame({"example_id": example_ids, "reward": reward_list})
    unique_rewards = set(reward_list)
    could_be_binary = unique_rewards.issubset({0.0, 1.0})
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
    message = f"Evaluated {eval_id} in {eval_time:.2f}s (Avg@{num_rollouts}={sample_stats.reward.mean():.4f}"
    if could_be_binary:
        assert pass_at_k is not None
        for pass_rate, pass_rate_score in pd.Series(pass_at_k.mean()).items():
            message += f", {capitalize(str(pass_rate))}: {pass_rate_score:.4f}"
    message += (
        f", Completion Length: {completion_lens.mean():.2f} (±{completion_lens.std():.2f}, ∈[{completion_lens.min():.2f}, {completion_lens.max():.2f}]), Truncated: {is_truncated.mean() * 100:.1f}%)"
    )
    logger.success(message)

    # Log statistics to monitor
    step = step if step is not None else ckpt_step
    common_metadata = {"progress/ckpt_step": ckpt_step, "step": step}
    
    # Completion length metrics
    completion_len_stats = {
        "avg": completion_lens.mean().item(),
        "max": completion_lens.max().item(),
        "min": completion_lens.min().item(),
    }
    eval_completion_len_metrics = {
        **{f"eval_completion_len/{eval_id}/{k}": v for k, v in completion_len_stats.items()},
        **common_metadata
    }
    monitor.log(eval_completion_len_metrics)

    # Evaluation metrics
    eval_metrics = {f"avg@{num_rollouts}": rewards.mean().item()}
    if could_be_binary and pass_at_k is not None:
        eval_metrics.update(pd.Series(pass_at_k.mean()).to_dict())
    
    eval_metrics = {
        **{f"eval/{eval_id}/{k}": v for k, v in eval_metrics.items()},
        **common_metadata
    }
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
        dataset = vf_eval.make_dataset(generate_outputs, push_to_hub=push_to_hub, hub_name=hub_name)
        dataset.save_to_disk(eval_dir)

        # Save "report"
        # TODO: Make this into an actually nice report, for now just JSON-dump eval metrics
        report_path = eval_dir / "report.json"
        report = {
            "metrics": eval_metrics,
            "truncated": {"mean": is_truncated.mean().item()},
            "completion_len": completion_len_stats,
        }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Saved samples and report to {eval_dir}")
