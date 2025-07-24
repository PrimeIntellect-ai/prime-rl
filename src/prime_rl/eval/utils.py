import asyncio
import json
import time

import pandas as pd
from openai import AsyncOpenAI

from prime_rl.eval.registry import (
    Benchmark,
    get_benchmark_dataset,
    get_benchmark_display_name,
)
from prime_rl.orchestrator.client import generate_completion
from prime_rl.orchestrator.config import ModelConfig, SamplingConfig
from prime_rl.orchestrator.utils import compute_rewards, parse_completions
from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor import MultiMonitor
from prime_rl.utils.utils import capitalize


def compute_pass_at_k(rewards: list[int]):
    k = len(rewards)
    return {f"pass@{k}": any(reward == 1.0 for reward in rewards)}


async def run_benchmark(
    client: AsyncOpenAI,
    benchmark: Benchmark,
    model_config: ModelConfig,
    sampling_config: SamplingConfig,
    rollouts_per_prompt: int,
    ckpt_step: int,
    monitor: MultiMonitor,
    step: int | None = None,
) -> None:
    # Get the logger
    logger = get_logger()
    benchmark_start_time = time.time()

    benchmark_name = get_benchmark_display_name(benchmark)
    logger.info(
        f"Evaluating {model_config.name} on {benchmark_name} {f'for checkpoint {ckpt_step}' if ckpt_step > 0 else ''}"
    )

    # Initializing the benchmark dataset
    logger.debug(f"Loading benchmark dataset ({benchmark})")
    load_data_start_time = time.time()
    dataset = get_benchmark_dataset(benchmark)

    # Check for required fields
    required_fields = ["verification_info", "task_type", "prompt"]
    if not all(field in dataset.column_names for field in required_fields):
        raise ValueError(
            f"Dataset is missing required fields: It has {dataset.column_names} but needs {required_fields}"
        )

    # Format prompts
    prompts = [item["prompt"] for item in dataset]
    problem_ids = list(range(len(dataset)))
    prompts = [prompt for prompt in prompts for _ in range(rollouts_per_prompt)]
    problem_ids = [problem_id for problem_id in problem_ids for _ in range(rollouts_per_prompt)]
    batch_messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    load_data_time = time.time() - load_data_start_time
    logger.debug(f"Loaded dataset in {load_data_time:.2f}s")

    # Generate completions
    logger.debug(
        f"Generating {rollouts_per_prompt} rollouts per prompt for {len(dataset)} problems ({len(prompts)} total requests)"
    )
    generate_completions_start_time = time.time()
    chat_completions = await asyncio.gather(
        *(generate_completion(client, model_config, sampling_config, messages) for messages in batch_messages)
    )
    generate_completions_time = time.time() - generate_completions_start_time
    logger.debug(f"Generated completions in {generate_completions_time:.2f}s")

    # Compute rewards
    logger.debug("Computing rewards")
    compute_rewards_start_time = time.time()
    completions = parse_completions(chat_completions)
    task_types = [item["task_type"] for item in dataset]
    verification_infos = [json.loads(item["verification_info"]) for item in dataset]
    task_types = [task_type for task_type in task_types for _ in range(rollouts_per_prompt)]
    verification_infos = [
        verification_info for verification_info in verification_infos for _ in range(rollouts_per_prompt)
    ]
    rewards = compute_rewards(completions, task_types, verification_infos)

    # Collect rewards
    rows = []
    for problem_id, reward in zip(problem_ids, rewards):
        row = {"problem_id": problem_id, "reward": reward}
        rows.append(row)

    # Compute scores
    k = rollouts_per_prompt
    sample_stats = pd.DataFrame(rows)
    unique_rewards = sample_stats.reward.unique()
    could_be_binary = set(unique_rewards).issubset({0.0, 1.0})
    if could_be_binary:
        pass_at_k = (
            sample_stats.groupby("problem_id")
            .apply(lambda x: compute_pass_at_k(x.reward), include_groups=False)
            .apply(pd.Series)
        )
    else:
        logger.warning("Skipping computing pass@k rates because the task rewards appear to be non-binary")
    compute_rewards_time = time.time() - compute_rewards_start_time
    logger.debug(f"Computed rewards in {compute_rewards_time:.2f}s")

    # Log statistics
    benchmark_time = time.time() - benchmark_start_time
    message = f"Evaluated {benchmark_name} in {benchmark_time:.2f}s (Avg@{k}={sample_stats.reward.mean():.2f}"
    if could_be_binary:
        for pass_rate, pass_rate_score in pass_at_k.mean().items():
            message += f", {capitalize(pass_rate)}: {pass_rate_score:.2f}"
    logger.success(message + ")")

    # Log statistics to monitor
    eval_metrics = {f"avg@{k}": float(sample_stats.reward.mean())}
    if could_be_binary:
        eval_metrics.update(pass_at_k.mean().to_dict())
    eval_metrics = {**{f"eval/{benchmark}/{k}": v for k, v in eval_metrics.items()}}
    if step is None:
        step = ckpt_step
    eval_metrics.update({"progress/ckpt_step": ckpt_step, "step": step})

    monitor.log(eval_metrics)

    # Log timing metrics to monitor
    time_metrics = {
        "step": step,
        f"time/eval/{benchmark}": benchmark_time,
        f"time/eval/{benchmark}/load_data": load_data_time,
        f"time/eval/{benchmark}/generate_completions": generate_completions_time,
        f"time/eval/{benchmark}/compute_rewards": compute_rewards_time,
    }
    monitor.log(time_metrics)
