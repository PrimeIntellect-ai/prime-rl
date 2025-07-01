import asyncio
import json
import time

import numpy as np
import pandas as pd
from openai import AsyncOpenAI

from zeroband.eval.registry import (
    Benchmark,
    get_benchmark_dataset,
    get_benchmark_display_name,
)
from zeroband.training.orchestrator.client import generate_completion
from zeroband.training.orchestrator.config import ModelConfig, SamplingConfig
from zeroband.training.orchestrator.utils import compute_rewards
from zeroband.utils.logger import get_logger
from zeroband.utils.monitor import get_monitor
from zeroband.utils.utils import capitalize


def compute_pass_rates(rewards: list[int]):
    pass_rates = [k for k in range(1, len(rewards) + 1) if (k & (k - 1)) == 0]
    return {f"pass@{k}": compute_pass_at_k(rewards, k) for k in pass_rates}


def compute_pass_at_k(rewards: list[int], k: int):
    sublists = [rewards[i : i + k] for i in range(0, len(rewards), k)]
    return np.array([any(sublist) for sublist in sublists]).mean()


async def run_benchmark(
    client: AsyncOpenAI,
    benchmark: Benchmark,
    model_config: ModelConfig,
    sampling_config: SamplingConfig,
    step: int,
    use_tqdm: bool = False,
) -> None:
    # Get the logger
    logger = get_logger()
    benchmark_start_time = time.time()

    # Get the monitor
    monitor = get_monitor()

    benchmark_name = get_benchmark_display_name(benchmark)
    logger.info(f"Running {benchmark_name}")

    # Initializing the benchmark dataset
    logger.info(f"Initializing dataset ({benchmark})")
    load_data_start_time = time.time()
    dataset = get_benchmark_dataset(benchmark)

    # Check for required fields
    required_fields = ["verification_info", "task_type", "prompt"]
    if not all(field in dataset.column_names for field in required_fields):
        raise ValueError(
            f"Dataset is missing required fields: It has {dataset.column_names} but needs {required_fields}"
        )

    # Format prompts
    prompts = [item["prompt"] for item in dataset] * sampling_config.n
    problem_ids = list(range(len(dataset))) * sampling_config.n
    batch_messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    load_data_time = time.time() - load_data_start_time
    logger.info(f"Loaded dataset in {load_data_time:.2f}s")

    # Generate completions
    logger.info(f"Generating completions for {len(dataset)} problems")
    generate_start_time = time.time()
    chat_completions = await asyncio.gather(
        *(generate_completion(client, model_config, sampling_config, messages) for messages in batch_messages)
    )
    generate_time = time.time() - generate_start_time

    # Compute rewards
    logger.info("Computing rewards")
    reward_start_time = time.time()
    completions = [chat_completion.choices[0].message.content for chat_completion in chat_completions]
    task_types = [item["task_type"] for item in dataset]
    verification_infos = [json.loads(item["verification_info"]) for item in dataset]
    rewards = compute_rewards(completions, task_types, verification_infos)

    # Collect rewards
    rows = []
    for problem_id, prompt, completion, reward in zip(problem_ids, prompts, completions, rewards):
        logger.debug(f"Problem ID: {problem_id}\n{prompt}\n{completion}")
        row = {"problem_id": problem_id, "reward": reward}
        rows.append(row)

    # Compute scores
    sample_stats = pd.DataFrame(rows)
    unique_rewards = sample_stats.reward.unique()
    could_be_binary = set(unique_rewards).issubset({0.0, 1.0})
    if could_be_binary:
        pass_rates = (
            sample_stats.groupby("problem_id")
            .apply(lambda x: compute_pass_rates(x.reward), include_groups=False)
            .apply(pd.Series)
        )
    else:
        logger.warning("Skipping computing pass@k rates because the task rewards appear to be non-binary")
    reward_time = time.time() - reward_start_time

    # Log statistics
    benchmark_time = time.time() - benchmark_start_time
    logger.success(f"Ran {benchmark_name} in {benchmark_time:.2f}s")
    logger.info(f"Score: {sample_stats.reward.mean():.2f}")
    if could_be_binary:
        for pass_rate, pass_rate_score in pass_rates.mean().items():
            logger.info(f"{capitalize(pass_rate)}: {pass_rate_score:.2f}")

    # Log statistics to monitor
    eval_metrics = {"step": step, "score": sample_stats.reward.mean()}
    if could_be_binary:
        eval_metrics.update(pass_rates.mean().to_dict())
    monitor.log(eval_metrics, wandb_prefix=f"eval/{benchmark}")

    # Log timing metrics to monitor
    time_metrics = {
        "step": step,
        "load_data_time": load_data_time,
        "generate_time": generate_time,
        "reward_time": reward_time,
        "benchmark_time": benchmark_time,
    }
    monitor.log(time_metrics, wandb_prefix=f"eval/{benchmark}/time")
