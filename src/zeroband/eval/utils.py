import json
import time
from typing import cast

import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams, TokensPrompt

from zeroband.eval.registry import Benchmark, get_benchmark_dataset, get_benchmark_display_name
from zeroband.inference.config import ModelConfig, SamplingConfig
from zeroband.inference.rewards import compute_vllm_rewards
from zeroband.inference.utils import format_prompts
from zeroband.utils.logger import get_logger
from zeroband.utils.monitor import get_monitor
from zeroband.utils.utils import capitalize


def compute_pass_rates(rewards: list[int]):
    pass_rates = [k for k in range(1, len(rewards) + 1) if (k & (k - 1)) == 0]
    return {f"pass@{k}": compute_pass_at_k(rewards, k) for k in pass_rates}


def compute_pass_at_k(rewards: list[int], k: int):
    sublists = [rewards[i : i + k] for i in range(0, len(rewards), k)]
    return np.array([any(sublist) for sublist in sublists]).mean()


def run_benchmark(
    llm: LLM,
    benchmark: Benchmark,
    model_config: ModelConfig,
    sampling_config: SamplingConfig,
    step: int,
    seed: int | None = None,
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
        raise ValueError(f"Dataset is missing required fields: It has {dataset.column_names} but needs {required_fields}")

    # Format prompts
    tokenized_prompts = format_prompts(
        [item["prompt"] for item in dataset],
        [-1] * len(dataset),
        len_rewards_config=None,
        tokenizer=llm.get_tokenizer(),
        enable_thinking=model_config.enable_thinking,
        tokenize=True,
    )
    prompts = [TokensPrompt(prompt_token_ids=cast(list[int], input_ids)) for input_ids in tokenized_prompts]
    load_data_time = time.time() - load_data_start_time

    # Initialize sampling parameters
    logger.info(f"Initializing sampling parameters ({sampling_config} seed={seed})")
    sampling_params = SamplingParams(**sampling_config.model_dump(), seed=seed)

    # Generate completions
    logger.info(f"Generating completions for {len(dataset)} problems")
    generate_start_time = time.time()
    request_outputs = llm.generate(prompts, sampling_params, use_tqdm=use_tqdm)
    generate_time = time.time() - generate_start_time

    # Compute rewards
    logger.info("Computing rewards")
    reward_start_time = time.time()
    verification_infos = [json.loads(item["verification_info"]) for item in dataset]
    task_types = [item["task_type"] for item in dataset]
    request_rewards = compute_vllm_rewards(request_outputs, verification_infos, task_types, None)

    # Collect rewards
    rows = []
    for request_output, request_reward in zip(request_outputs, request_rewards):
        req_id = request_output.request_id
        for output, reward in zip(request_output.outputs, request_reward.rewards):
            logger.debug(f"Request ID: {req_id}\n{llm.get_tokenizer().decode(request_output.prompt_token_ids)}{output.text}")
            row = {"request_id": req_id, "reward": reward.reward}
            rows.append(row)

    # Compute scores
    sample_stats = pd.DataFrame(rows)
    unique_rewards = sample_stats.reward.unique()
    could_be_binary = set(unique_rewards).issubset({0.0, 1.0})
    if could_be_binary:
        pass_rates = sample_stats.groupby("request_id").apply(lambda x: compute_pass_rates(x.reward), include_groups=False).apply(pd.Series)
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
