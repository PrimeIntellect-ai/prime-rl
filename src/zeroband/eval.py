# Import environment before any other imports
# ruff: noqa
from pathlib import Path
from zeroband.inference import envs
from typing import cast

import json
import time
import pandas as pd
from vllm import LLM, SamplingParams, TokensPrompt
from huggingface_hub import snapshot_download

from zeroband.utils.monitor import setup_monitor
from zeroband.utils.pydantic_config import parse_argv
from zeroband.inference.eval.config import Config as EvalConfig
from zeroband.inference.eval.registry import get_benchmark_dataset, get_benchmark_display_name
from zeroband.inference.utils import setup_model, format_prompts, reload_model_weights
from zeroband.inference.rewards import compute_vllm_rewards
from zeroband.inference.eval.logger import setup_logger
from zeroband.inference.eval.registry import Benchmark
from zeroband.utils.utils import clean_exit
from zeroband.utils.logger import get_logger
from zeroband.utils.monitor import get_monitor


def run_benchmark(llm: LLM, benchmark: Benchmark, config: EvalConfig) -> None:
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
        enable_thinking=config.model.enable_thinking,
        tokenize=True,
    )
    prompts = [TokensPrompt(prompt_token_ids=cast(list[int], input_ids)) for input_ids in tokenized_prompts]
    load_data_time = time.time() - load_data_start_time

    # Initialize sampling parameters
    logger.info(f"Initializing sampling parameters ({config.sampling} seed={config.seed})")
    sampling_params = SamplingParams(**config.sampling.model_dump(), seed=config.seed)

    # Generate completions
    logger.info(f"Generating completions for {len(dataset)} problems")
    generate_start_time = time.time()
    request_outputs = llm.generate(prompts, sampling_params, use_tqdm=config.use_tqdm)
    generate_time = time.time() - generate_start_time

    # Compute rewards
    logger.info(f"Computing rewards")
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
            rows.append(
                {
                    "request_id": req_id,
                    "reward": reward.reward,
                }
            )
    sample_stats = pd.DataFrame(rows)
    problem_stats = sample_stats.groupby("request_id").agg({"reward": "mean"})

    # Compute scores
    mean_sample_score = sample_stats["reward"].mean()
    mean_problem_score = problem_stats["reward"].mean()
    reward_time = time.time() - reward_start_time

    # Log statistics
    benchmark_time = time.time() - benchmark_start_time
    logger.success(f"Ran {benchmark_name} in {benchmark_time:.2f}s")
    logger.info(f"Mean problem score: {mean_problem_score:.2f}")
    logger.info(f"Mean sample score: {mean_sample_score:.2f}")

    # Log statistics to monitor
    eval_metrics = {
        "mean_sample_score": mean_sample_score,
        "mean_problem_score": mean_problem_score,
    }
    monitor.log(eval_metrics, prefix=f"eval/{benchmark}")

    # Log timing metrics to monitor
    monitor.log(
        {
            "load_data_time": load_data_time,
            "generate_time": generate_time,
            "reward_time": reward_time,
            "benchmark_time": benchmark_time,
        },
        prefix=f"eval/{benchmark}/time",
    )


@clean_exit
def main(config: EvalConfig):
    # Initialize the logger
    logger = setup_logger(config.log)
    logger.info("Starting evaluation")
    logger.info(f"Evaluation config: {config.eval}")

    # Initialize the monitor
    setup_monitor(config.monitor, None, config)

    # Pre-download the model weights
    logger.info(f"Downloading model weights for {config.model.name}")
    start_time = time.time()
    snapshot_download(config.model.name)
    logger.success(f"Downloaded model weights in {time.time() - start_time:.2f}s")

    # Initializing the model and tokenizer
    logger.info(f"Initializing model and tokenizer ({config.model} tensor_parallel_size={config.parallel.tp} seed={config.seed})")
    start_time = time.time()
    llm, _ = setup_model(config.model, tp=config.parallel.tp, seed=config.seed)
    logger.success(f"Initialized model and tokenizer in {time.time() - start_time:.2f}s")

    # Run benchmarks on base model
    logger.info(f"Running benchmarks on base model {config.model.name}")
    for benchmark in config.eval.benchmarks:
        run_benchmark(llm, benchmark, config)

    # If specified, run online evaluation
    if config.eval.online:
        logger.info(
            f"Running online evaluation on {config.model.name} every {config.eval.online.interval} steps from checkpoint directory {config.eval.online.ckpt_path}"
        )
        while True:
            step, num_attempts = config.eval.online.interval, 0
            while True:
                ckpt_path = Path(config.eval.online.ckpt_path) / f"step_{step}"
                stable_file = ckpt_path / "stable"
                if stable_file.exists():
                    logger.info(f"Found checkpoint for step {step} at {stable_file}. Reloading model weights.")
                    llm = reload_model_weights(llm, ckpt_path / "model.safetensors")
                    break
                if num_attempts % 30 == 0:  # Every 30s
                    logger.info(f"Waiting for checkpoint for step {step} at {stable_file}")
                time.sleep(1)
                num_attempts += 1

            # Run benchmarks on updated model
            logger.info(f"Running benchmarks for checkpoint step {step}")
            for benchmark in config.eval.benchmarks:
                run_benchmark(llm, benchmark, config)

            # Wait for the next step
            step += config.eval.online.interval

            if config.eval.online.max_steps and step > config.eval.online.max_steps:
                logger.info(f"Reached maximum number of steps ({config.eval.online.max_steps}). Stopping online evaluation.")
                break

    logger.info("Evaluation finished!")


if __name__ == "__main__":
    main(parse_argv(EvalConfig))
