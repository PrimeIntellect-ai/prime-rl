# Import environment before any other imports
# ruff: noqa
from zeroband.inference import envs
from typing import cast

import json
import time
import pandas as pd
from datasets import Dataset, load_dataset
from vllm import SamplingParams, TokensPrompt
from huggingface_hub import snapshot_download

from zeroband.utils.monitor import setup_monitor
from zeroband.utils.pydantic_config import parse_argv
from zeroband.inference.eval.config import Config as EvalConfig
from zeroband.inference.utils import setup_model, format_prompts, filter_data_by_prompt_length
from zeroband.inference.rewards import compute_vllm_rewards
from zeroband.inference.eval.logger import setup_logger
from zeroband.utils.utils import clean_exit


@clean_exit
def main(config: EvalConfig):
    # Initialize the logger
    logger = setup_logger(config.log)
    logger.info("Starting evaluation")

    # Initialize the monitor
    monitor = setup_monitor(config.monitor, None, config)

    # Pre-download the model weights
    logger.info(f"Downloading model weights for {config.model.name}")
    start_time = time.time()
    snapshot_download(config.model.name)
    logger.success(f"Downloaded model weights in {time.time() - start_time:.2f}s")

    # Initializing the model and tokenizer
    logger.info(f"Initializing model and tokenizer ({config.model} tensor_parallel_size={config.parallel.tp} seed={config.seed})")
    start_time = time.time()
    llm, tokenizer = setup_model(config.model, tp=config.parallel.tp, seed=config.seed)
    logger.success(f"Initialized model and tokenizer in {time.time() - start_time:.2f}s")

    # Initializing the benchmark dataset
    logger.info(f"Initializing dataset (name={config.data.name}, split={config.data.split})")
    start_time = time.time()
    dataset = cast(Dataset, load_dataset(config.data.name, split=config.data.split))
    logger.success(f"Initialized dataset with {len(dataset):,} problems in {time.time() - start_time:.2f}s")

    # Check for required fields
    required_fields = ["verification_info", "task_type", "prompt"]
    if not all(field in dataset.column_names for field in required_fields):
        raise ValueError(f"Dataset is missing required fields: It has {dataset.column_names} but needs {required_fields}")

    # Initialize sampling parameters
    logger.info(f"Initializing sampling parameters ({config.sampling} seed={config.seed})")
    sampling_params = SamplingParams(n=1, max_tokens=None)  # SamplingParams(**config.sampling.model_dump(), seed=config.seed)

    # Format prompts
    tokenized_prompts = format_prompts(
        [item["prompt"] for item in dataset],
        [-1] * len(dataset),
        len_rewards_config=None,
        tokenizer=tokenizer,
        enable_thinking=config.model.enable_thinking,
    )
    prompts = [TokensPrompt(prompt_token_ids=cast(list[int], input_ids)) for input_ids in tokenized_prompts]

    # Generate completions
    logger.info(f"Generating completions for {len(dataset)} problems")
    start_time = time.time()
    request_outputs = llm.generate(prompts, sampling_params)
    logger.success(f"Generated completions in {time.time() - start_time:.2f}s")

    # Compute rewards
    verification_infos = [json.loads(item["verification_info"]) for item in dataset]
    task_types = [item["task_type"] for item in dataset]
    request_rewards = compute_vllm_rewards(request_outputs, verification_infos, task_types, None)

    # Collect rewards
    rows = []
    for request_output, request_reward in zip(request_outputs, request_rewards):
        req_id = request_output.request_id
        for output, reward in zip(request_output.outputs, request_reward.rewards):
            logger.debug(f"Request ID: {req_id}\n{tokenizer.decode(request_output.prompt_token_ids)}{output.text}")
            rows.append(
                {
                    "request_id": req_id,
                    "reward": reward.reward,
                }
            )
    sample_stats = pd.DataFrame(rows)

    # Compute overall statistics
    mean_sample_reward = sample_stats["reward"].mean()
    std_sample_reward = sample_stats["reward"].std()

    # Compute per-problem statistics
    problem_stats = sample_stats.groupby("request_id").agg({"reward": "mean"})
    mean_problem_reward = problem_stats["reward"].mean()
    std_problem_reward = problem_stats["reward"].std()

    # Log statistics
    logger.info(f"Mean problem reward: {mean_problem_reward:.2f} ± {std_problem_reward:.2f}")
    logger.info(f"Mean sample reward: {mean_sample_reward:.2f} ± {std_sample_reward:.2f}")

    # Log statistics to monitor
    monitor.log({"rewards/mean_sample_reward": mean_sample_reward, "rewards/std_sample_reward": std_sample_reward})
    monitor.log({"rewards/mean_problem_reward": mean_problem_reward, "rewards/std_problem_reward": std_problem_reward})

    logger.info("Evaluation finished!")


if __name__ == "__main__":
    main(parse_argv(EvalConfig))
