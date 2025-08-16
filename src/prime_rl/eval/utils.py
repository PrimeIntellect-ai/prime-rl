import time
from typing import Any

import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from verifiers import load_environment

from prime_rl.orchestrator.config import EvalSamplingConfig, ModelConfig
from prime_rl.orchestrator.utils import parse_completion_tokens
from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor import MultiMonitor
from prime_rl.utils.utils import capitalize


def compute_pass_at_k(rewards: list[int]) -> dict[str, float]:
    k = len(rewards)
    return {f"pass@{k}": float(any(reward == 1.0 for reward in rewards))}


async def run_eval(
    client: AsyncOpenAI,
    eval_id: str,
    env_args: dict,
    model_config: ModelConfig,
    sampling_config: EvalSamplingConfig,
    num_examples: int,
    rollouts_per_example: int,
    ckpt_step: int,
    monitor: MultiMonitor,
    step: int | None = None,
) -> None:
    # Get the logger
    logger = get_logger()
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
        inputs = vf_eval.get_dataset(n=num_examples)
    else:
        inputs = vf_eval.get_eval_dataset(n=num_examples)

    assert inputs is not None
    num_unique_examples = len(inputs)
    if rollouts_per_example > 1:
        inputs = inputs.repeat(rollouts_per_example)

    logger.debug(
        f"Evaluating {eval_id} (num_examples={num_unique_examples}, rollouts_per_example={rollouts_per_example}) with args {env_args}"
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
    results = await vf_eval.a_generate(
        inputs=inputs,
        client=client,
        model=model_config.name,
        sampling_args=sampling_args,
        score_rollouts=True,
    )
    run_eval_time = time.time() - run_eval_start_time
    logger.debug(f"Generated and scored rollouts in {run_eval_time:.2f}s")

    problem_ids = list(range(num_unique_examples)) * rollouts_per_example

    completion_lengths = [sum([len(parse_completion_tokens(r)) for r in state["responses"]]) for state in results.state]
    avg_completion_len = sum(completion_lengths) / len(completion_lengths)
    completion_len_df = pd.DataFrame({"problem_id": problem_ids, "completion_len": completion_lengths})
    avg_completion_len_per_problem = completion_len_df.groupby("problem_id").completion_len.mean()
    max_avg_completion_len = avg_completion_len_per_problem.max()
    min_avg_completion_len = avg_completion_len_per_problem.min()

    # Average reward
    rewards = np.array(results.reward, dtype=float)
    rows = []
    for problem_id, reward in zip(problem_ids, rewards):
        row = {"problem_id": problem_id, "reward": reward}
        rows.append(row)

    k = rollouts_per_example
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
        pass_at_k = None
        logger.warning("Skipping computing pass@k rates because the task rewards appear to be non-binary")

    # Log statistics
    eval_time = time.time() - eval_start_time
    message = f"Evaluated {eval_id} in {eval_time:.2f}s (Avg@{k}={sample_stats.reward.mean():.2f}"
    if could_be_binary:
        assert pass_at_k is not None
        for pass_rate, pass_rate_score in pd.Series(pass_at_k.mean()).items():
            message += f", {capitalize(str(pass_rate))}: {pass_rate_score:.2f}"
    message += (
        f", Seq. Len: {avg_completion_len:.2f}, Max Seq. Len: {max_avg_completion_len:.2f}, "
        f"Min Seq. Len: {min_avg_completion_len:.2f}"
    )
    logger.success(message + ")")

    # Log statistics to monitor
    eval_metrics = {
        f"avg@{k}": float(sample_stats.reward.mean()),
        "completion_len": float(avg_completion_len),
        "max_completion_len": float(max_avg_completion_len),
        "min_completion_len": float(min_avg_completion_len),
    }
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
