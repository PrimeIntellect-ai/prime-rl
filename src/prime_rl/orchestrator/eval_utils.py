import time
from typing import Any

import numpy as np
import pandas as pd
import verifiers as vf

from prime_rl.orchestrator.config import EvalSamplingConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor import get_monitor
from prime_rl.utils.utils import capitalize
from prime_rl.utils.vf import evaluate, get_completion_len


def get_eval_sampling_args(sampling_config: EvalSamplingConfig) -> dict[str, Any]:
    """Get sampling args for evaluation."""
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

    extra_body: dict[str, Any] = sampling_config.extra_body.copy()

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


async def evaluate_env(
    env: vf.Environment,
    env_name: str,
    client: vf.ClientConfig,
    model_name: str,
    sampling_args: dict,
    num_examples: int,
    rollouts_per_example: int,
    ckpt_step: int,
    step: int | None,
):
    logger = get_logger()
    logger.info(f"Evaluating {env_name} ({num_examples=}, {rollouts_per_example=})")
    eval_start_time = time.perf_counter()
    results = await evaluate(env, client, model_name, sampling_args, num_examples, rollouts_per_example)
    eval_time = time.perf_counter() - eval_start_time

    rows = []
    for output in results["outputs"]:
        rows.append(
            {
                "example_id": output["example_id"],
                "reward": output["reward"],
                "completion_len": get_completion_len(output),
                "is_truncated": output["is_truncated"],
                "has_error": output.get("error") is not None,
                "no_response": not output.get("completion"),
            }
        )
    results_df = pd.DataFrame(rows)

    unique_rewards = results_df.reward.dropna().unique()
    could_be_binary = set(unique_rewards).issubset({0.0, 1.0})
    if could_be_binary:
        pass_at_k = (
            results_df.groupby("example_id")
            .apply(lambda x: compute_pass_at_k(x.reward.dropna()), include_groups=False)
            .apply(pd.Series)
        )
    else:
        pass_at_k = None
        logger.warning("Skipping computing pass@k rates because the task rewards appear to be non-binary")

    # Log statistics to console
    no_response_rate = (
        float((results_df.rollout_status == "no_response").mean()) if "rollout_status" in results_df else 0.0
    )
    message = f"Evaluated {env.env_id} in {eval_time:.2f}s (Avg@{rollouts_per_example}={results_df.reward.mean():.4f}"
    if could_be_binary:
        assert pass_at_k is not None
        for pass_rate, pass_rate_score in pd.Series(pass_at_k.mean()).items():
            message += f", {capitalize(str(pass_rate))}: {pass_rate_score:.4f}"
    message += (
        f", No-response: {no_response_rate * 100:.1f}%"
        f", Completion Length: {results_df.completion_len.mean():.2f} (±{results_df.completion_len.std():.2f}, ∈[{results_df.completion_len.min():.2f}, {results_df.completion_len.max():.2f}])"
        f", Truncated: {results_df.is_truncated.mean() * 100:.1f}%)"
    )
    logger.success(message)

    # Log statistics to monitor
    eval_metrics = {
        f"avg@{rollouts_per_example}": results_df.reward.mean(),
        "no_response/pct": no_response_rate * 100.0,
        "no_response/count": int((results_df.rollout_status == "no_response").sum())
        if "rollout_status" in results_df
        else 0,
        "completion_len/avg": results_df.completion_len.mean().item(),
        "completion_len/max": results_df.completion_len.max().item(),
        "completion_len/min": results_df.completion_len.min().item(),
        "is_truncated/mean": results_df.is_truncated.mean().item(),
        "time": eval_time,
    }
    if could_be_binary:
        assert pass_at_k is not None
        eval_metrics.update(pd.Series(pass_at_k.mean()).to_dict())
    eval_metrics = {**{f"eval/{env_name}/{k}": v for k, v in eval_metrics.items()}}
    eval_metrics.update({"progress/ckpt_step": ckpt_step, "step": step or ckpt_step})
    monitor = get_monitor()
    monitor.log(eval_metrics, step=None)
