import asyncio
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from prime_evals import AsyncEvalsClient
from verifiers import load_environment

from prime_rl.eval.config import OfflineEvalConfig
from prime_rl.orchestrator.config import ClientConfig, EvalConfig, EvalSamplingConfig, EvalSaveConfig, ModelConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor import get_monitor
from prime_rl.utils.utils import capitalize
from prime_rl.utils.vf import run_rollouts


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


def prepare_sampling_args(sampling_config: EvalSamplingConfig, client_config: ClientConfig) -> dict[str, Any]:
    """Prepare sampling args for the client."""
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

    if client_config.server_type == "vllm":
        # Always return logprobs and token IDs from vLLM server
        sampling_args["logprobs"] = True
        extra_body: dict[str, Any] = {"return_tokens_as_token_ids": True}

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


async def run_eval(
    clients: list[AsyncOpenAI],
    env_id: str,
    env_name: str | None,
    env_args: dict,
    num_examples: int,
    rollouts_per_example: int,
    output_dir: Path,
    ckpt_step: int,
    semaphore: asyncio.Semaphore | None,
    model_config: ModelConfig,
    sampling_config: EvalSamplingConfig,
    client_config: ClientConfig,
    save_config: EvalSaveConfig,
    evals_client: AsyncEvalsClient,
    step: int | None = None,
) -> None:
    # Get the logger
    logger = get_logger()
    monitor = get_monitor()
    eval_start_time = time.time()

    # Load the eval environment
    env_name_or_id = env_name or env_id
    env = load_environment(env_id, **env_args)
    dataset = env.get_eval_dataset(n=num_examples)
    sampling_args = prepare_sampling_args(sampling_config, client_config)

    logger.info(
        f"Evaluating {env_name_or_id} ({num_examples=}, {rollouts_per_example=}) {'with default args' if env_args == {} else f'with args {env_args}'}"
    )
    # Run async generation and scoring (returns list[State])
    states = await run_rollouts(
        env=env,
        model_name=model_config.name,
        problems=dataset.to_list(),
        clients=clients,
        rollouts_per_example=rollouts_per_example,
        sampling_args=sampling_args,
        max_concurrent=None,
        pbar_description=f"Evaluating {env_name_or_id}",
    )

    # Build results from trajectory steps per rollout
    k = rollouts_per_example

    def sum_and_longest_tokens(s):
        traj = s.get("trajectory", [])
        total_in, total_out, longest = 0, 0, 0
        for step in traj:
            tokens = step.get("tokens")
            if not tokens:
                continue
            in_len = len(tokens["prompt_ids"])
            out_len = len(tokens["completion_ids"])
            total_in += in_len
            total_out += out_len
            longest = max(longest, in_len + out_len)
        return total_in, total_out, longest

    def any_truncated(s):
        traj = s.get("trajectory", [])
        for step in traj:
            resp = step.get("response")
            try:
                if (
                    resp is not None
                    and len(resp.choices) > 0
                    and getattr(resp.choices[0], "finish_reason", "") == "length"
                ):
                    return True
            except Exception:
                continue
        return False

    total_in_list, total_out_list, longest_step_list, trunc_flags = [], [], [], []
    for s in states:
        in_len, out_len, longest = sum_and_longest_tokens(s)
        total_in_list.append(in_len)
        total_out_list.append(out_len)
        longest_step_list.append(longest)
        trunc_flags.append(any_truncated(s))

    results_df = pd.DataFrame(
        {
            "example_id": [int(s.get("example_id", 0)) for s in states],
            "reward": [float(s.get("reward", 0.0)) for s in states],
            "tokens_in_total": total_in_list,
            "tokens_out_total": total_out_list,
            "tokens_total": [i + o for i, o in zip(total_in_list, total_out_list)],
            "tokens_longest_step": longest_step_list,
            "is_truncated": trunc_flags,
        }
    )
    unique_rewards = results_df.reward.unique()
    could_be_binary = set(unique_rewards).issubset({0.0, 1.0})
    if could_be_binary:
        pass_at_k = (
            results_df.groupby("example_id")
            .apply(lambda x: compute_pass_at_k(x.reward), include_groups=False)
            .apply(pd.Series)
        )
    else:
        pass_at_k = None
        logger.warning("Skipping computing pass@k rates because the task rewards appear to be non-binary")

    # Log statistics to console
    eval_time = time.time() - eval_start_time
    message = f"Evaluated {env_name_or_id} in {eval_time:.2f}s (Avg@{k}={results_df.reward.mean():.4f}"
    if could_be_binary:
        assert pass_at_k is not None
        for pass_rate, pass_rate_score in pd.Series(pass_at_k.mean()).items():
            message += f", {capitalize(str(pass_rate))}: {pass_rate_score:.4f}"
    message += (
        f", Tokens Total: {results_df.tokens_total.mean():.2f} "
        f"(±{results_df.tokens_total.std():.2f}, ∈[{results_df.tokens_total.min():.2f}, {results_df.tokens_total.max():.2f}])"
        f", Longest Step: {results_df.tokens_longest_step.mean():.2f} "
        f"(±{results_df.tokens_longest_step.std():.2f}, ∈[{results_df.tokens_longest_step.min():.2f}, {results_df.tokens_longest_step.max():.2f}])"
        f", Truncated: {results_df.is_truncated.mean() * 100:.1f}%)"
    )
    logger.success(message)

    # Log statistics to monitor
    eval_metrics = {
        f"avg@{k}": results_df.reward.mean(),
        "tokens/total_avg": results_df.tokens_total.mean().item(),
        "tokens/total_max": results_df.tokens_total.max().item(),
        "tokens/total_min": results_df.tokens_total.min().item(),
        "tokens/longest_step_avg": results_df.tokens_longest_step.mean().item(),
        "tokens/longest_step_max": results_df.tokens_longest_step.max().item(),
        "tokens/longest_step_min": results_df.tokens_longest_step.min().item(),
        "is_truncated/mean": results_df.is_truncated.mean().item(),
        "time": eval_time,
    }
    if could_be_binary:
        assert pass_at_k is not None
        eval_metrics.update(pd.Series(pass_at_k.mean()).to_dict())
    eval_metrics = {**{f"eval/{env_name_or_id}/{k}": v for k, v in eval_metrics.items()}}
    eval_metrics.update({"progress/ckpt_step": ckpt_step, "step": step or ckpt_step})
    monitor.log(eval_metrics)

    # Skipping persisted saving in State-only mode for now


async def run_evals(
    clients: list[AsyncOpenAI],
    eval_config: EvalConfig | OfflineEvalConfig,
    model_config: ModelConfig,
    sampling_config: EvalSamplingConfig,
    client_config: ClientConfig,
    evals_client: AsyncEvalsClient,
    output_dir: Path,
    ckpt_step: int,
    semaphore: asyncio.Semaphore | None = None,
    step: int | None = None,
):
    await asyncio.gather(
        *[
            run_eval(
                clients=clients,
                env_id=env.id,
                env_name=env.name,
                env_args=env.args,
                num_examples=env.num_examples or eval_config.num_examples,
                rollouts_per_example=env.rollouts_per_example or eval_config.rollouts_per_example,
                semaphore=semaphore,
                output_dir=output_dir,
                model_config=model_config,
                sampling_config=sampling_config,
                client_config=client_config,
                save_config=eval_config.save,
                evals_client=evals_client,
                ckpt_step=ckpt_step,
                step=step,
            )
            for env in eval_config.env
        ]
    )
