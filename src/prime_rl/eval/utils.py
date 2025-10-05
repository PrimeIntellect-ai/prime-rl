import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from verifiers import load_environment
from verifiers.utils.eval_utils import make_dataset, make_metadata

from prime_rl.eval.config import OfflineEvalConfig
from prime_rl.orchestrator.config import ClientConfig, EvalConfig, EvalSamplingConfig, ModelConfig
from prime_rl.orchestrator.utils import parse_is_truncated_completions, parse_num_completion_tokens
from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor import get_monitor
from prime_rl.utils.utils import capitalize, get_eval_dir, get_step_path


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
    client: AsyncOpenAI,
    environment_id: str,
    env_args: dict,
    model_name: str,
    num_examples: int,
    rollouts_per_example: int,
    max_concurrent: int,
    sampling_args: dict,
    output_dir: Path,
    save_to_disk: bool,
    save_to_hf_hub: bool,
    save_to_env_hub: bool,
    hf_hub_dataset_name: str | None,
    step: int | None,
) -> None:
    """Eval a single environment. Analogous to `vf-eval`"""
    logger, monitor = get_logger(), get_monitor()
    eval_start_time = time.time()

    # Load environment
    env, model = environment_id, model_name  # Shorthands
    logger.debug(f"Loading {env} environment with args {env_args}")
    vf_env = load_environment(env_id=env, **env_args)

    logger.debug(
        f"Evaluating {env} against {model} ({num_examples=}, {rollouts_per_example=}, {max_concurrent=}, {sampling_args=})"
    )
    outputs = vf_env.evaluate(
        client=client,
        model=model,
        sampling_args=sampling_args,
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        max_concurrent=max_concurrent,
        score_rollouts=True,
        interleave_scoring=True,
    )

    # Compute statistics
    k, n = rollouts_per_example, len(outputs.reward) // rollouts_per_example
    responses = [state["responses"] for state in outputs.state]
    results = pd.DataFrame(
        {
            "example_id": list(range(n)) * k,
            "reward": outputs.reward,
            "completion_len": parse_num_completion_tokens(responses),
            "is_truncated": parse_is_truncated_completions(responses),
        }
    )
    unique_rewards = results.reward.unique()
    could_be_binary = set(unique_rewards).issubset({0.0, 1.0})
    pass_at_k = None
    if could_be_binary:
        pass_at_k = (
            results.groupby("example_id")
            .apply(lambda x: compute_pass_at_k(x.reward), include_groups=False)
            .apply(pd.Series)
        )

    # Log statistics
    eval_time = time.time() - eval_start_time
    message = f"Evaluated {env} in {eval_time:.2f}s (Avg@{k}={results.reward.mean():.4f}"
    if could_be_binary:
        assert pass_at_k is not None
        for pass_rate, pass_rate_score in pd.Series(pass_at_k.mean()).items():
            message += f", {capitalize(str(pass_rate))}: {pass_rate_score:.4f}"
    message += f", Completion Length: {results.completion_len.mean():.2f} (±{results.completion_len.std():.2f}, ∈[{results.completion_len.min():.2f}, {results.completion_len.max():.2f}]), Truncated: {results.is_truncated.mean() * 100:.1f}%)"
    logger.success(message)

    # Log statistics to monitor
    eval_metrics = {f"avg@{k}": results.reward.mean()}
    if could_be_binary:
        assert pass_at_k is not None
        eval_metrics.update(pd.Series(pass_at_k.mean()).to_dict())
    eval_metrics = {**{f"eval/{env}/{k}": v for k, v in eval_metrics.items()}}
    monitor.log({**eval_metrics, "step": step})

    eval_completion_len_metrics = {
        f"eval/{env}/completion_len/mean": results.completion_len.mean(),
        f"eval/{env}/completion_len/max": results.completion_len.max(),
        f"eval/{env}/completion_len/min": results.completion_len.min(),
        "step": step,
    }
    monitor.log(eval_completion_len_metrics)

    # Log timing metrics to monitor
    time_metrics = {
        f"eval/{env}/time": eval_time,
        "step": step,
    }
    monitor.log(time_metrics)

    if save_to_disk or save_to_hf_hub or save_to_env_hub:
        dataset = make_dataset(outputs, num_examples, rollouts_per_example)
        metadata = make_metadata(
            env, model, num_examples, rollouts_per_example, sampling_args, eval_time, eval_start_time, outputs
        )

        uuid_str = str(uuid.uuid4())[:8]
        env_model_str = f"{env}--{model.replace('/', '--')}"
        if save_to_disk:
            # Save results to disk
            if step is not None:
                eval_dir = get_step_path(get_eval_dir(output_dir), step)
            else:
                eval_dir = get_eval_dir(output_dir)
            results_path = eval_dir / env_model_str / uuid_str
            results_path.mkdir(parents=True, exist_ok=True)
            dataset.to_json(results_path / "results.jsonl")
            with open(results_path / "metadata.json", "w") as f:
                json.dump(metadata, f)
            logger.info(f"Saved eval results to {results_path}")

        if save_to_hf_hub:
            assert hf_hub_dataset_name is not None
            dataset.push_to_hub(hf_hub_dataset_name, env)
            logger.info(f"Pushed eval results to HF Hub (https://huggingface.co/datasets/{hf_hub_dataset_name})")

        if save_to_env_hub:
            raise NotImplementedError("Pushing eval results to PI Environment Hub is not yet supported.")


async def run_evals(
    client: AsyncOpenAI,
    eval_config: EvalConfig | OfflineEvalConfig,
    model_config: ModelConfig,
    sampling_config: EvalSamplingConfig,
    client_config: ClientConfig,
    output_dir: Path,
    step: int | None,
):
    await asyncio.gather(
        *[
            run_eval(
                client=client,
                environment_id=environment_id,
                env_args=eval_config.environment_args.get(environment_id, {}),
                model_name=model_config.name,
                num_examples=num_examples,
                rollouts_per_example=rollouts_per_example,
                max_concurrent=max_concurrent,
                sampling_args=prepare_sampling_args(sampling_config, client_config),
                output_dir=output_dir,
                save_to_disk=eval_config.save_to_disk,
                save_to_hf_hub=eval_config.save_to_hf_hub,
                save_to_env_hub=eval_config.save_to_env_hub,
                hf_hub_dataset_name=eval_config.hf_hub_dataset_name,
                step=step,
            )
            for environment_id, num_examples, rollouts_per_example, max_concurrent in zip(
                eval_config.environment_ids,
                eval_config.num_examples,
                eval_config.rollouts_per_example,
                eval_config.max_concurrent,
            )
        ]
    )
