import asyncio
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from huggingface_hub import whoami
from openai import AsyncOpenAI
from prime_evals import AsyncEvalsClient
from verifiers import load_environment
from verifiers.types import GenerateOutputs
from verifiers.utils.eval_utils import get_hf_hub_dataset_name, make_dataset, sanitize_metadata, save_to_disk

from prime_rl.eval.config import OfflineEvalConfig
from prime_rl.orchestrator.config import ClientConfig, EvalConfig, EvalSamplingConfig, EvalSaveConfig, ModelConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor import get_monitor
from prime_rl.utils.utils import capitalize, get_eval_dir, get_step_path
from prime_rl.utils.vf import generate_batch


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
    eval_start_time = time.perf_counter()

    # Load the eval environment
    env_name_or_id = env_name or env_id
    env = load_environment(env_id, **env_args)
    dataset = env.get_eval_dataset(n=num_examples)
    sampling_args = prepare_sampling_args(sampling_config, client_config)

    logger.info(
        f"Evaluating {env_name_or_id} ({num_examples=}, {rollouts_per_example=}) {'with default args' if env_args == {} else f'with args {env_args}'}"
    )
    # Run async generation and scoring
    results: GenerateOutputs = await generate_batch(
        env=env,
        model_name=model_config.name,
        problems=dataset.to_list(),
        clients=clients,
        rollouts_per_example=rollouts_per_example,
        sampling_args=sampling_args,
        pbar_description=f"Evaluating {env_name_or_id}",
    )

    # Parse trajectory tokens
    k = rollouts_per_example
    completion_lens: list[int] = []
    is_truncated: list[bool] = []
    states = results["state"]
    for rollout_state_raw in states:
        rollout_state: dict[str, Any] = cast(dict[str, Any], rollout_state_raw)
        trajectory = rollout_state.get("trajectory", [])
        if not trajectory:
            raise RuntimeError("Evaluation rollout is missing trajectory data.")
        total_completion_tokens = 0
        truncated = False
        for step in trajectory:
            tokens = step.get("tokens")
            if tokens is None:
                raise RuntimeError(
                    "Trajectory step is missing token data. Ensure evaluation sampling requests token_ids/logprobs."
                )
            total_completion_tokens += len(tokens["completion_ids"])
            truncated = truncated or bool(tokens.get("is_truncated") or tokens.get("overlong_prompt"))
        completion_lens.append(total_completion_tokens)
        is_truncated.append(truncated)
    results_df = pd.DataFrame(
        {
            "example_id": results["example_id"],
            "reward": results["reward"],
            "completion_len": completion_lens,
            "is_truncated": is_truncated,
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
    eval_time = time.perf_counter() - eval_start_time
    message = f"Evaluated {env_name_or_id} in {eval_time:.2f}s (Avg@{k}={results_df.reward.mean():.4f}"
    if could_be_binary:
        assert pass_at_k is not None
        for pass_rate, pass_rate_score in pd.Series(pass_at_k.mean()).items():
            message += f", {capitalize(str(pass_rate))}: {pass_rate_score:.4f}"
    message += f", Completion Length: {results_df.completion_len.mean():.2f} (±{results_df.completion_len.std():.2f}, ∈[{results_df.completion_len.min():.2f}, {results_df.completion_len.max():.2f}]), Truncated: {results_df.is_truncated.mean() * 100:.1f}%)"
    logger.success(message)

    # Log statistics to monitor
    eval_metrics = {
        f"avg@{k}": results_df.reward.mean(),
        "completion_len/avg": results_df.completion_len.mean().item(),
        "completion_len/max": results_df.completion_len.max().item(),
        "completion_len/min": results_df.completion_len.min().item(),
        "is_truncated/mean": results_df.is_truncated.mean().item(),
        "time": eval_time,
    }
    if could_be_binary:
        assert pass_at_k is not None
        eval_metrics.update(pd.Series(pass_at_k.mean()).to_dict())
    eval_metrics = {**{f"eval/{env_name_or_id}/{k}": v for k, v in eval_metrics.items()}}
    eval_metrics.update({"progress/ckpt_step": ckpt_step, "step": step or ckpt_step})
    monitor.log(eval_metrics)

    # Save results
    if save_config.disk is not None or save_config.hf is not None or save_config.env_hub:
        dataset = make_dataset(results)
        metadata_dict = sanitize_metadata(results["metadata"])

        if save_config.disk is not None:
            is_online = step is not None
            default_save_path = (
                get_step_path(get_eval_dir(output_dir), ckpt_step) / env_name_or_id
                if is_online
                else results["metadata"]["path_to_save"]
            )
            save_path = save_config.disk.path or default_save_path
            save_to_disk(dataset, metadata_dict, save_path)
            logger.info(f"Saved eval results for {env_name_or_id} to disk ({save_path})")

        if save_config.hf is not None:
            dataset_name = save_config.hf.dataset_name or get_hf_hub_dataset_name(results)
            dataset_subset = save_config.hf.dataset_subset or env.env_id
            dataset_split = save_config.hf.dataset_split or "evals"
            dataset.push_to_hub(dataset_name, dataset_subset, split=dataset_split, private=save_config.hf.private)
            default_org = whoami().get("name", "")
            repo_name = dataset_name if "/" in dataset_name else f"{default_org}/{dataset_name}"
            logger.info(
                f"Pushed {'private' if save_config.hf.private else 'public'} eval results for {env_name_or_id} to HF Hub (https://huggingface.co/datasets/{repo_name})"
            )

        if save_config.env_hub:
            eval_name = f"{env_id}--{model_config.name.replace('/', '--')}"

            # Create evaluation for environment
            create_response = await evals_client.create_evaluation(
                name=eval_name,
                environments=[{"id": env_id}],
                model_name=model_config.name,
                framework="verifiers",
                metadata=metadata_dict,
                metrics=eval_metrics,
            )

            eval_id = create_response.get("evaluation_id")
            assert eval_id is not None

            # Push samples
            await evals_client.push_samples(eval_id, dataset.to_list())

            # Finalize evaluation
            await evals_client.finalize_evaluation(eval_id, metrics=eval_metrics)

            logger.info(f"Pushed eval results for {env_id} to Environment Hub (eval_id: {eval_id})")


async def run_evals(
    clients: list[AsyncOpenAI],
    eval_config: EvalConfig | OfflineEvalConfig,
    model_config: ModelConfig,
    sampling_config: EvalSamplingConfig,
    client_config: ClientConfig,
    evals_client: AsyncEvalsClient,
    output_dir: Path,
    ckpt_step: int,
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
