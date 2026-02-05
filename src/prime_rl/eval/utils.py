import asyncio
import json
import time
from itertools import cycle
from multiprocessing import Process
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import verifiers as vf
from huggingface_hub import whoami
from openai import AsyncOpenAI
from prime_evals import AsyncEvalsClient
from verifiers import load_environment
from verifiers.utils.path_utils import get_results_path
from verifiers.utils.save_utils import get_hf_hub_dataset_name, make_dataset, sanitize_metadata, save_to_disk

from prime_rl.eval.config import OfflineEvalConfig
from prime_rl.orchestrator.config import EvalConfig, EvalSamplingConfig, EvalSaveConfig, ModelConfig, RetryConfig
from prime_rl.orchestrator.utils import set_semaphore
from prime_rl.synthesize.utils import merge_reasoning_content, save_result
from prime_rl.utils.client import setup_clients, setup_evals_client
from prime_rl.utils.config import ClientConfig
from prime_rl.utils.elastic import ServerDiscovery
from prime_rl.utils.logger import ProgressTracker, get_logger, intercept_verifiers_logging, reset_logger, setup_logger
from prime_rl.utils.monitor import get_monitor
from prime_rl.utils.utils import capitalize, get_eval_dir, get_step_path
from prime_rl.utils.vf import (
    generate_group,
    generate_rollout,
    get_completion_len,
    get_is_truncated,
    to_serializable_state,
)

WRITE_LOCK = asyncio.Lock()


def read_existing_rollout_ids(results_file: Path) -> set[tuple[int, int]]:
    """Read existing (example_id, rollout_idx) pairs from results file for resume."""
    rollout_ids: set[tuple[int, int]] = set()
    with open(results_file, "r") as f:
        for line in f:
            result = json.loads(line)
            rollout_ids.add((result["example_id"], result["rollout_idx"]))
    return rollout_ids


def read_existing_results(results_file: Path) -> pd.DataFrame:
    results = []
    with open(results_file, "r") as f:
        for line in f:
            result = json.loads(line)
            info = result.get("info", {})
            prime_info = info.get("prime_rl", {})
            rollout_status = prime_info.get("rollout_status", "ok")
            results.append(
                {
                    "example_id": result["example_id"],
                    "reward": result["reward"],
                    "completion_len": result["completion_len"],
                    "is_truncated": result["is_truncated"],
                    "rollout_status": rollout_status,
                }
            )

    return pd.DataFrame(results)


def read_completed_example_ids(results_file: Path, rollouts_per_example: int) -> set[int]:
    """Read example_ids that have complete groups (all rollouts_per_example completed) for group-based resume."""
    example_id_counts: dict[int, int] = {}
    with open(results_file, "r") as f:
        for line in f:
            result = json.loads(line)
            example_id = result["example_id"]
            example_id_counts[example_id] = example_id_counts.get(example_id, 0) + 1

    # Return example_ids that have exactly rollouts_per_example results
    completed_example_ids = {
        example_id for example_id, count in example_id_counts.items() if count == rollouts_per_example
    }
    return completed_example_ids


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


def prepare_sampling_args(sampling_config: EvalSamplingConfig) -> dict[str, Any]:
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


# TODO: Move to verifiers to avoid code drift
def make_result(output: vf.RolloutOutput, reasoning_field: str, rollout_idx: int) -> dict:
    """Translates a finished rollout state to a synthetic dataset row."""
    completion = merge_reasoning_content(output["completion"], output["trajectory"], reasoning_field)
    result_dict = {
        "example_id": output["example_id"],
        "rollout_idx": rollout_idx,
        "prompt": output["prompt"],
        "completion": completion,
        "task": output["task"],
        "reward": output["reward"],
        "generation_ms": output["timing"]["generation_ms"],
        "scoring_ms": output["timing"]["scoring_ms"],
        "total_ms": output["timing"]["total_ms"],
        "info": output.get("info", {}),
        "answer": output.get("answer", ""),
        "completion_len": get_completion_len(output),
        "is_truncated": get_is_truncated(output),
    }
    for metric_name, metric_value in output["metrics"].items():
        result_dict[metric_name] = metric_value

    result_dict["oai_tools"] = json.dumps(output.get("oai_tools", []))

    return result_dict


async def make_and_save_result(output: vf.RolloutOutput, save_file: Path, reasoning_field: str, rollout_idx: int):
    """Translates and saves a finished rollout state to a synthetic dataset row."""
    result_dict = await asyncio.to_thread(make_result, output, reasoning_field, rollout_idx)
    await save_result(result_dict, save_file)


def _is_valid_reward(reward: Any) -> bool:
    if reward is None:
        return False
    if isinstance(reward, float) and np.isnan(reward):
        return False
    return isinstance(reward, (int, float))


def _format_avg_reward(rewards: list) -> str:
    valid = [r for r in rewards if _is_valid_reward(r)]
    if not valid:
        return "N/A"
    return f"{(sum(valid) / len(valid)):.4f}"


async def generate_and_save_rollout(
    client: AsyncOpenAI,
    env: vf.Environment,
    model_name: str,
    example: dict,
    rollout_idx: int,
    env_name_or_id: str,
    sampling_args: dict,
    save_file: Path | None,
    reasoning_field: str,
    retry_config: RetryConfig,
    pbar: ProgressTracker,
    rewards_accumulator: list,
    rewards_lock: asyncio.Lock,
) -> vf.RolloutOutput:
    """Generate and optionally save a single rollout, updating progress bar per-rollout."""
    logger = get_logger()

    # max_retries = max_attempts - 1 (verifiers counts retries, not attempts)
    max_retries = max(0, retry_config.max_attempts - 1)
    state = await generate_rollout(client, env, model_name, example, sampling_args, max_retries=max_retries)

    # Check if the rollout failed with an error (after retries exhausted)
    if state.get("error") is not None:
        logger.error(
            f"Rollout generation failed after retries; recording no_response "
            f"(env={env_name_or_id}, example_id={example.get('example_id')}, rollout_idx={rollout_idx}): {state.get('error')}"
        )

    # Save with rollout_idx if streaming saves enabled.
    if save_file is not None:
        result_dict = await asyncio.to_thread(make_result, state, reasoning_field, rollout_idx)
        await save_result(result_dict, save_file)

    # Update running average immediately.
    async with rewards_lock:
        rewards_accumulator.append(state.get("reward"))
        pbar.set_postfix({"Avg Reward": _format_avg_reward(rewards_accumulator)})
        pbar.update(1)

    return state


async def generate_and_save_group(
    client: AsyncOpenAI,
    env: vf.Environment,
    model_name: str,
    example: dict,
    rollouts_per_example: int,
    env_name_or_id: str,
    sampling_args: dict,
    save_file: Path | None,
    reasoning_field: str,
    retry_config: RetryConfig,
    pbar: ProgressTracker,
    rewards_accumulator: list,
    rewards_lock: asyncio.Lock,
) -> list[vf.RolloutOutput]:
    """Generate a group of rollouts, save results, and update progress (group-level)."""
    logger = get_logger()

    # max_retries = max_attempts - 1 (verifiers counts retries, not attempts)
    max_retries = max(0, retry_config.max_attempts - 1)
    states = await generate_group(
        client, env, model_name, example, rollouts_per_example, sampling_args, max_retries=max_retries
    )

    # Log errors for any states that failed (after retries exhausted)
    for state in states:
        if state.get("error") is not None:
            logger.error(
                f"Rollout in group failed after retries "
                f"(env={env_name_or_id}, example_id={example.get('example_id')}): {state.get('error')}"
            )

    # Save results for the generated group.
    if save_file is not None:
        await asyncio.gather(
            *[
                make_and_save_result(state, save_file, reasoning_field, rollout_idx)
                for rollout_idx, state in enumerate(states)
            ]
        )

    # Update running average after group completes.
    async with rewards_lock:
        for state in states:
            rewards_accumulator.append(state.get("reward"))
        pbar.set_postfix({"Avg Reward": _format_avg_reward(rewards_accumulator)})
        pbar.update(rollouts_per_example)

    return states


async def run_eval(
    clients: list[AsyncOpenAI],
    env_id: str,
    env_name: str | None,
    env_args: dict,
    num_examples: int,
    rollouts_per_example: int,
    reasoning_field: str,
    output_dir: Path,
    ckpt_step: int,
    model_name: str,
    sampling_config: EvalSamplingConfig,
    save_config: EvalSaveConfig,
    retry_config: RetryConfig,
    evals_client: AsyncEvalsClient,
    per_rollout: bool = False,
    step: int | None = None,
    resume_path: Path | None = None,
) -> None:
    # Get the logger
    logger = get_logger()
    monitor = get_monitor()
    eval_start_time = time.perf_counter()

    # Load the eval environment
    env_name_or_id = env_name or env_id
    env = load_environment(env_id, **env_args)
    dataset = env.get_eval_dataset(n=num_examples)
    sampling_args = prepare_sampling_args(sampling_config)

    # Determine streaming save path
    if save_config.stream:
        if resume_path is not None:
            # resume_path points to a directory containing results.jsonl
            path_to_save = resume_path / "results.jsonl"
        else:
            base_path = get_results_path(env_name_or_id, model_name, base_path=output_dir)
            path_to_save = base_path / "results.jsonl"
        path_to_save.parent.mkdir(parents=True, exist_ok=True)
    else:
        path_to_save = None

    # Create shared structure for tracking rewards
    rewards_accumulator: list = []
    rewards_lock = asyncio.Lock()
    examples = dataset.to_list()

    if per_rollout:
        # Per-rollout scheduling: enables live progress updates and per-rollout resume
        all_rollouts: list[tuple[dict, int]] = [
            (example, rollout_idx) for example in examples for rollout_idx in range(rollouts_per_example)
        ]

        # Filter out already-completed rollouts on resume
        if resume_path is not None:
            assert path_to_save is not None
            existing_rollout_ids = read_existing_rollout_ids(path_to_save)
            original_count = len(all_rollouts)
            all_rollouts = [
                (ex, ridx) for ex, ridx in all_rollouts if (ex["example_id"], ridx) not in existing_rollout_ids
            ]
            skipped_count = original_count - len(all_rollouts)
            logger.info(
                f"Resuming from {path_to_save}: skipping {skipped_count} already-completed rollouts, "
                f"{len(all_rollouts)} remaining"
            )
            # Populate rewards_accumulator with existing rewards
            existing_results_df = read_existing_results(path_to_save)
            rewards_accumulator.extend(existing_results_df.reward.tolist())

        total_rollouts = len(all_rollouts)
        logger.info(
            f"Evaluating {model_name} on {env_name_or_id} ({num_examples=}, {rollouts_per_example=}) "
            f"{'with default args' if env_args == {} else f'with args {env_args}'} and extra_body {sampling_args['extra_body']}\n"
            f"{'Saving results to ' + str(path_to_save) if save_config.stream else 'Results will be saved at end of evaluation'}"
        )

        pbar = ProgressTracker(total=total_rollouts, desc="Evaluating")
        if rewards_accumulator:
            pbar.set_postfix({"Avg Reward": _format_avg_reward(rewards_accumulator)})

        all_states = await asyncio.gather(
            *[
                generate_and_save_rollout(
                    client,
                    env,
                    model_name,
                    example,
                    rollout_idx,
                    env_name_or_id,
                    sampling_args,
                    path_to_save,
                    reasoning_field,
                    retry_config,
                    pbar,
                    rewards_accumulator,
                    rewards_lock,
                )
                for client, (example, rollout_idx) in zip(cycle(clients), all_rollouts)
            ]
        )
    else:
        # Group-based scheduling: preserves group-based rubric scoring
        # Filter out completed example_ids on resume
        if resume_path is not None and path_to_save is not None and path_to_save.exists():
            completed_example_ids = read_completed_example_ids(path_to_save, rollouts_per_example)
            original_count = len(examples)
            examples = [ex for ex in examples if ex["example_id"] not in completed_example_ids]
            skipped_count = original_count - len(examples)
            logger.info(
                f"Resuming from {path_to_save}: skipping {skipped_count} already-completed example_ids "
                f"({len(completed_example_ids)} complete groups), {len(examples)} remaining"
            )
            # Populate rewards_accumulator with existing rewards for metrics
            existing_results_df = read_existing_results(path_to_save)
            rewards_accumulator.extend(existing_results_df.reward.tolist())

        total_rollouts = len(examples) * rollouts_per_example
        logger.info(
            f"Evaluating {env_name_or_id} ({num_examples=}, {rollouts_per_example=}) "
            f"{'with default args' if env_args == {} else f'with args {env_args}'} and extra_body {sampling_args['extra_body']}\n"
            f"{'Saving results to ' + str(path_to_save) if save_config.stream else 'Results will be saved at end of evaluation'}"
        )

        pbar = ProgressTracker(total=total_rollouts, desc="Evaluating")
        if rewards_accumulator:
            pbar.set_postfix({"Avg Reward": _format_avg_reward(rewards_accumulator)})

        group_states_list = await asyncio.gather(
            *[
                generate_and_save_group(
                    client,
                    env,
                    model_name,
                    example,
                    rollouts_per_example,
                    env_name_or_id,
                    sampling_args,
                    path_to_save if save_config.stream else None,
                    reasoning_field,
                    retry_config,
                    pbar,
                    rewards_accumulator,
                    rewards_lock,
                )
                for client, example in zip(cycle(clients), examples)
            ]
        )
        all_states = [state for group in group_states_list for state in group]

    k = rollouts_per_example
    new_results_df = pd.DataFrame(
        {
            "example_id": [state["example_id"] for state in all_states],
            "reward": [state["reward"] for state in all_states],
            "completion_len": [get_completion_len(state) for state in all_states],
            "is_truncated": [get_is_truncated(state) for state in all_states],
            "rollout_status": [
                (state.get("info", {}) or {}).get("prime_rl", {}).get("rollout_status", "ok") for state in all_states
            ],
        }
    )

    # If resuming, combine with existing results for accurate metrics
    if resume_path is not None:
        assert path_to_save is not None
        existing_results_df = read_existing_results(path_to_save)
        results_df = pd.concat([existing_results_df, new_results_df], ignore_index=True)
        logger.info(
            f"Combined existing ({len(existing_results_df)}) and new ({len(new_results_df)}) results for metrics"
        )
    else:
        results_df = new_results_df

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
    eval_time = time.perf_counter() - eval_start_time
    no_response_rate = (
        float((results_df.rollout_status == "no_response").mean()) if "rollout_status" in results_df else 0.0
    )
    message = f"Evaluated {model_name} on {env_name_or_id} in {eval_time:.2f}s (Avg@{k}={results_df.reward.mean():.4f}"
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
        f"avg@{k}": results_df.reward.mean(),
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
    eval_metrics = {**{f"eval/{env_name_or_id}/{k}": v for k, v in eval_metrics.items()}}
    eval_metrics.update({"progress/ckpt_step": ckpt_step, "step": step or ckpt_step})
    monitor.log(eval_metrics, step=None)

    # Save results
    if save_config.disk is not None or save_config.hf is not None or save_config.env_hub:
        outputs = env._prepare_rollout_results(
            all_states=[to_serializable_state(state) for state in all_states],  # type: ignore
            model=model_name,
            client=clients[0],  # We use the first client
            state_columns=None,
            results_path=None,
            gen_sampling_args=sampling_args,
            start_time=eval_start_time,
        )
        dataset = make_dataset(outputs)
        metadata_dict = sanitize_metadata(outputs["metadata"])

        if save_config.disk is not None:
            is_online = step is not None
            default_save_path = (
                get_step_path(get_eval_dir(output_dir), ckpt_step) / env_name_or_id
                if is_online
                else outputs["metadata"]["path_to_save"]
            )
            save_path = save_config.disk.path or default_save_path
            save_to_disk(dataset, metadata_dict, save_path)
            logger.info(f"Saved eval results for {env_name_or_id} to disk ({save_path})")

        if save_config.hf is not None:
            dataset_name = save_config.hf.dataset_name or get_hf_hub_dataset_name(outputs)
            dataset_subset = save_config.hf.dataset_subset or env.env_id
            dataset_split = save_config.hf.dataset_split or "evals"
            dataset.push_to_hub(dataset_name, dataset_subset, split=dataset_split, private=save_config.hf.private)
            default_org = whoami().get("name", "")
            repo_name = dataset_name if "/" in dataset_name else f"{default_org}/{dataset_name}"
            logger.info(
                f"Pushed {'private' if save_config.hf.private else 'public'} eval results for {env_name_or_id} to HF Hub (https://huggingface.co/datasets/{repo_name})"
            )

        if save_config.env_hub:
            eval_name = f"{env_id}--{model_name.replace('/', '--')}"

            # Create evaluation for environment
            create_response = await evals_client.create_evaluation(
                name=eval_name,
                environments=[{"id": env_id}],
                model_name=model_name,
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

            logger.info(
                f"Pushed eval results for {env_id} to Environments Hub (https://app.primeintellect.ai/dashboard/evaluations/{eval_id})"
            )


async def run_evals(
    clients: list[AsyncOpenAI],
    eval_config: EvalConfig | OfflineEvalConfig,
    model_name: str,
    sampling_config: EvalSamplingConfig,
    evals_client: AsyncEvalsClient,
    reasoning_field: str,
    output_dir: Path,
    ckpt_step: int,
    step: int | None = None,
    resume_path: Path | None = None,
):
    logger = get_logger()

    async def run_eval_safe(env):
        """Wrapper to catch RetryError and skip env."""
        await run_eval(
            clients=clients,
            env_id=env.id,
            env_name=env.name,
            env_args=env.args,
            num_examples=env.num_examples or eval_config.num_examples,
            reasoning_field=reasoning_field,
            rollouts_per_example=env.rollouts_per_example or eval_config.rollouts_per_example,
            output_dir=output_dir,
            model_name=model_name,
            sampling_config=sampling_config,
            save_config=eval_config.save,
            retry_config=eval_config.retry,
            evals_client=evals_client,
            per_rollout=eval_config.per_rollout,
            ckpt_step=ckpt_step,
            step=step,
            resume_path=resume_path,
        )
        return True

    results = await asyncio.gather(*[run_eval_safe(env) for env in eval_config.env])

    skipped = sum(1 for r in results if r is None)
    if skipped > 0:
        logger.warning(f"Skipped {skipped}/{len(eval_config.env)} environments due to retry exhaustion")


async def _get_clients(client_config: ClientConfig, model_name: str) -> list[AsyncOpenAI]:
    """Discover inference clients via elastic discovery or static config."""
    if not client_config.is_elastic:
        return setup_clients(client_config)

    discovery = ServerDiscovery.from_config(client_config, model_name)
    return await discovery.wait_for_clients()


def _run_evals_in_subprocess(
    client_config: ClientConfig,
    eval_config: EvalConfig,
    model_config: ModelConfig,
    sampling_config: EvalSamplingConfig,
    reasoning_field: str,
    output_dir: str,
    ckpt_step: int,
    step: int | None,
    max_concurrent: int,
    json_logging: bool = False,
):
    """Entry point for eval subprocess. Creates its own event loop and clients."""
    # Setup logger for subprocess (reset first since we inherit parent's global state when forked)
    reset_logger()
    logger = setup_logger("info", json_logging=json_logging)
    intercept_verifiers_logging(level="info")
    logger.info(f"Eval subprocess started for checkpoint step {ckpt_step}")

    evals_client = setup_evals_client()

    async def _run():
        await set_semaphore(max_concurrent)

        model_name = model_config.name
        if model_config.lora and ckpt_step > 0:
            model_name = model_config.lora.name
        clients = await _get_clients(client_config, model_name)
        logger.info(f"Using {len(clients)} inference client(s) for model {model_name}")

        await run_evals(
            clients=clients,
            eval_config=eval_config,
            model_name=model_name,
            sampling_config=sampling_config,
            evals_client=evals_client,
            reasoning_field=reasoning_field,
            output_dir=Path(output_dir),
            ckpt_step=ckpt_step,
            step=step,
        )

    asyncio.run(_run())
    get_monitor().close()
    logger.info(f"Eval subprocess finished for checkpoint step {ckpt_step}")


async def run_evals_subprocess(
    client_config: ClientConfig,
    eval_config: EvalConfig | OfflineEvalConfig,
    model_config: ModelConfig,
    sampling_config: EvalSamplingConfig,
    reasoning_field: str,
    output_dir: Path,
    ckpt_step: int,
    step: int | None = None,
    max_concurrent: int = -1,
    json_logging: bool = False,
):
    """Run evals in a separate subprocess to isolate the event loop.

    This prevents eval's async work from causing event loop lag in the
    orchestrator's main loop. The orchestrator blocks until eval completes.
    """
    logger = get_logger()
    logger.info(f"Spawning eval subprocess for checkpoint step {ckpt_step}")

    process = Process(
        target=_run_evals_in_subprocess,
        args=(
            client_config,
            eval_config,
            model_config,
            sampling_config,
            reasoning_field,
            str(output_dir),
            ckpt_step,
            step,
            max_concurrent,
            json_logging,
        ),
        daemon=True,
    )
    process.start()

    # Wait for process to complete without blocking the event loop
    while process.is_alive():
        await asyncio.sleep(0.5)

    if process.exitcode != 0:
        logger.error(f"Eval subprocess failed with exit code {process.exitcode}")
    else:
        logger.success(f"Eval subprocess completed for checkpoint step {ckpt_step}")
