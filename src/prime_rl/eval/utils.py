import asyncio
import json
import time
from itertools import cycle
from pathlib import Path
from typing import Any

import aiofiles
import numpy as np
import pandas as pd
import verifiers as vf
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from verifiers import load_environment
from verifiers.envs.environment import get_results_path

from prime_rl.eval.config import OfflineEvalConfig
from prime_rl.orchestrator.config import EvalConfig, EvalSamplingConfig, ModelConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor import get_monitor
from prime_rl.utils.utils import capitalize
from prime_rl.utils.vf import generate_group, get_completion_len, get_is_truncated

WRITE_LOCK = asyncio.Lock()


def read_existing_example_ids(results_file: Path) -> set[int]:
    example_ids = set()
    with open(results_file, "r") as f:
        for line in f:
            result = json.loads(line)
            example_id = result["example_id"]
            example_ids.add(example_id)
    return example_ids


def read_existing_results(results_file: Path) -> pd.DataFrame:
    results = []
    with open(results_file, "r") as f:
        for line in f:
            result = json.loads(line)
            results.append(
                {
                    "example_id": result["example_id"],
                    "reward": result["reward"],
                    "completion_len": result["completion_len"],
                    "is_truncated": result["is_truncated"],
                }
            )

    return pd.DataFrame(results)


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
    """Prepare sampling args for synthetic data generation."""
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


# TODO: This is a hotfix for as long as verifiers doesn't support reasoning content parsing
def merge_reasoning_content(
    completion: list[vf.ChatMessage],
    trajectory: list[vf.TrajectoryStep],
    reasoning_field: str = "reasoning_content",
) -> list[vf.ChatMessage]:
    """Parse reasoning content from the raw model response and add it to the completion."""
    # Parse responses from trajectory
    responses: list[vf.ModelResponse] = [trajectory_step["response"] for trajectory_step in trajectory]
    assistant_messages: list[vf.ChatMessage] = [c for c in completion if c.get("role") == "assistant"]
    assert len(assistant_messages) == len(responses), "Number of assistant messages and responses must match"

    for assistant_message, response in zip(assistant_messages, responses):
        assert isinstance(response, vf.ChatCompletion)
        response_message = response.choices[0].message
        if getattr(response_message, reasoning_field, None) is not None:
            assistant_message[reasoning_field] = getattr(response_message, reasoning_field)

    return completion


# TODO: Move to verifiers to avoid code drift
def make_result(state: vf.State, reasoning_field: str) -> dict:
    """Translates a finished rollout state to a synthetic dataset row."""
    completion = merge_reasoning_content(state["completion"], state["trajectory"], reasoning_field)
    result_dict = {
        "example_id": state["example_id"],
        "prompt": state["prompt"],
        "completion": completion,
        "task": state["task"],
        "reward": state["reward"],
        "generation_ms": state["timing"]["generation_ms"],
        "scoring_ms": state["timing"]["scoring_ms"],
        "total_ms": state["timing"]["total_ms"],
        "info": state.get("info", {}),
        "answer": state.get("answer", ""),
        "completion_len": get_completion_len(state),
        "is_truncated": get_is_truncated(state),
    }
    for metric_name, metric_value in state["metrics"].items():
        result_dict[metric_name] = metric_value

    result_dict["oai_tools"] = json.dumps(state["oai_tools"])

    return result_dict


async def save_result(result_dict: dict, save_file: Path):
    """Saves a finished rollout to a file."""
    async with WRITE_LOCK:
        async with aiofiles.open(save_file, "a") as f:
            await f.write(json.dumps(result_dict) + "\n")


async def make_and_save_result(state: vf.State, save_file: Path, reasoning_field: str):
    """Translates and saves a finished rollout state to a synthetic dataset row."""
    result_dict = await asyncio.to_thread(make_result, state, reasoning_field)
    await save_result(result_dict, save_file)


async def generate_and_save_group(
    client: AsyncOpenAI,
    env: vf.Environment,
    model_name: str,
    example: dict,
    index: int,
    rollouts_per_example: int,
    sampling_args: dict,
    save_file: Path,
    reasoning_field: str,
    pbar: tqdm,
    rewards_accumulator: list,
    rewards_lock: asyncio.Lock,
) -> list[vf.State]:
    logger = get_logger()
    try:
        states = await generate_group(client, env, model_name, example, rollouts_per_example, sampling_args)
        await asyncio.gather(*[make_and_save_result(state, save_file, reasoning_field) for state in states])

        # Accumulate rewards and update progress bar
        async with rewards_lock:
            group_rewards = [state["reward"] for state in states]
            rewards_accumulator.extend(group_rewards)
            avg_reward = sum(rewards_accumulator) / len(rewards_accumulator)
            pbar.set_postfix(f"Avg Reward: {avg_reward:.4f}")

        pbar.update(rollouts_per_example)
        return states
    except Exception as e:
        logger.error(f"Error evaluating group {index}: {repr(e)}")
        import traceback

        logger.debug(f"Traceback for group {index}:\n{traceback.format_exc()}")


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
    model_config: ModelConfig,
    sampling_config: EvalSamplingConfig,
    step: int | None = None,
    resume_uuid: str | None = None,
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

    if resume_uuid is not None:
        base_path = get_results_path(env_name_or_id, model_config.name, base_path=output_dir)
        # Replace the UUID directory with the one provided by `resume_uuid`
        path_to_save = base_path.parent / resume_uuid / "results.jsonl"
        existing_example_ids = read_existing_example_ids(path_to_save)
        logger.info(f"Resuming from {path_to_save}: found {len(existing_example_ids)} already-evaluated examples")

        # Filter dataset to exclude already-evaluated examples
        original_size = len(dataset)
        dataset = dataset.filter(lambda example: example["example_id"] not in existing_example_ids)
        remaining_size = len(dataset)

        logger.info(
            f"Resuming evaluation of {env_name_or_id} and saving results to {path_to_save}: filtered dataset from {original_size} to {remaining_size} remaining examples\n"
            f"({num_examples=}, {rollouts_per_example=}) {'with default args' if env_args == {} else f'with args {env_args}'} and extra_body {sampling_args['extra_body']}"
        )
    else:
        path_to_save = get_results_path(env_name_or_id, model_config.name, base_path=output_dir) / "results.jsonl"
        path_to_save.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Evaluating {env_name_or_id} ({num_examples=}, {rollouts_per_example=}) {'with default args' if env_args == {} else f'with args {env_args}'} and extra_body {sampling_args['extra_body']}\n"
            f"Saving results to {path_to_save}"
        )
    total_rollouts = len(dataset) * rollouts_per_example
    pbar = tqdm(total=total_rollouts, desc="Evaluating")

    # Create shared structure for tracking rewards
    rewards_accumulator: list = []
    rewards_lock = asyncio.Lock()

    # If resuming, populate rewards_accumulator with existing rewards
    if resume_uuid is not None:
        existing_results_df = read_existing_results(path_to_save)
        rewards_accumulator.extend(existing_results_df.reward.tolist())
        if len(rewards_accumulator) > 0:
            avg_reward = sum(rewards_accumulator) / len(rewards_accumulator)
            pbar.set_postfix(f"Avg Reward: {avg_reward:.4f}")

    # Run async generation and scoring
    all_groups = await asyncio.gather(
        *[
            generate_and_save_group(
                client,
                env,
                model_config.name,
                example,
                index,
                rollouts_per_example,
                sampling_args,
                path_to_save,
                reasoning_field,
                pbar,
                rewards_accumulator,
                rewards_lock,
            )
            for index, (client, example) in enumerate(zip(cycle(clients), dataset.to_list()))
        ]
    )
    # Parse vLLM responses
    k = rollouts_per_example
    all_states = [state for group in all_groups for state in group]
    new_results_df = pd.DataFrame(
        {
            "example_id": [state["example_id"] for state in all_states],
            "reward": [state["reward"] for state in all_states],
            "completion_len": [get_completion_len(state) for state in all_states],
            "is_truncated": [get_is_truncated(state) for state in all_states],
        }
    )

    # If resuming, combine with existing results for accurate metrics
    if resume_uuid is not None:
        existing_results_df = read_existing_results(path_to_save)
        results_df = pd.concat([existing_results_df, new_results_df], ignore_index=True)
        logger.info(
            f"Combined existing ({len(existing_results_df)}) and new ({len(new_results_df)}) results for metrics"
        )
    else:
        results_df = new_results_df

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

    # # Save results
    # if save_config.disk is not None or save_config.hf is not None or save_config.env_hub:
    #     outputs = env._prepare_rollout_results(
    #         all_states=[to_serializable_state(state) for state in states],  # type: ignore
    #         model=model_config.name,
    #         client=clients[0],  # We use the first client
    #         state_columns=None,
    #         results_path=None,
    #         gen_sampling_args=sampling_args,
    #         start_time=eval_start_time,
    #     )
    #     dataset = make_dataset(outputs)
    #     metadata_dict = sanitize_metadata(outputs["metadata"])

    #     if save_config.disk is not None:
    #         is_online = step is not None
    #         default_save_path = (
    #             get_step_path(get_eval_dir(output_dir), ckpt_step) / env_name_or_id
    #             if is_online
    #             else outputs["metadata"]["path_to_save"]
    #         )
    #         save_path = save_config.disk.path or default_save_path
    #         save_to_disk(dataset, metadata_dict, save_path)
    #         logger.info(f"Saved eval results for {env_name_or_id} to disk ({save_path})")

    #     if save_config.hf is not None:
    #         dataset_name = save_config.hf.dataset_name or get_hf_hub_dataset_name(outputs)
    #         dataset_subset = save_config.hf.dataset_subset or env.env_id
    #         dataset_split = save_config.hf.dataset_split or "evals"
    #         dataset.push_to_hub(dataset_name, dataset_subset, split=dataset_split, private=save_config.hf.private)
    #         default_org = whoami().get("name", "")
    #         repo_name = dataset_name if "/" in dataset_name else f"{default_org}/{dataset_name}"
    #         logger.info(
    #             f"Pushed {'private' if save_config.hf.private else 'public'} eval results for {env_name_or_id} to HF Hub (https://huggingface.co/datasets/{repo_name})"
    #         )

    #     if save_config.env_hub:
    #         eval_name = f"{env_id}--{model_config.name.replace('/', '--')}"

    #         # Create evaluation for environment
    #         create_response = await evals_client.create_evaluation(
    #             name=eval_name,
    #             environments=[{"id": env_id}],
    #             model_name=model_config.name,
    #             framework="verifiers",
    #             metadata=metadata_dict,
    #             metrics=eval_metrics,
    #         )

    #         eval_id = create_response.get("evaluation_id")
    #         assert eval_id is not None

    #         # Push samples
    #         await evals_client.push_samples(eval_id, dataset.to_list())

    #         # Finalize evaluation
    #         await evals_client.finalize_evaluation(eval_id, metrics=eval_metrics)

            # logger.info(
            #     f"Pushed eval results for {env_id} to Environments Hub (https://app.primeintellect.ai/dashboard/evaluations/{eval_id})"
            # )


async def run_evals(
    clients: list[AsyncOpenAI],
    eval_config: EvalConfig | OfflineEvalConfig,
    model_config: ModelConfig,
    sampling_config: EvalSamplingConfig,
    reasoning_field: str,
    output_dir: Path,
    ckpt_step: int,
    step: int | None = None,
    resume_uuid: str | None = None,
):
    await asyncio.gather(
        *[
            run_eval(
                clients=clients,
                env_id=env.id,
                env_name=env.name,
                env_args=env.args,
                num_examples=env.num_examples or eval_config.num_examples,
                reasoning_field=reasoning_field,
                rollouts_per_example=env.rollouts_per_example or eval_config.rollouts_per_example,
                output_dir=output_dir,
                model_config=model_config,
                sampling_config=sampling_config,
                ckpt_step=ckpt_step,
                step=step,
                resume_uuid=resume_uuid,
            )
            for env in eval_config.env
        ]
    )
