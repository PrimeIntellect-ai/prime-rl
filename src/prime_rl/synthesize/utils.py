import asyncio
import json
from itertools import cycle
from pathlib import Path
from typing import Any

import aiofiles
import verifiers as vf
from openai import AsyncOpenAI
from verifiers import load_environment
from verifiers.utils.path_utils import get_results_path

from prime_rl.orchestrator.config import ClientConfig, EvalSamplingConfig, ModelConfig
from prime_rl.utils.logger import ProgressTracker, get_logger
from prime_rl.utils.vf import generate_group

WRITE_LOCK = asyncio.Lock()


def prepare_sampling_args(sampling_config: EvalSamplingConfig, client_config: ClientConfig) -> dict[str, Any]:
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
    }
    for metric_name, metric_value in state["metrics"].items():
        result_dict[metric_name] = metric_value

    result_dict["oai_tools"] = json.dumps(state.get("oai_tools", []))

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
    pbar: ProgressTracker,
) -> None:
    logger = get_logger()
    try:
        states = await generate_group(client, env, model_name, example, rollouts_per_example, sampling_args)
        await asyncio.gather(*[make_and_save_result(state, save_file, reasoning_field) for state in states])
        pbar.update(rollouts_per_example)
    except Exception as e:
        logger.error(f"Error generating synthetic data for group {index}: {repr(e)}")


async def generate_synthetic_data(
    clients: list[AsyncOpenAI],
    env_id: str,
    env_name: str | None,
    env_args: dict,
    num_examples: int,
    rollouts_per_example: int,
    skip_first: int,
    reasoning_field: str,
    output_dir: Path,
    model_config: ModelConfig,
    sampling_config: EvalSamplingConfig,
    client_config: ClientConfig,
) -> None:
    """Generates synthetic data for an environment."""
    # Get the logger
    logger = get_logger()

    # Load the environment
    env_name_or_id = env_name or env_id
    env = load_environment(env_id, **env_args)
    try:
        dataset = env.get_dataset(n=num_examples + skip_first)
    except ValueError:
        logger.warning(f"Could not find a training dataset for {env_name_or_id}. Falling back to eval dataset.")
        dataset = env.get_eval_dataset(n=num_examples + skip_first)
    if skip_first > 0:
        dataset = dataset.skip(skip_first)
    sampling_args = prepare_sampling_args(sampling_config, client_config)
    path_to_save = get_results_path(env_name_or_id, model_config.name, base_path=output_dir) / "results.jsonl"
    path_to_save.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Generating synthetic data for {env_name_or_id} (num_examples={len(dataset)}, {rollouts_per_example=}) {'with default args' if env_args == {} else f'with args {env_args}'}"
    )

    total_rollouts = len(dataset) * rollouts_per_example
    pbar = ProgressTracker(total=total_rollouts, desc="Generating synthetic data")

    await asyncio.gather(
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
            )
            for index, (client, example) in enumerate(zip(cycle(clients), dataset.to_list()))
        ]
    )

    logger.info(f"Synthetic data generated for {env_name_or_id} and saved to {path_to_save}")
