import asyncio
from itertools import cycle
from typing import Any, cast

import verifiers as vf
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion

from prime_rl.utils.logger import ProgressTracker


async def generate_group(
    client: AsyncOpenAI,
    env: vf.Environment,
    model_name: str,
    example: dict,
    rollouts_per_example: int,
    sampling_args: dict,
    max_retries: int = 0,
    state_columns: list[str] = ["trajectory"],
) -> list[vf.RolloutOutput]:
    """Asynchronously generate and score rollouts for a single group."""
    group_inputs = [vf.RolloutInput(**example) for _ in range(rollouts_per_example)]
    return await env.run_group(
        group_inputs=group_inputs,
        client=client,
        model=model_name,
        sampling_args=sampling_args,
        max_retries=max_retries,
        state_columns=state_columns,
    )


async def generate_rollout(
    client: AsyncOpenAI,
    env: vf.Environment,
    model_name: str,
    example: dict,
    sampling_args: dict,
    max_retries: int = 0,
    state_columns: list[str] = ["trajectory"],
) -> vf.RolloutOutput:
    """Asynchronously generate and score a single rollout."""
    rollout_input = vf.RolloutInput(**example)
    return await env.run_rollout(
        rollout_input, client, model_name, sampling_args, max_retries=max_retries, state_columns=state_columns
    )


async def generate_batch(
    clients: list[AsyncOpenAI],
    env: vf.Environment,
    model_name: str,
    examples: list[dict],
    rollouts_per_example: int,
    sampling_args: dict,
    pbar_description: str = "Generating rollouts",
) -> list[vf.RolloutOutput]:
    """Asynchronously generate and score rollouts for a list of groups (batch)."""

    total_rollouts = len(examples) * rollouts_per_example
    pbar = ProgressTracker(total=total_rollouts, desc=pbar_description)

    async def generate_group_with_progress(client, example):
        """Generate rollouts for one problem and update progress."""
        result = await generate_group(client, env, model_name, example, rollouts_per_example, sampling_args)
        pbar.update(rollouts_per_example)
        return result

    try:
        group_outputs_list: list[list[vf.RolloutOutput]] = await asyncio.gather(
            *[generate_group_with_progress(client, example) for client, example in zip(cycle(clients), examples)]
        )
    finally:
        pbar.close()

    return [output for group_outputs in group_outputs_list for output in group_outputs]


def get_prompt_len(state: vf.State) -> int:
    """Compute the number of prompt tokens from vf.State. Defined as the number of prompt IDs from the first trajectory step. If raw tokens are not available, falls back to checking the usage of the first response."""
    if not state["trajectory"]:
        return 0
    first_step = state["trajectory"][0]
    if first_step["tokens"] is not None:
        return len(first_step["tokens"]["prompt_ids"])
    first_step_response = cast(ChatCompletion, first_step["response"])
    return getattr(first_step_response.usage, "prompt_tokens", 0)


def get_seq_len(state: vf.State) -> int:
    """Compute the number of tokens from vf.State. Defined as the sum of prompt and completion tokens from the last trajectory step. If raw tokens are not available, falls back to checking the usage of the last response."""
    if not state["trajectory"]:
        return 0
    last_step = state["trajectory"][-1]
    if last_step["tokens"] is not None:
        return len(last_step["tokens"]["prompt_ids"]) + len(last_step["tokens"]["completion_ids"])
    last_step_response = cast(ChatCompletion, last_step["response"])
    return getattr(last_step_response.usage, "total_tokens", 0)


def get_completion_len(state: vf.State) -> int:
    """Compute the number of completion tokens from vf.State. Defined as the difference between the total number of tokens and the number of prompt tokens."""
    return get_seq_len(state) - get_prompt_len(state)


def get_is_truncated(state: vf.State) -> bool:
    """Check if the rollout is truncated. If raw tokens are not available, falls back to checking the finish reason of the last response."""
    if not state["trajectory"]:
        return False
    last_step = state["trajectory"][-1]
    if last_step["tokens"] is not None:
        return last_step["tokens"]["is_truncated"]
    last_step_response = cast(ChatCompletion, last_step["response"])
    return last_step_response.choices[0].finish_reason == "length"


def to_serializable_trajectory_step(step: vf.TrajectoryStep) -> dict:
    """Returns a serializable version of vf.TrajectoryStep."""
    serializable_trajectory_step = cast(dict, step.copy())
    if "response" in step and isinstance(step["response"], ChatCompletion):
        serializable_trajectory_step["response"] = step["response"].model_dump()
    return serializable_trajectory_step


def from_serializable_trajectory_step(step: dict) -> vf.TrajectoryStep:
    """Inverse of to_serializable_trajectory_step."""
    deserialized_trajectory_step = vf.TrajectoryStep(**step)
    if "response" in step and isinstance(step["response"], dict):
        deserialized_trajectory_step["response"] = ChatCompletion.model_validate(step["response"])
    return deserialized_trajectory_step


def to_serializable_state(state: vf.State) -> dict:
    """Returns a serializable copy of vf.State."""
    serializable_state = cast(dict, state.copy())

    # Flatten input fields to top level for serialization
    # This is necessary because the dict object cannot forward access to input fields anymore, e.g. state["prompt"] will automatically forward to state["input"]["prompt"] in vf.State but not in dict. We solve this by populating the inputs into the top-level explicitly
    if "input" in state:
        input_dict = serializable_state.pop("input")
        for field in vf.State.INPUT_FIELDS:
            if field in input_dict:
                serializable_state[field] = input_dict[field]

    if "trajectory" in state:
        serializable_state["trajectory"] = [to_serializable_trajectory_step(step) for step in state["trajectory"]]

    return serializable_state


def from_serializable_state(serializable_state: dict) -> vf.State:
    """Inverse of to_serializable_state."""
    # Extract input fields from top level and reconstruct input dict
    input_dict: dict[str, Any] = {}
    for field in vf.State.INPUT_FIELDS:
        if field in serializable_state:
            input_dict[field] = serializable_state.pop(field)

    if input_dict:
        serializable_state["input"] = input_dict

    state = vf.State(**serializable_state)

    if "trajectory" in state:
        state["trajectory"] = [from_serializable_trajectory_step(step) for step in state["trajectory"]]

    return state
