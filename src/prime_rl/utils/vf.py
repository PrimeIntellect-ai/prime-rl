import asyncio
from itertools import cycle
from typing import TypedDict

import verifiers as vf
from openai import AsyncOpenAI
from tqdm import tqdm

from prime_rl.orchestrator.utils import get_semaphore


async def generate_group(
    client: AsyncOpenAI,
    env: vf.Environment,
    model_name: str,
    example: dict,
    rollouts_per_example: int,
    sampling_args: dict,
) -> list[vf.State]:
    """Asynchronously generate and score rollouts for a single group."""
    semaphore = await get_semaphore()
    group_inputs = [vf.RolloutInput(**example) for _ in range(rollouts_per_example)]
    return await env.run_group(
        group_inputs=group_inputs,
        client=client,
        model=model_name,
        gen_sampling_args=sampling_args,
        gen_sem=semaphore,
        score_sem=semaphore,
    )


async def generate_batch(
    clients: list[AsyncOpenAI],
    env: vf.Environment,
    model_name: str,
    examples: list[dict],
    rollouts_per_example: int,
    sampling_args: dict,
    pbar_description: str = "Generating rollouts",
) -> list[vf.State]:
    """Asynchronously generate and score rollouts for a list of groups (batch)."""

    total_rollouts = len(examples) * rollouts_per_example
    pbar = tqdm(total=total_rollouts, desc=pbar_description)

    async def generate_group_with_progress(client, example):
        """Generate rollouts for one problem and update progress."""
        result = await generate_group(client, env, model_name, example, rollouts_per_example, sampling_args)
        pbar.update(rollouts_per_example)
        return result

    try:
        group_states_list: list[list[vf.State]] = await asyncio.gather(
            *[generate_group_with_progress(client, example) for client, example in zip(cycle(clients), examples)]
        )
    finally:
        pbar.close()

    return [state for group_states in group_states_list for state in group_states]


class Rollout(TypedDict):
    example_id: int
    task: str
    stop_condition: str | None
    reward: float
    advantage: float | None
    metrics: dict[str, float] | None
    trajectory_tokens: list[vf.TrajectoryStepTokens]


def make_interleaved_rollouts(states: list[vf.State]) -> list[Rollout]:
    """Convert vf.State to trainable rollout using interleaved trajectories."""
    rollouts = []
    for state in states:
        trajectory: vf.Trajectory = state["trajectory"]
        first_step = trajectory[0]
        first_step_tokens = first_step["tokens"]
        interleaved_trajectory: vf.TrajectoryStepTokens = {
            "prompt_ids": first_step_tokens["prompt_ids"],
            "prompt_mask": first_step_tokens["prompt_mask"],
            "completion_ids": first_step_tokens["completion_ids"],
            "completion_mask": first_step_tokens["completion_mask"],
            "completion_logprobs": first_step_tokens["completion_logprobs"],
            "overlong_prompt": trajectory[-1]["tokens"]["overlong_prompt"],
            "is_truncated": trajectory[-1]["tokens"]["is_truncated"],
        }
        prefix_tokens = first_step_tokens["prompt_ids"]
        for step in trajectory[1:]:
            tokens = step["tokens"]
            assert tokens is not None
            current_tokens = tokens["prompt_ids"] + tokens["completion_ids"]
            assert prefix_tokens == current_tokens[: len(prefix_tokens)]
            interleaved_trajectory["completion_ids"].extend(tokens["completion_ids"])
            interleaved_trajectory["completion_mask"].extend(tokens["completion_mask"])
            interleaved_trajectory["completion_logprobs"].extend(tokens["completion_logprobs"])
            prefix_tokens = tokens["prompt_ids"]
        rollout = Rollout(
            example_id=state["input"]["example_id"],
            task=state["input"]["task"],
            reward=state["reward"],
            advantage=state["advantage"],
            stop_condition=state["stop_condition"],
            metrics=state["metrics"],
            trajectory_tokens=[interleaved_trajectory],
        )
        rollouts.append(rollout)
    assert len(rollouts) == len(states)
    return rollouts


def make_branching_rollouts(states: list[vf.State]) -> list[Rollout]:
    """Convert vf.State to trainable rollout using branching trajectories."""
    rollouts = []
    for state in states:
        rollout = Rollout(
            example_id=state["input"]["example_id"],
            task=state["input"]["task"],
            reward=state["reward"],
            advantage=state["advantage"],
            stop_condition=state["stop_condition"],
            metrics=state["metrics"],
            trajectory_tokens=[step["tokens"] for step in state["trajectory"]],
        )
        rollouts.append(rollout)
    assert len(rollouts) == len(states)
    return rollouts


def get_completion_len(rollout: Rollout) -> int:
    """Get the length of the completion for a rollout."""
    completion_len = 0
    for step in rollout["trajectory_tokens"]:
        completion_len += len(step["completion_ids"])
    return completion_len


def get_prompt_len(rollout: Rollout) -> int:
    """Get the length of the prompt for a rollout."""
    return len(rollout["trajectory_tokens"][0]["prompt_ids"])


def get_seq_len(rollout: Rollout) -> int:
    """Get the length of the sequence for a rollout."""
    return get_prompt_len(rollout) + get_completion_len(rollout)
