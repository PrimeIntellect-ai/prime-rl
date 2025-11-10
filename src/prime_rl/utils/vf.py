import asyncio
from itertools import cycle
from typing import TypedDict

import verifiers as vf
from openai import AsyncOpenAI
from verifiers.utils.async_utils import maybe_semaphore


class Rollout(TypedDict):
    """
    Prime-RL internal rollout object (1-1 with verifiers State), pared down to essentials
    and structured for downstream trainer/buffer usage.
    """

    example_id: int
    task: str
    reward: float
    metrics: dict[str, float]
    is_truncated: bool
    # Per-step training sequences and per-step advantages
    step_tokens: list[vf.TrajectoryStepTokens]
    step_advantages: list[float]


def state_to_rollout(state: vf.State) -> Rollout:
    """Convert a verifiers State to a minimal Rollout suitable for prime-rl internals."""
    example_id = int(state.get("example_id", 0))
    task = state.get("task", "default")
    reward = float(state.get("reward", 0.0))
    metrics = state.get("metrics", {}) or {}

    step_tokens: list[vf.TrajectoryStepTokens] = []
    step_advantages: list[float] = []
    is_truncated = False
    for step in state.get("trajectory", []):
        tokens = step.get("tokens")
        assert tokens is not None, "TrajectoryStep tokens must be present"
        advantage = step.get("advantage", None)
        assert advantage is not None, "TrajectoryStep advantage must be present"
        step_tokens.append(tokens)  # type: ignore[arg-type]
        step_advantages.append(float(advantage))
        # mark truncation if any step ended by length
        resp = step.get("response")
        try:
            if resp is not None and len(resp.choices) > 0 and getattr(resp.choices[0], "finish_reason", "") == "length":
                is_truncated = True
        except Exception:
            pass

    return Rollout(
        example_id=example_id,
        task=task,
        reward=reward,
        metrics=metrics,
        is_truncated=is_truncated,
        step_tokens=step_tokens,
        step_advantages=step_advantages,
    )


async def run_rollouts(
    clients: list[AsyncOpenAI],
    env: vf.Environment,
    model_name: str,
    problems: list[dict],
    rollouts_per_example: int,
    sampling_args: dict,
    max_concurrent: int | None,
    pbar_description: str = "Generating rollouts",
) -> list[vf.State]:
    """Asynchronously generate and score rollouts for a list of problems using run_group.

    Returns a flat list of State objects (one per rollout)."""
    from tqdm import tqdm

    pbar = tqdm(total=len(problems), desc=pbar_description)

    # Create distinct semaphores for generation and scoring (same limit by default)
    gen_sem = await maybe_semaphore(max_concurrent)
    score_sem = await maybe_semaphore(max_concurrent)

    async def generate_group_with_progress(client, problem):
        """Generate rollouts for one problem group and update progress."""
        group_inputs = [problem] * rollouts_per_example
        states = await env.run_group(  # type: ignore[attr-defined]
            group_inputs=group_inputs,
            client=client,
            model=model_name,
            gen_sampling_args=sampling_args,
            gen_sem=gen_sem,
            score_sem=score_sem,
        )
        pbar.update(1)
        return states

    try:
        # Schedule one group per problem, round-robin across clients
        all_group_states: list[list[vf.State]] = await asyncio.gather(
            *[generate_group_with_progress(client, problem) for client, problem in zip(cycle(clients), problems)]
        )
    finally:
        pbar.close()

    # Flatten to a single list of States
    return [state for group_states in all_group_states for state in group_states]
