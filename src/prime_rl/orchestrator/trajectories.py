from typing import TypedDict

import verifiers as vf


class TrainableRollout(TypedDict):
    prompt_ids: list[int]
    prompt_mask: list[bool]
    completion_ids: list[int]
    completion_mask: list[int]
    completion_logprobs: list[float]
    advantage: float | None


def interleave_rollout(state: vf.State) -> list[TrainableRollout]:
    """
    Convert vf.State to a *single* trainable rollout by interleaving the trajectory.

    NOTE: This requires that consecutive trajectory steps share token prefixes (incremental tokenization)
    """

    # Initialize the rollout with prompt and completion from first trajectory step
    trajectory = state["trajectory"]
    first_step = trajectory[0]
    interleaved_rollout = TrainableRollout(
        prompt_ids=first_step["tokens"]["prompt_ids"],
        prompt_mask=first_step["tokens"]["prompt_mask"],
        completion_ids=first_step["tokens"]["completion_ids"],
        completion_mask=first_step["tokens"]["completion_mask"],
        completion_logprobs=first_step["tokens"]["completion_logprobs"],
        advantage=None,
    )

    # Interleave all other trajectory steps into completion
    prefix_tokens = first_step["tokens"]["prompt_ids"] + first_step["tokens"]["completion_ids"]
    prefix_messages = first_step["prompt"] + first_step["completion"]
    for step in trajectory[1:]:
        tokens = step["tokens"]
        assert tokens is not None
        prev_trajectory_and_new_prompt_ids = tokens["prompt_ids"]
        prev_trajectory_and_new_prompt = step["prompt"]
        # Incremental tokenization assumption
        assert prefix_tokens == prev_trajectory_and_new_prompt_ids[: len(prefix_tokens)]
        assert prefix_messages == prev_trajectory_and_new_prompt[: len(prefix_messages)]

        # Extend the completion with the new prompt
        prompt_ids = prev_trajectory_and_new_prompt_ids[len(prefix_tokens) :]
        prompt = prev_trajectory_and_new_prompt[len(prefix_messages) :]
        for msg in prompt:
            assert msg["role"] in ["user", "tool"]
        interleaved_rollout["completion_ids"].extend(prompt_ids)
        interleaved_rollout["completion_mask"].extend([0] * len(prompt_ids))
        interleaved_rollout["completion_logprobs"].extend([0.0] * len(prompt_ids))

        # Extend the completion with the new completion tokens
        interleaved_rollout["completion_ids"].extend(tokens["completion_ids"])
        interleaved_rollout["completion_mask"].extend([1] * len(tokens["completion_ids"]))
        interleaved_rollout["completion_logprobs"].extend(tokens["completion_logprobs"])

        # New prefix is the the current prompt and completion ids concatenated
        prefix_tokens = tokens["prompt_ids"] + tokens["completion_ids"]
        prefix_messages = step["prompt"] + step["completion"]

    return [interleaved_rollout]


def branch_rollout(state: vf.State) -> list[TrainableRollout]:
    """Convert vf.State to *multiple* trainable rollouts using branching trajectories strategy."""
    rollouts = []
    for step in state["trajectory"]:
        assert "tokens" in step
        tokens = step["tokens"]
        rollout = TrainableRollout(
            prompt_ids=tokens["prompt_ids"],
            prompt_mask=tokens["prompt_mask"],
            completion_ids=tokens["completion_ids"],
            completion_mask=tokens["completion_mask"],
            completion_logprobs=tokens["completion_logprobs"],
            advantage=None,
        )
        rollouts.append(rollout)
    return rollouts
