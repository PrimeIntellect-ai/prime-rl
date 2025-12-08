from copy import deepcopy

import verifiers as vf

from prime_rl.orchestrator.types import TrainingExample
from prime_rl.utils.logger import get_logger


def interleave_rollout(state: vf.State) -> list[TrainingExample]:
    """
    Convert vf.State to a *single* trainable rollout by interleaving the trajectory.

    NOTE:
    - This requires that consecutive trajectory steps share token prefixes (incremental tokenization)
    - This approach is suceptible to introduce subtle difference due to re-tokenization in multi-turn environments.
    """
    logger = get_logger()

    # Initialize the rollout with prompt and completion from first trajectory step
    trajectory = state["trajectory"]
    first_step = trajectory[0]
    interleaved_rollout = TrainingExample(
        prompt_ids=deepcopy(first_step["tokens"]["prompt_ids"]),
        prompt_mask=deepcopy(first_step["tokens"]["prompt_mask"]),
        completion_ids=deepcopy(first_step["tokens"]["completion_ids"]),
        completion_mask=deepcopy(first_step["tokens"]["completion_mask"]),
        completion_logprobs=deepcopy(first_step["tokens"]["completion_logprobs"]),
        advantage=None,
    )

    # Interleave all other trajectory steps into completion
    prefix_tokens = first_step["tokens"]["prompt_ids"] + first_step["tokens"]["completion_ids"]
    for step_idx, step in enumerate(trajectory[1:], start=2):
        tokens = step["tokens"]
        assert tokens is not None
        prev_trajectory_and_new_prompt_ids = tokens["prompt_ids"]

        # Incremental tokenization assumption
        if not prefix_tokens == prev_trajectory_and_new_prompt_ids[: len(prefix_tokens)]:
            logger.warning(
                f"Found mismatch in prefix tokens for example {state['example_id']} at trajectory step {step_idx}"
            )

        # Extend the completion with the new prompt
        prompt_ids = deepcopy(prev_trajectory_and_new_prompt_ids[len(prefix_tokens) :])
        interleaved_rollout["completion_ids"].extend(prompt_ids)
        interleaved_rollout["completion_mask"].extend([0] * len(prompt_ids))
        interleaved_rollout["completion_logprobs"].extend([0.0] * len(prompt_ids))

        # Extend the completion with the new completion tokens
        completion_ids = deepcopy(tokens["completion_ids"])
        completion_logprobs = deepcopy(tokens["completion_logprobs"])
        interleaved_rollout["completion_ids"].extend(completion_ids)
        interleaved_rollout["completion_mask"].extend([1] * len(completion_ids))
        interleaved_rollout["completion_logprobs"].extend(completion_logprobs)

        # New prefix is the the current prompt and completion ids concatenated
        prefix_tokens = tokens["prompt_ids"] + tokens["completion_ids"]

    return [interleaved_rollout]


def branch_rollout(state: vf.State) -> list[TrainingExample]:
    """Convert vf.State to *multiple* trainable rollouts using branching trajectories strategy."""
    rollouts = []
    for step in state["trajectory"]:
        assert "tokens" in step
        tokens = step["tokens"]
        rollout = TrainingExample(
            prompt_ids=deepcopy(tokens["prompt_ids"]),
            prompt_mask=deepcopy(tokens["prompt_mask"]),
            completion_ids=deepcopy(tokens["completion_ids"]),
            completion_mask=deepcopy(tokens["completion_mask"]),
            completion_logprobs=deepcopy(tokens["completion_logprobs"]),
            advantage=None,
        )
        rollouts.append(rollout)
    return rollouts
