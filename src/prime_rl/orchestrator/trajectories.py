from collections import defaultdict
from copy import deepcopy

import verifiers as vf

from prime_rl.transport import TrainingSample
from prime_rl.utils.logger import get_logger


def _interleave_steps(steps: list[dict], has_error: bool, mismatch_label: str) -> TrainingSample | None:
    """Interleave a list of trajectory steps into a single training sample."""
    if not steps:
        return None

    logger = get_logger()

    first_step = steps[0]
    if has_error:
        completion_mask = [False] * len(first_step["tokens"]["completion_mask"])
    else:
        completion_mask = [bool(i) for i in first_step["tokens"]["completion_mask"]]

    rollout = TrainingSample(
        prompt_ids=deepcopy(first_step["tokens"]["prompt_ids"]),
        prompt_mask=[bool(i) for i in first_step["tokens"]["prompt_mask"]],
        completion_ids=deepcopy(first_step["tokens"]["completion_ids"]),
        completion_mask=completion_mask,
        completion_logprobs=deepcopy(first_step["tokens"]["completion_logprobs"]),
        teacher_logprobs=None,  # Populated at the end after full sequence length is known if teacher model is configured
        advantage=None,
    )

    prefix_tokens = first_step["tokens"]["prompt_ids"] + first_step["tokens"]["completion_ids"]
    for step_idx, step in enumerate(steps[1:], start=2):
        tokens = step["tokens"]
        assert tokens is not None
        prev_trajectory_and_new_prompt_ids = tokens["prompt_ids"]

        if not prefix_tokens == prev_trajectory_and_new_prompt_ids[: len(prefix_tokens)]:
            logger.warning(f"Found mismatch in prefix tokens for {mismatch_label} at step {step_idx}")

        prompt_ids = deepcopy(prev_trajectory_and_new_prompt_ids[len(prefix_tokens) :])
        rollout.completion_ids.extend(prompt_ids)
        rollout.completion_mask.extend([False] * len(prompt_ids))
        rollout.completion_logprobs.extend([0.0] * len(prompt_ids))

        completion_ids = deepcopy(tokens["completion_ids"])
        completion_logprobs = deepcopy(tokens["completion_logprobs"])
        rollout.completion_ids.extend(completion_ids)
        if has_error:
            rollout.completion_mask.extend([False] * len(tokens["completion_mask"]))
        else:
            rollout.completion_mask.extend([bool(i) for i in tokens["completion_mask"]])
        rollout.completion_logprobs.extend(completion_logprobs)

        prefix_tokens = tokens["prompt_ids"] + tokens["completion_ids"]

    return rollout


def interleave_rollout(state: vf.State) -> list[TrainingSample] | None:
    """
    Convert vf.State to trainable rollouts using interleaving by trajectory_id.

    Steps with the same trajectory_id are interleaved into one training sample.
    Different trajectory_ids become separate training samples.
    """
    logger = get_logger()

    trajectory = state["trajectory"]
    if len(trajectory) == 0:
        logger.warning(f"No trajectory steps for example {state['example_id']}. Skipping rollout.")
        return None

    has_error = state["error"] is not None

    # Group steps by trajectory_id, preserving order within groups
    groups: dict[str, list[dict]] = defaultdict(list)
    for step in trajectory:
        trajectory_id = step.get("trajectory_id", "main")
        groups[trajectory_id].append(step)

    # Interleave within each group
    rollouts: list[TrainingSample] = []
    for trajectory_id, steps in groups.items():
        rollout = _interleave_steps(steps, has_error, f"trajectory {trajectory_id}")
        if rollout:
            rollouts.append(rollout)

    return rollouts if rollouts else None


def branch_rollout(state: vf.State) -> list[TrainingSample] | None:
    """Convert vf.State to separate training samples for each trajectory step."""
    logger = get_logger()

    rollouts = []
    trajectory = state["trajectory"]
    if len(trajectory) == 0:
        logger.warning(f"No trajectory steps for example {state['example_id']}. Skipping rollout.")
        return None

    has_error = state["error"] is not None
    for step in trajectory:
        tokens = step["tokens"]
        if has_error:
            completion_mask = [False] * len(tokens["completion_mask"])
        else:
            completion_mask = [bool(i) for i in tokens["completion_mask"]]
        rollout = TrainingSample(
            prompt_ids=deepcopy(tokens["prompt_ids"]),
            prompt_mask=[bool(i) for i in tokens["prompt_mask"]],
            completion_ids=deepcopy(tokens["completion_ids"]),
            completion_mask=completion_mask,
            completion_logprobs=deepcopy(tokens["completion_logprobs"]),
            advantage=None,
            teacher_logprobs=None,
        )
        rollouts.append(rollout)
    return rollouts
