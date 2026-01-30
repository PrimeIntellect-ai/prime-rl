from typing import TypedDict

import verifiers as vf

from prime_rl.transport import TrainingSample
from prime_rl.utils.logger import get_logger


class TrajectoryStepWithTemp(TypedDict):
    """Trajectory step with temperature field added by prime-rl's extract_result."""

    tokens: vf.TrajectoryStepTokens
    temperature: float


def interleave_rollout(state: vf.State) -> list[TrainingSample] | None:
    """
    Convert vf.State to trainable rollouts by interleaving trajectory steps
    where the extension property holds.

    When consecutive steps share token prefixes (extension property), they are
    merged into a single sample. When extension breaks (e.g., due to context
    compaction or a change in control-flow), a new sample is started.

    Supports multi-prefix matching to handle interleaved agents. For example,
    [agent1-step1, agent1-step2, agent2-step1, agent1-step3] produces two samples:
    agent1 steps merged together, agent2 step separate.

    Returns a list of samples - could be 1 (extension always held) or up to T
    (extension never held).
    """
    logger = get_logger()

    trajectory = state["trajectory"]
    if len(trajectory) == 0:
        logger.warning(f"No trajectory steps for example {state['example_id']}. Skipping rollout.")
        return None

    has_error = state["error"] is not None

    def make_sample(step: TrajectoryStepWithTemp) -> TrainingSample:
        """Create a new TrainingSample from a trajectory step."""
        tokens = step["tokens"]
        temperature = step["temperature"]
        if has_error:
            completion_mask = [False] * len(tokens["completion_mask"])
        else:
            completion_mask = [bool(i) for i in tokens["completion_mask"]]
        completion_ids = list(tokens["completion_ids"])
        return TrainingSample(
            prompt_ids=list(tokens["prompt_ids"]),
            prompt_mask=[bool(i) for i in tokens["prompt_mask"]],
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            completion_logprobs=list(tokens["completion_logprobs"]),
            completion_temperatures=[temperature] * len(completion_ids),
            teacher_logprobs=None,
            advantage=None,
        )

    def extend_sample(sample: TrainingSample, step: TrajectoryStepWithTemp, prefix_len: int) -> None:
        """Extend an existing sample with a new trajectory step (extension property holds)."""
        tokens = step["tokens"]
        temperature = step["temperature"]

        # Extend with new prompt tokens (mask=False, no gradient)
        new_prompt_ids = tokens["prompt_ids"][prefix_len:]
        sample.completion_ids.extend(new_prompt_ids)
        sample.completion_mask.extend([False] * len(new_prompt_ids))
        sample.completion_logprobs.extend([0.0] * len(new_prompt_ids))
        sample.completion_temperatures.extend([temperature] * len(new_prompt_ids))

        # Extend with new completion tokens
        completion_ids = tokens["completion_ids"]
        sample.completion_ids.extend(completion_ids)
        if has_error:
            sample.completion_mask.extend([False] * len(tokens["completion_mask"]))
        else:
            sample.completion_mask.extend(bool(i) for i in tokens["completion_mask"])
        sample.completion_logprobs.extend(tokens["completion_logprobs"])
        sample.completion_temperatures.extend([temperature] * len(completion_ids))

    # Track multiple active (prefix, sample) pairs to handle interleaved agents
    # Each entry is [prefix_tokens, sample] where prefix_tokens is the accumulated token sequence
    active_samples: list[list] = []

    first_tokens = trajectory[0]["tokens"]
    first_prefix = first_tokens["prompt_ids"] + first_tokens["completion_ids"]
    active_samples.append([first_prefix, make_sample(trajectory[0])])

    for step_idx, step in enumerate(trajectory[1:], start=2):
        tokens = step["tokens"]
        step_prompt_ids = tokens["prompt_ids"]

        # Check if this step extends ANY active prefix
        matched_idx = None
        for idx, (prefix_tokens, _) in enumerate(active_samples):
            if step_prompt_ids[: len(prefix_tokens)] == prefix_tokens:
                matched_idx = idx
                break

        if matched_idx is not None:
            # Extension holds - merge into matched sample
            prefix_tokens, sample = active_samples[matched_idx]
            extend_sample(sample, step, len(prefix_tokens))
            # Update prefix for this sample
            active_samples[matched_idx][0] = tokens["prompt_ids"] + tokens["completion_ids"]
        else:
            # No prefix matches - start a new sample
            logger.debug(
                f"Extension property broke at step {step_idx} for example {state['example_id']}. "
                f"Starting new sample (active_prefixes={len(active_samples)}, step_prompt_len={len(step_prompt_ids)})."
            )
            new_prefix = tokens["prompt_ids"] + tokens["completion_ids"]
            active_samples.append([new_prefix, make_sample(step)])

    # Return all samples
    return [sample for _, sample in active_samples]
