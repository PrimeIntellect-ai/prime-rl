import verifiers as vf

from prime_rl.transport import TrainingSample
from prime_rl.utils.logger import get_logger


def interleave_rollout(state: vf.State) -> list[TrainingSample] | None:
    """
    Convert vf.State to trainable rollouts by interleaving trajectory steps
    where the extension property holds.

    When consecutive steps share token prefixes (extension property), they are
    merged into a single sample for O(T) compute. When extension breaks (e.g.,
    due to context compaction or re-rendering), a new sample is started.

    Returns a list of samples - could be 1 (extension always held) or up to T
    (extension never held).
    """
    logger = get_logger()

    trajectory = state["trajectory"]
    if len(trajectory) == 0:
        logger.warning(f"No trajectory steps for example {state['example_id']}. Skipping rollout.")
        return None

    has_error = state["error"] is not None

    def make_sample(step: dict) -> TrainingSample:
        """Create a new TrainingSample from a trajectory step."""
        tokens = step["tokens"]
        if has_error:
            completion_mask = [False] * len(tokens["completion_mask"])
        else:
            completion_mask = list(tokens["completion_mask"])
        return TrainingSample(
            prompt_ids=list(tokens["prompt_ids"]),
            prompt_mask=list(tokens["prompt_mask"]),
            completion_ids=list(tokens["completion_ids"]),
            completion_mask=completion_mask,
            completion_logprobs=list(tokens["completion_logprobs"]),
            teacher_logprobs=None,
            advantage=None,
        )

    def extend_sample(sample: TrainingSample, step: dict, prefix_len: int) -> None:
        """Extend an existing sample with a new trajectory step (extension property holds)."""
        tokens = step["tokens"]

        # Extend with new prompt tokens (mask=False, no gradient)
        new_prompt_ids = tokens["prompt_ids"][prefix_len:]
        sample.completion_ids.extend(new_prompt_ids)
        sample.completion_mask.extend([False] * len(new_prompt_ids))
        sample.completion_logprobs.extend([0.0] * len(new_prompt_ids))

        # Extend with new completion tokens
        sample.completion_ids.extend(tokens["completion_ids"])
        if has_error:
            sample.completion_mask.extend([False] * len(tokens["completion_mask"]))
        else:
            sample.completion_mask.extend(tokens["completion_mask"])
        sample.completion_logprobs.extend(tokens["completion_logprobs"])

    # Start with first trajectory step
    samples: list[TrainingSample] = []
    current_sample = make_sample(trajectory[0])
    prefix_tokens = trajectory[0]["tokens"]["prompt_ids"] + trajectory[0]["tokens"]["completion_ids"]

    for step_idx, step in enumerate(trajectory[1:], start=2):
        tokens = step["tokens"]
        step_prompt_ids = tokens["prompt_ids"]

        # Check extension property: does new prompt start with our accumulated prefix?
        if step_prompt_ids[: len(prefix_tokens)] == prefix_tokens:
            # Extension holds - merge into current sample
            extend_sample(current_sample, step, len(prefix_tokens))
        else:
            # Extension breaks - finalize current sample and start fresh
            logger.debug(
                f"Extension property broke at step {step_idx} for example {state['example_id']}. "
                f"Starting new sample (prefix_len={len(prefix_tokens)}, step_prompt_len={len(step_prompt_ids)})."
            )
            samples.append(current_sample)
            current_sample = make_sample(step)

        # Update prefix for next iteration
        prefix_tokens = tokens["prompt_ids"] + tokens["completion_ids"]

    # Finalize the last sample
    samples.append(current_sample)

    return samples
