from copy import deepcopy

import verifiers as vf

from prime_rl.transport import TrainingSample
from prime_rl.utils.logger import get_logger


def _normalize_completion_weights(
    raw_weights: list[float],
    completion_mask: list[bool],
    logger,
) -> list[float]:
    """Normalize raw weights to mean 1 over completion_mask == True tokens."""
    if not raw_weights or not completion_mask:
        return raw_weights
    if len(raw_weights) != len(completion_mask):
        logger.warning(
            f"Completion weights length mismatch ({len(raw_weights)} != {len(completion_mask)}); dropping weights."
        )
        return []
    masked_weights = [w for w, m in zip(raw_weights, completion_mask) if m]
    if not masked_weights:
        return [0.0 if not m else 1.0 for m in completion_mask]
    mean_weight = sum(masked_weights) / len(masked_weights)
    if mean_weight <= 0:
        return [0.0 if not m else 1.0 for m in completion_mask]
    return [0.0 if not m else (w / mean_weight) for w, m in zip(raw_weights, completion_mask)]


def _apply_weight_dampen(
    weights: list[float],
    completion_mask: list[bool],
    dampen: float,
    advantage: float | None,
) -> list[float]:
    if not weights or dampen == 1.0:
        return weights
    if len(weights) != len(completion_mask):
        return weights
    # Flip redistribution for negative advantages so "good" turns
    # are down-weighted (less negative) and "bad" turns are up-weighted.
    sign = -1.0 if (advantage or 0.0) < 0.0 else 1.0
    scaled: list[float] = []
    for w, m in zip(weights, completion_mask):
        if not m:
            scaled.append(0.0)
        else:
            scaled.append(1.0 + sign * dampen * (w - 1.0))
    return scaled


def _get_turn_scores(turn_scores: list[float] | None, num_steps: int, logger) -> list[float] | None:
    if turn_scores is None:
        return None
    if len(turn_scores) != num_steps:
        logger.warning(
            f"Turn score count ({len(turn_scores)}) does not match trajectory steps ({num_steps}); ignoring turn scores."
        )
        return None
    return turn_scores


def interleave_rollout(
    state: vf.State,
    turn_scores: list[float] | None = None,
    advantage: float | None = None,
    weight_dampen: float = 1.0,
) -> list[TrainingSample] | None:
    """
    Convert vf.State to a *single* trainable rollout by interleaving the trajectory.

    NOTE:
    - This requires that consecutive trajectory steps share token prefixes (incremental tokenization)
    - This approach is suceptible to introduce subtle difference due to re-tokenization in multi-turn environments.
    """
    logger = get_logger()

    trajectory = state["trajectory"]
    if len(trajectory) == 0:
        logger.warning(f"No trajectory steps for example {state['example_id']}. Skipping rollout.")
        return None

    turn_scores = _get_turn_scores(turn_scores, len(trajectory), logger)

    has_error = state["error"] is not None

    # Initialize the rollout with prompt and completion from first trajectory step
    first_step = trajectory[0]
    temperature = first_step["temperature"]
    if has_error:
        completion_mask = [False] * len(first_step["tokens"]["completion_mask"])
    else:
        completion_mask = [bool(i) for i in first_step["tokens"]["completion_mask"]]
    completion_ids = deepcopy(first_step["tokens"]["completion_ids"])
    interleaved_rollout = TrainingSample(
        prompt_ids=deepcopy(first_step["tokens"]["prompt_ids"]),
        prompt_mask=[bool(i) for i in first_step["tokens"]["prompt_mask"]],
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        completion_logprobs=deepcopy(first_step["tokens"]["completion_logprobs"]),
        completion_temperatures=[temperature] * len(completion_ids),  # Per-token temperatures
        teacher_logprobs=None,  # Populated at the end after full sequence length is known if teacher model is configured
        advantage_weights=None,
        advantage=None,
    )
    completion_weights: list[float] | None = None
    if turn_scores is not None:
        completion_weights = [1.0 + turn_scores[0]] * len(completion_ids)

    # Interleave all other trajectory steps into completion
    prefix_tokens = first_step["tokens"]["prompt_ids"] + first_step["tokens"]["completion_ids"]
    for step_idx, step in enumerate(trajectory[1:], start=2):
        tokens = step["tokens"]
        step_temperature = step["temperature"]
        assert tokens is not None
        prev_trajectory_and_new_prompt_ids = tokens["prompt_ids"]

        # Incremental tokenization assumption
        if not prefix_tokens == prev_trajectory_and_new_prompt_ids[: len(prefix_tokens)]:
            logger.warning(
                f"Found mismatch in prefix tokens for example {state['example_id']} at trajectory step {step_idx}"
            )

        # Extend the completion with the new prompt (use step's temperature for prompt tokens too)
        prompt_ids = deepcopy(prev_trajectory_and_new_prompt_ids[len(prefix_tokens) :])
        interleaved_rollout.completion_ids.extend(prompt_ids)
        interleaved_rollout.completion_mask.extend([False] * len(prompt_ids))
        interleaved_rollout.completion_logprobs.extend([0.0] * len(prompt_ids))
        interleaved_rollout.completion_temperatures.extend([step_temperature] * len(prompt_ids))
        if completion_weights is not None:
            completion_weights.extend([0.0] * len(prompt_ids))

        # Extend the completion with the new completion tokens
        completion_ids = deepcopy(tokens["completion_ids"])
        completion_logprobs = deepcopy(tokens["completion_logprobs"])
        interleaved_rollout.completion_ids.extend(completion_ids)
        if has_error:
            interleaved_rollout.completion_mask.extend([False] * len(tokens["completion_mask"]))
        else:
            interleaved_rollout.completion_mask.extend([bool(i) for i in tokens["completion_mask"]])
        interleaved_rollout.completion_logprobs.extend(completion_logprobs)
        interleaved_rollout.completion_temperatures.extend([step_temperature] * len(completion_ids))
        if completion_weights is not None:
            completion_weights.extend([1.0 + turn_scores[step_idx - 1]] * len(completion_ids))

        # New prefix is the current prompt and completion ids concatenated
        prefix_tokens = tokens["prompt_ids"] + tokens["completion_ids"]

    if completion_weights is not None:
        normalized = _normalize_completion_weights(completion_weights, interleaved_rollout.completion_mask, logger)
        if normalized and len(normalized) == len(interleaved_rollout.completion_ids):
            interleaved_rollout.advantage_weights = _apply_weight_dampen(
                normalized, interleaved_rollout.completion_mask, weight_dampen, advantage
            )

    return [interleaved_rollout]


def branch_rollout(
    state: vf.State,
    turn_scores: list[float] | None = None,
    advantage: float | None = None,
    weight_dampen: float = 1.0,
) -> list[TrainingSample] | None:
    """Convert vf.State to *multiple* trainable rollouts using branching trajectories strategy."""
    logger = get_logger()

    rollouts = []
    trajectory = state["trajectory"]
    if len(trajectory) == 0:
        logger.warning(f"No trajectory steps for example {state['example_id']}. Skipping rollout.")
        return None

    turn_scores = _get_turn_scores(turn_scores, len(trajectory), logger)

    has_error = state["error"] is not None
    for step_idx, step in enumerate(state["trajectory"]):
        assert "tokens" in step
        tokens = step["tokens"]
        temperature = step["temperature"]
        if has_error:
            completion_mask = [False] * len(tokens["completion_mask"])
        else:
            completion_mask = [bool(i) for i in tokens["completion_mask"]]
        completion_ids = deepcopy(tokens["completion_ids"])
        rollout = TrainingSample(
            prompt_ids=deepcopy(tokens["prompt_ids"]),
            prompt_mask=[bool(i) for i in tokens["prompt_mask"]],
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            completion_logprobs=deepcopy(tokens["completion_logprobs"]),
            completion_temperatures=[temperature] * len(completion_ids),  # Per-token temperatures
            advantage_weights=None,
            advantage=None,
            teacher_logprobs=None,
        )
        if turn_scores is not None:
            raw_weights = [1.0 + turn_scores[step_idx]] * len(completion_ids)
            normalized = _normalize_completion_weights(raw_weights, completion_mask, logger)
            if normalized and len(normalized) == len(completion_ids):
                rollout.advantage_weights = _apply_weight_dampen(normalized, completion_mask, weight_dampen, advantage)
        rollouts.append(rollout)
    return rollouts
