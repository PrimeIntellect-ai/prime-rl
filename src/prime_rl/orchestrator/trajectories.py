from copy import deepcopy

import verifiers as vf

from prime_rl.orchestrator.types import TrainingExample
from prime_rl.utils.logger import get_logger


def interleave_rollout(state: vf.State) -> list[TrainingExample]:
    """
    Convert vf.State to a *single* trainable rollout by interleaving the trajectory.

    NOTE: This requires that consecutive trajectory steps share token prefixes (incremental tokenization)
    """
    logger = get_logger()
    trajectory = state["trajectory"]

    if len(trajectory) == 1:
        first_step = trajectory[0]
        return [
            TrainingExample(
                prompt_ids=deepcopy(first_step["tokens"]["prompt_ids"]),
                prompt_mask=deepcopy(first_step["tokens"]["prompt_mask"]),
                completion_ids=deepcopy(first_step["tokens"]["completion_ids"]),
                completion_mask=deepcopy(first_step["tokens"]["completion_mask"]),
                completion_logprobs=deepcopy(first_step["tokens"]["completion_logprobs"]),
                advantage=None,
            )
        ]

    first_step = trajectory[0]
    final_step = trajectory[-1]
    final_tokens = final_step["tokens"]
    assert final_tokens is not None

    final_prompt_ids = final_tokens["prompt_ids"]
    final_prompt_logprobs = final_tokens.get("prompt_logprobs")
    final_completion_ids = final_tokens["completion_ids"]
    final_completion_logprobs = final_tokens["completion_logprobs"]
    first_prompt_len = len(first_step["tokens"]["prompt_ids"])

    if final_prompt_logprobs is None:
        raise ValueError(
            f"prompt_logprobs not available for multi-turn example {state.get('example_id', 'unknown')}. "
            f"Ensure vLLM is configured with prompt_logprobs=True."
        )

    completion_ids = deepcopy(final_prompt_ids[first_prompt_len:]) + deepcopy(final_completion_ids)
    completion_logprobs = deepcopy(final_prompt_logprobs[first_prompt_len:]) + deepcopy(final_completion_logprobs)
    completion_mask = _build_completion_mask_from_trajectory(
        trajectory, first_prompt_len, final_prompt_ids, final_completion_ids
    )

    if len(completion_ids) != len(completion_logprobs):
        logger.warning(
            f"Length mismatch in example {state.get('example_id', 'unknown')}: "
            f"completion_ids={len(completion_ids)}, completion_logprobs={len(completion_logprobs)}"
        )
        min_len = min(len(completion_ids), len(completion_logprobs))
        completion_ids = completion_ids[:min_len]
        completion_logprobs = completion_logprobs[:min_len]
        completion_mask = completion_mask[:min_len]

    return [
        TrainingExample(
            prompt_ids=deepcopy(final_prompt_ids[:first_prompt_len]),
            prompt_mask=[0] * first_prompt_len,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            completion_logprobs=completion_logprobs,
            advantage=None,
        )
    ]


def _build_completion_mask_from_trajectory(
    trajectory: list[vf.TrajectoryStep],
    first_prompt_len: int,
    final_prompt_ids: list[int],
    final_completion_ids: list[int],
) -> list[int]:
    """Build mask: 1 for model completions, 0 for user/system/tool tokens."""
    total_completion_len = len(final_prompt_ids) - first_prompt_len + len(final_completion_ids)
    mask = [0] * total_completion_len
    current_pos = 0

    for step_idx, step in enumerate(trajectory):
        tokens = step["tokens"]
        assert tokens is not None

        if step_idx == 0:
            completion_len = len(tokens["completion_ids"])
            for i in range(completion_len):
                if current_pos + i < len(mask):
                    mask[current_pos + i] = 1
            current_pos += completion_len
        else:
            prev_total_len = len(trajectory[step_idx - 1]["tokens"]["prompt_ids"]) + len(
                trajectory[step_idx - 1]["tokens"]["completion_ids"]
            )
            new_prompt_len = len(tokens["prompt_ids"]) - prev_total_len
            current_pos += new_prompt_len

            completion_len = len(tokens["completion_ids"])
            for i in range(completion_len):
                if current_pos + i < len(mask):
                    mask[current_pos + i] = 1
            current_pos += completion_len

    return mask


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
