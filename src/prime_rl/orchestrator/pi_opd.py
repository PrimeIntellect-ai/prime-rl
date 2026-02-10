"""Privileged-information on-policy distillation (PI-OPD).

Extracts tool calls from successful rollouts and uses them as context
for a same-model teacher to produce per-token advantage weights.
"""

import json
import math
import random

from prime_rl.utils.logger import get_logger


def extract_tool_calls(rollout: dict) -> str:
    """Extract tool-call names and arguments from a rollout's completion messages.

    Returns a formatted string of tool calls (no reasoning, no tool responses).
    """
    completion = rollout.get("completion")
    if not completion or not isinstance(completion, list):
        return ""

    blocks: list[str] = []
    for msg in completion:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls")
        if not tool_calls or not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function")
            if not isinstance(fn, dict):
                continue
            name = fn.get("name", "unknown")
            args = fn.get("arguments", "")
            if isinstance(args, str):
                # Pretty-print JSON arguments if possible
                try:
                    args = json.dumps(json.loads(args), indent=2)
                except (json.JSONDecodeError, TypeError):
                    pass
            elif isinstance(args, dict):
                args = json.dumps(args, indent=2)
            blocks.append(f"```\n{name}({args})\n```")

    return "\n\n".join(blocks)


def build_pi_prefix(tool_calls_text: str, template: str, tokenizer) -> list[int]:
    """Tokenize the PI prefix text."""
    prefix_text = template.format(tool_calls=tool_calls_text)
    return tokenizer.encode(prefix_text, add_special_tokens=False)


def select_donor_rollout(
    group: list[tuple[int, dict]],
    current_idx: int,
) -> dict | None:
    """Pick a random winning rollout from the group, excluding the current one.

    Args:
        group: list of (rollout_index, rollout_dict) for the same example_id.
        current_idx: index of the rollout we are building weights for.

    Returns:
        A donor rollout dict, or None if no suitable donor exists.
    """
    winners = [(i, r) for i, r in group if r["reward"] == 1.0 and i != current_idx]
    if not winners:
        return None
    return random.choice(winners)[1]


def compute_pi_advantage_weights(
    teacher_logprobs: list[float],
    student_logprobs: list[float],
    completion_mask: list[bool],
    dampen: float,
    advantage: float,
) -> list[float]:
    """Convert teacher/student logprob ratio into per-token advantage weights.

    Steps:
        1. ratio = exp(teacher_lp - student_lp) for each completion token
        2. Normalize so mean(ratio[mask]) == 1
        3. Dampen: 1 + dampen * (ratio - 1)
        4. Sign-flip for negative advantages
    """
    logger = get_logger()

    if len(teacher_logprobs) != len(student_logprobs):
        logger.warning(
            f"PI-OPD logprob length mismatch ({len(teacher_logprobs)} vs {len(student_logprobs)}); skipping."
        )
        return []
    if len(teacher_logprobs) != len(completion_mask):
        logger.warning(
            f"PI-OPD logprob/mask length mismatch ({len(teacher_logprobs)} vs {len(completion_mask)}); skipping."
        )
        return []

    # Compute per-token ratio (clamped to avoid extreme values)
    ratios: list[float] = []
    for t_lp, s_lp in zip(teacher_logprobs, student_logprobs):
        log_ratio = t_lp - s_lp
        log_ratio = max(-10.0, min(10.0, log_ratio))
        ratios.append(math.exp(log_ratio))

    # Normalize to mean 1 over masked tokens
    masked_ratios = [r for r, m in zip(ratios, completion_mask) if m]
    if not masked_ratios:
        return [0.0] * len(ratios)
    mean_ratio = sum(masked_ratios) / len(masked_ratios)
    if mean_ratio <= 0:
        return [0.0 if not m else 1.0 for m in completion_mask]
    normalized = [0.0 if not m else (r / mean_ratio) for r, m in zip(ratios, completion_mask)]

    # Dampen
    if dampen < 1.0:
        sign = -1.0 if (advantage or 0.0) < 0.0 else 1.0
        normalized = [0.0 if not m else (1.0 + sign * dampen * (w - 1.0)) for w, m in zip(normalized, completion_mask)]

    return normalized
