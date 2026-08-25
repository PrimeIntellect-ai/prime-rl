"""Training annotations and transport loss routing."""

from __future__ import annotations

import verifiers.v1 as vf

from prime_rl.configs.algorithm import ActionLossType
from prime_rl.transports.batch import TrainingSample


def assign_advantages(trace: vf.Trace, values: float | list[float]) -> None:
    """Assign credit in compact sampled-token order across the trace graph."""
    nodes = [node for node in trace.nodes if any(node.mask)]
    for node in nodes:
        if len(node.mask) != len(node.token_ids):
            raise ValueError(
                f"node masks must align with token ids in trace {trace.id!r}: "
                f"got {len(node.mask)}, expected {len(node.token_ids)}"
            )
    total = sum(sum(node.mask) for node in nodes)
    if isinstance(values, (int, float)):
        advantages = [float(values)] * total
    else:
        if len(values) != total:
            raise ValueError(
                f"advantages must align with trace {trace.id!r}'s sampled tokens: got {len(values)}, expected {total}"
            )
        advantages = [float(value) for value in values]

    offset = 0
    for node in nodes:
        end = offset + sum(node.mask)
        node.advantages = advantages[offset:end]
        offset = end


def assign_reference_logprobs(branch: vf.Branch, values: list[float]) -> None:
    """Project branch-aligned reference logprobs onto sampled graph nodes."""
    if len(values) != len(branch.token_ids):
        raise ValueError(
            f"reference logprobs must align with branch tokens: got {len(values)}, expected {len(branch.token_ids)}"
        )
    offset = 0
    for node in branch.nodes:
        end = offset + len(node.token_ids)
        if any(node.mask) and node.reference_logprobs is None:
            node.reference_logprobs = [
                float(value) for value, sampled in zip(values[offset:end], node.mask, strict=True) if sampled
            ]
        offset = end


def scalar_advantage(trace: vf.Trace) -> float | None:
    """Mean nonzero token advantage, or zero for an assigned-zero trace."""
    advantages = [value for node in trace.nodes for value in node.advantages or []]
    if not advantages:
        return None
    nonzero = [value for value in advantages if value != 0.0]
    return sum(nonzero) / len(nonzero) if nonzero else 0.0


def is_trainable(trace: vf.Trace) -> bool:
    """Whether any sampled token carries nonzero RL credit."""
    return any(value != 0.0 for node in trace.nodes for value in node.advantages or [])


def stamp_loss_routing(sample: TrainingSample, action_loss_type: ActionLossType) -> None:
    """Route action tokens into the algorithm's declared loss component."""
    if action_loss_type == "rl":
        return

    seq_len = len(sample.token_ids)
    sample.rl_weights = [0.0] * seq_len
    action_weights = (
        sample.ce_weights if action_loss_type == "ce" and sample.ce_weights is not None else [0.0] * seq_len
    )
    for i, trains in enumerate(sample.mask):
        if trains:
            action_weights[i] = 1.0
    if action_loss_type == "ce":
        sample.ce_weights = action_weights
    else:
        assert action_loss_type == "ref_kl"
        sample.ref_kl_weights = action_weights
