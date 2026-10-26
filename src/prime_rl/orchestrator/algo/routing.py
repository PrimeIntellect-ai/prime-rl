"""Training annotations and transport loss routing."""

from __future__ import annotations

import numpy as np
import verifiers.v1 as vf

from prime_rl.configs.algorithm import ActionLossType
from prime_rl.transports.batch import TrainingSample


def trainable_nodes(trace: vf.Trace) -> list[vf.MessageNode]:
    """Nodes with sampled (mask-True) tokens on the trace's trainable branches, in graph order."""
    on_trainable_branch = {id(node) for branch in trace.branches if branch.trainable for node in branch.nodes}
    return [node for node in trace.nodes if id(node) in on_trainable_branch and np.any(node.mask)]


def assign_advantages(trace: vf.Trace, values: float | list[float] | np.ndarray) -> None:
    """Assign credit in compact sampled-token order across trainable graph paths, as float32
    arrays (views into one array per trace)."""
    nodes = trainable_nodes(trace)
    for node in nodes:
        if len(node.mask) != len(node.token_ids):
            raise ValueError(
                f"node masks must align with token ids in trace {trace.id!r}: "
                f"got {len(node.mask)}, expected {len(node.token_ids)}"
            )
    counts = [int(np.count_nonzero(node.mask)) for node in nodes]
    total = sum(counts)
    if isinstance(values, (int, float, np.floating)):
        advantages = np.full(total, values, dtype=np.float32)
    else:
        if len(values) != total:
            raise ValueError(
                f"advantages must align with trace {trace.id!r}'s sampled tokens: got {len(values)}, expected {total}"
            )
        advantages = np.asarray(values, dtype=np.float32)

    offset = 0
    for node, count in zip(nodes, counts, strict=True):
        node.advantages = advantages[offset : offset + count]
        offset += count


def assign_reference_logprobs(branch: vf.Branch, values: list[float] | np.ndarray) -> None:
    """Project branch-aligned reference logprobs onto sampled graph nodes."""
    if len(values) != len(branch.token_ids):
        raise ValueError(
            f"reference logprobs must align with branch tokens: got {len(values)}, expected {len(branch.token_ids)}"
        )
    values = np.asarray(values, dtype=np.float32)
    offset = 0
    for node in branch.nodes:
        end = offset + len(node.token_ids)
        mask = np.asarray(node.mask, dtype=bool)
        if mask.any() and node.reference_logprobs is None:
            node.reference_logprobs = values[offset:end][mask]
        offset = end


def scalar_advantage(trace: vf.Trace) -> float | None:
    """Mean nonzero token advantage, or zero for an assigned-zero trace."""
    parts = [np.asarray(node.advantages, dtype=np.float32) for node in trace.nodes if node.advantages is not None]
    advantages = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    if not len(advantages):
        return None
    nonzero = advantages[advantages != 0.0]
    return float(nonzero.mean(dtype=np.float64)) if len(nonzero) else 0.0


def is_trainable(trace: vf.Trace) -> bool:
    """Whether any sampled token carries nonzero RL credit."""
    return any(np.any(np.asarray(node.advantages) != 0.0) for node in trace.nodes if node.advantages is not None)


def stamp_loss_routing(sample: TrainingSample, action_loss_type: ActionLossType) -> None:
    """Route action tokens into the algorithm's declared loss component."""
    if action_loss_type == "rl":
        return

    seq_len = len(sample.token_ids)
    sample.rl_weights = np.zeros(seq_len, dtype=np.float32)
    action_weights = (
        np.array(sample.ce_weights, dtype=np.float32)
        if action_loss_type == "ce" and sample.ce_weights is not None
        else np.zeros(seq_len, dtype=np.float32)
    )
    action_weights[np.asarray(sample.mask, dtype=bool)] = 1.0
    if action_loss_type == "ce":
        sample.ce_weights = action_weights
    else:
        assert action_loss_type == "ref_kl"
        sample.ref_kl_weights = action_weights
