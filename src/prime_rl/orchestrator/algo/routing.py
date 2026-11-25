"""Training annotations and transport loss routing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import verifiers.v1 as vf

from prime_rl.configs.algorithm import ActionLossType
from prime_rl.orchestrator.credit import ADVANTAGE_KEY, RL_WEIGHT_KEY, advantage_values, token_advantages
from prime_rl.orchestrator.trajectories import iter_trainable_branches

if TYPE_CHECKING:
    from prime_rl.orchestrator.trajectories import BranchSample


def trainable_nodes(trace: vf.Trace) -> list[vf.MessageNode]:
    """Nodes with sampled (mask-True) tokens on the trace's trainable branches, in graph order."""
    on_trainable_branch = {id(node) for branch in trace.branches if branch.trainable for node in branch.nodes}
    return [node for node in trace.nodes if id(node) in on_trainable_branch and np.any(node.mask)]


def assign_advantage(trace: vf.Trace, value: float) -> None:
    """Credit every sampled token on the trace's trainable branches with ``value``. A trace
    with no such token gets no credit."""
    if trainable_nodes(trace):
        trace.info[ADVANTAGE_KEY] = float(value)


def token_layout(trace: vf.Trace) -> dict[int, list[int]]:
    """The trainable tokens of each of the trace's training samples, by branch index: one
    count per sampled node of the branch that trains in it, in order. A sampled node shared
    by several branches trains only in the first."""
    layout: dict[int, list[int]] = {}
    for branch, mask in iter_trainable_branches(trace):
        counts, offset = [], 0
        for node in branch.nodes:
            count = int(np.count_nonzero(mask[offset : offset + len(node.mask)]))
            offset += len(node.mask)
            if count:
                counts.append(count)
        layout[branch.index] = counts
    return layout


def assign_advantages(trace: vf.Trace, branch: int, values: list[float] | np.ndarray) -> None:
    """Credit each trainable token of one of the trace's branches with its own advantage:
    ``values`` holds one per token, ``sum(token_layout(trace)[branch])`` in order. A trace
    credited per token uses token advantages on every branch; a branch assigned none gets 0.0."""
    for candidate, mask in iter_trainable_branches(trace):
        if candidate.index == branch:
            break
    else:
        raise ValueError(f"trace {trace.id!r} has no training sample for branch {branch}")
    values = np.asarray(values, dtype=np.float32)
    if len(values) != int(np.count_nonzero(mask)):
        raise ValueError(
            f"advantages must align with the trainable tokens of branch {branch} of trace {trace.id!r}: "
            f"got {len(values)}, expected {int(np.count_nonzero(mask))}"
        )
    offset, taken = 0, 0
    for node in candidate.nodes:
        count = int(np.count_nonzero(mask[offset : offset + len(node.mask)]))
        offset += len(node.mask)
        if count:
            node.advantages = values[taken : taken + count]
            taken += count


def branch_token_advantages(trace: vf.Trace) -> dict[int, np.ndarray] | None:
    """The advantages of a trace credited per token, by branch index: one per trainable token
    of each training sample, in order (0.0 on a node assigned none); None for a trace credited
    with one advantage or none."""
    if token_advantages(trace) is None:
        return None
    values: dict[int, np.ndarray] = {}
    for branch, mask in iter_trainable_branches(trace):
        parts, offset = [], 0
        for node in branch.nodes:
            count = int(np.count_nonzero(mask[offset : offset + len(node.mask)]))
            offset += len(node.mask)
            if count:
                part = np.zeros(count, np.float32)
                assigned = np.asarray(node.advantages if node.advantages is not None else [], np.float32)[:count]
                part[: len(assigned)] = assigned
                parts.append(part)
        values[branch.index] = np.concatenate(parts) if parts else np.zeros(0, np.float32)
    return values


def assign_rl_weight(trace: vf.Trace, value: float) -> None:
    """Weight the rl loss of every sampled token on the trace's trainable branches by ``value``."""
    trace.info[RL_WEIGHT_KEY] = float(value)


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


def is_trainable(trace: vf.Trace) -> bool:
    """Whether the trace's sampled tokens carry nonzero RL credit."""
    values = advantage_values(trace)
    return values is not None and bool(np.any(values != 0.0))


def stamp_loss_routing(sample: BranchSample, action_loss_type: ActionLossType) -> None:
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
