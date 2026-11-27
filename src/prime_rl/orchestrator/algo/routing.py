"""Training annotations and transport loss routing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import verifiers.v1 as vf

from prime_rl.configs.algorithm import ActionLossType
from prime_rl.orchestrator.credit import ADVANTAGE_KEY, RL_WEIGHT_KEY, advantage_values

if TYPE_CHECKING:
    from prime_rl.env.heads import TraceHead
    from prime_rl.orchestrator.trajectories import BranchSample


def trainable_nodes(trace: vf.Trace) -> list[vf.MessageNode]:
    """Nodes with sampled (mask-True) tokens on the trace's trainable branches, in graph order."""
    on_trainable_branch = {id(node) for branch in trace.branches if branch.trainable for node in branch.nodes}
    return [node for node in trace.nodes if id(node) in on_trainable_branch and np.any(node.mask)]


def assign_advantage(trace: TraceHead, value: float) -> None:
    """Credit every sampled token on the trace's trainable branches with ``value``. A trace
    with no such token gets no credit."""
    if trace.trainable_tokens:
        trace.info[ADVANTAGE_KEY] = float(value)


def token_layout(trace: TraceHead) -> dict[int, list[int]]:
    """The trainable tokens of each of the trace's training samples, by branch index: one
    count per sampled node of the branch that trains in it, in order. A sampled node shared
    by several branches trains only in the first."""
    return trace.token_layout


def assign_advantages(trace: TraceHead, branch: int, values: list[float] | np.ndarray) -> None:
    """Credit each trainable token of one of the trace's branches with its own advantage:
    ``values`` holds one per token, ``sum(token_layout(trace)[branch])`` in order. A trace
    credited per token uses token advantages on every branch; a branch assigned none gets 0.0."""
    if branch not in trace.token_layout:
        raise ValueError(f"trace {trace.id!r} has no training sample for branch {branch}")
    values = np.asarray(values, dtype=np.float32)
    expected = sum(trace.token_layout[branch])
    if len(values) != expected:
        raise ValueError(
            f"advantages must align with the trainable tokens of branch {branch} of trace {trace.id!r}: "
            f"got {len(values)}, expected {expected}"
        )
    trace.branch_advantages[branch] = values


def assign_rl_weight(trace: TraceHead, value: float) -> None:
    """Weight the rl loss of every sampled token on the trace's trainable branches by ``value``."""
    trace.info[RL_WEIGHT_KEY] = float(value)


def is_trainable(trace: TraceHead) -> bool:
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
