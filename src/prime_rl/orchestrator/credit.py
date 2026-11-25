"""A trace's training credit and its optional rl loss weight.

Credit is one advantage for every sampled token on the trace's trainable branches, kept in
its head's ``info``, or, for an algorithm that credits tokens unevenly, one advantage per
trainable token of each training sample, kept on its head by branch index
(``TraceHead.branch_advantages``). A trace credited per token uses token advantages on every
branch."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from prime_rl.env.heads import TraceHead

ADVANTAGE_KEY = "advantage"
"""``trace.info`` key of the trace's advantage."""

RL_WEIGHT_KEY = "rl_weight"
"""``trace.info`` key of the trace's rl loss weight; unset means 1.0."""


def branch_token_advantages(trace: TraceHead) -> dict[int, np.ndarray] | None:
    """The advantages of a trace credited per token, by branch index: one per trainable token
    of each training sample, in order (0.0 on a branch assigned none); None for a trace
    credited with one advantage or none."""
    if not trace.branch_advantages:
        return None
    return {
        branch: trace.branch_advantages.get(branch, np.zeros(sum(counts), np.float32))
        for branch, counts in trace.token_layout.items()
    }


def token_advantages(trace: TraceHead) -> np.ndarray | None:
    """The advantages of a trace credited per token, branch by branch, or None for a trace
    credited with one advantage or none."""
    values = branch_token_advantages(trace)
    return None if values is None else np.concatenate([np.zeros(0, np.float32), *values.values()])


def trace_advantage(trace: TraceHead) -> float | None:
    """The trace's advantage, or None if it was assigned none. For a trace credited per token,
    the mean of its nonzero token advantages, 0.0 if all are zero."""
    values = token_advantages(trace)
    if values is None:
        return trace.info.get(ADVANTAGE_KEY)
    nonzero = values[values != 0.0]
    return float(nonzero.mean(dtype=np.float64)) if len(nonzero) else 0.0


def advantage_values(trace: TraceHead) -> np.ndarray | None:
    """Every advantage the trace was assigned: its token advantages, or its one advantage."""
    values = token_advantages(trace)
    if values is not None:
        return values
    advantage = trace.info.get(ADVANTAGE_KEY)
    return None if advantage is None else np.array([advantage], np.float32)


def trace_rl_weight(trace: TraceHead) -> float | None:
    """The trace's rl loss weight, or None if it was assigned none."""
    return trace.info.get(RL_WEIGHT_KEY)
