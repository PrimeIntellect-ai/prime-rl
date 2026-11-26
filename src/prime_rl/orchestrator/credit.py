"""A trace's training credit and its optional rl loss weight.

Credit is one advantage for every sampled token on the trace's trainable branches, kept in
``trace.info``, or, for an algorithm that credits tokens unevenly, one advantage per
trainable token, kept on the sampled nodes (``node.advantages``, one value per sampled
token). A trace credited per token uses its token advantages on every branch."""

from __future__ import annotations

import numpy as np
import verifiers.v1 as vf

ADVANTAGE_KEY = "advantage"
"""``trace.info`` key of the trace's advantage."""

RL_WEIGHT_KEY = "rl_weight"
"""``trace.info`` key of the trace's rl loss weight; unset means 1.0."""


def token_advantages(trace: vf.Trace) -> np.ndarray | None:
    """The advantages of a trace credited per token, in graph order, or None for a trace
    credited with one advantage or none."""
    parts = [np.asarray(node.advantages, np.float32) for node in trace.nodes if node.advantages is not None]
    return np.concatenate(parts) if parts else None


def trace_advantage(trace: vf.Trace) -> float | None:
    """The trace's advantage, or None if it was assigned none. For a trace credited per token,
    the mean of its nonzero token advantages, 0.0 if all are zero."""
    values = token_advantages(trace)
    if values is None:
        return trace.info.get(ADVANTAGE_KEY)
    nonzero = values[values != 0.0]
    return float(nonzero.mean(dtype=np.float64)) if len(nonzero) else 0.0


def advantage_values(trace: vf.Trace) -> np.ndarray | None:
    """Every advantage the trace was assigned: its token advantages, or its one advantage."""
    values = token_advantages(trace)
    if values is not None:
        return values
    advantage = trace.info.get(ADVANTAGE_KEY)
    return None if advantage is None else np.array([advantage], np.float32)


def trace_rl_weight(trace: vf.Trace) -> float | None:
    """The trace's rl loss weight, or None if it was assigned none."""
    return trace.info.get(RL_WEIGHT_KEY)
