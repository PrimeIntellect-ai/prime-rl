"""A trace's training credit: one advantage and an optional rl loss weight for every
sampled token on its trainable branches, kept in its head's ``info``."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prime_rl.env.heads import TraceHead

ADVANTAGE_KEY = "advantage"
"""``trace.info`` key of the trace's advantage."""

RL_WEIGHT_KEY = "rl_weight"
"""``trace.info`` key of the trace's rl loss weight; unset means 1.0."""


def trace_advantage(trace: TraceHead) -> float | None:
    """The trace's advantage, or None if it was assigned none."""
    return trace.info.get(ADVANTAGE_KEY)


def trace_rl_weight(trace: TraceHead) -> float | None:
    """The trace's rl loss weight, or None if it was assigned none."""
    return trace.info.get(RL_WEIGHT_KEY)
