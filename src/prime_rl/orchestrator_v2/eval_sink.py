"""EvalSink: buckets eval rollouts by ``(env_name, eval_step)`` and flushes
per-env epochs when the dispatcher signals completion via ``is_group_complete``.

Filters are train-only concepts; eval rollouts are not filtered here.

Single-purpose. ``add(rollout)`` returns an ``EvalFlush`` event when an env's
epoch is complete (or ``None`` otherwise); the orchestrator logs the per-env
metrics from that flush.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import verifiers as vf

from prime_rl.orchestrator_v2.dispatcher import Rollout


@dataclass
class EvalFlush:
    """One env's eval epoch is complete and ready to log."""

    env_name: str
    eval_step: int
    rollouts: list[vf.RolloutOutput]


class EvalSink:
    """Eval-side rollout sink. Group → flush per-env on ``is_group_complete``."""

    def __init__(self) -> None:
        # Per (env_name, eval_step) accumulation.
        self.pending: dict[tuple[str, int], list[Rollout]] = defaultdict(list)

    def add(self, rollout: Rollout) -> EvalFlush | None:
        """Add one rollout. Returns an ``EvalFlush`` when this rollout
        completes the env's epoch (``is_group_complete=True``)."""
        assert rollout.kind == "eval", "EvalSink only handles eval rollouts"
        assert rollout.eval_step is not None, "eval Rollout missing eval_step"
        key = (rollout.env_name, rollout.eval_step)
        self.pending[key].append(rollout)
        if not rollout.is_group_complete:
            return None
        group = self.pending.pop(key)
        return EvalFlush(
            env_name=rollout.env_name,
            eval_step=rollout.eval_step,
            rollouts=[r.raw for r in group],
        )
