from __future__ import annotations

from typing import TYPE_CHECKING

from prime_rl.orchestrator.algo.advantage import apply_advantage_fn
from prime_rl.orchestrator.algo.base import Algorithm

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import TrainRollout


class RewardAlgorithm(Algorithm):
    """REINFORCE-style: credit = raw reward, no group baseline; action tokens
    feed the ``rl`` loss."""

    def assign_advantages(self, rollouts: list[TrainRollout]) -> None:
        apply_advantage_fn(rollouts, None)
