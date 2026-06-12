from __future__ import annotations

from typing import TYPE_CHECKING

from prime_rl.orchestrator.algo.advantage import assign_advantages
from prime_rl.orchestrator.algo.base import Algorithm

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import TrainRollout


class RewardAlgorithm(Algorithm):
    """REINFORCE-style: credit = raw reward, no group baseline; action tokens
    feed the ``rl`` loss."""

    def assign(self, rollouts: list[TrainRollout]) -> None:
        assign_advantages(rollouts, None)
