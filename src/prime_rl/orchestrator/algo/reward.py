from __future__ import annotations

from typing import TYPE_CHECKING

from prime_rl.orchestrator.algo.base import Algorithm

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import RolloutView


class RewardAlgorithm(Algorithm):
    """REINFORCE-style: credit = raw reward, no group baseline. Purely
    rollout-local — no siblings needed — so it scores on arrival; action
    tokens feed the ``rl`` loss."""

    async def score_rollout(self, rollout: RolloutView) -> None:
        rollout.assign_advantages(rollout.reward)
