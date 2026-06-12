from __future__ import annotations

from typing import TYPE_CHECKING

from prime_rl.orchestrator.algo.advantage import assign_advantages, max_rl_advantage_fn
from prime_rl.orchestrator.algo.base import Algorithm

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import TrainRollout


class MaxRLAlgorithm(Algorithm):
    """Maximum-likelihood RL (arXiv:2602.02710): the GRPO pipeline with
    mean-normalized advantages — ``(reward − group mean) / group mean``
    instead of plain centering. The mean normalization upweights low-pass-rate
    examples like the maximum-likelihood gradient does, and ``group_size``
    doubles as the truncation order of the likelihood expansion the gradient
    is unbiased for (REINFORCE at 1 → exact maximum likelihood as it grows)."""

    def assign(self, rollouts: list[TrainRollout]) -> None:
        assign_advantages(rollouts, max_rl_advantage_fn)
