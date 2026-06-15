from __future__ import annotations

from typing import TYPE_CHECKING

from prime_rl.orchestrator.algo.advantage import apply_advantage_fn, max_rl_advantage_fn
from prime_rl.orchestrator.algo.base import Algorithm

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import RolloutView


class MaxRLAlgorithm(Algorithm):
    """Maximum-likelihood RL (arXiv:2602.02710): the GRPO pipeline with
    mean-normalized advantages — ``(reward − group mean) / group mean``
    instead of plain centering. The mean normalization upweights low-pass-rate
    examples like the maximum-likelihood gradient does, and ``group_size``
    doubles as the truncation order of the likelihood expansion the gradient
    is unbiased for (REINFORCE at 1 → exact maximum likelihood as it grows)."""

    async def score_group(self, group: list[RolloutView]) -> None:
        apply_advantage_fn(group, max_rl_advantage_fn)
