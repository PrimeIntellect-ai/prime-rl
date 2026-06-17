from __future__ import annotations

from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import (
    AdvantageConfig,
    GRPOAdvantageConfig,
    LinearLengthPenaltyConfig,
    TokensLengthPenaltyConfig,
    TurnsLengthPenaltyConfig,
)
from prime_rl.orchestrator.algo.advantage import (
    efficiency_shaping_advantage,
    grpo_advantage,
    length_penalty_advantage,
)
from prime_rl.orchestrator.algo.base import Algorithm

if TYPE_CHECKING:
    from renderers.base import Renderer

    from prime_rl.orchestrator.types import RolloutView
    from prime_rl.utils.client import InferencePool


class GRPOAlgorithm(Algorithm):
    """Group Relative Policy Optimization: sample a group of rollouts from the
    policy per example; credit = reward minus the group mean (optionally
    length-shaped); action tokens feed the ``rl`` loss."""

    def __init__(self, advantage: AdvantageConfig, policy_pool: InferencePool, renderer: Renderer | None):
        super().__init__(advantage, policy_pool, renderer)
        assert isinstance(advantage, GRPOAdvantageConfig)
        self.length_penalty = advantage.length_penalty
        self.length_weighted_baseline = advantage.length_weighted_baseline

    async def score_group(self, group: list[RolloutView]) -> None:
        length_penalty = self.length_penalty
        # tokens/turns are non-additive reward shaping — they replace the baseline.
        if isinstance(length_penalty, (TokensLengthPenaltyConfig, TurnsLengthPenaltyConfig)):
            advantages = efficiency_shaping_advantage(group, length_penalty)
        else:
            # The linear length penalty is a separate advantage that sums onto GRPO's.
            advantages = grpo_advantage(group, self.length_weighted_baseline)
            if isinstance(length_penalty, LinearLengthPenaltyConfig):
                penalty = length_penalty_advantage(
                    group, length_penalty, self.max_seq_len, self.length_weighted_baseline
                )
                advantages = [a + p for a, p in zip(advantages, penalty, strict=True)]
        for view, advantage in zip(group, advantages, strict=True):
            view.assign_advantages(advantage)
