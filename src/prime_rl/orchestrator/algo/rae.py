from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, ClassVar

from prime_rl.configs.algorithm import RAEAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout
    from prime_rl.utils.client import InferencePool


class RAEAlgorithm(Algorithm):
    """Role-conditioned Advantage Estimation (SPIRAL, arXiv 2506.24119): advantage
    = reward minus a per-role EMA baseline maintained across the whole run — the
    self-play estimator for zero-sum multi-seat games, where a sibling-relative
    baseline would couple the opponents' credit. The algorithm instance lives
    per-env, so the paper's game×role conditioning reduces to role here.

    A role's first group centers on its own mean (advantage sums to zero, like a
    first GRPO group); after that the advantage is measured against the pre-update
    baseline, then the EMA absorbs the group's mean."""

    multi_seat: ClassVar[bool] = True

    def __init__(self, config: RAEAlgoConfig, policy_pool: InferencePool):
        super().__init__(config, policy_pool)
        self.alpha = config.alpha
        self.baselines: dict[str | None, float] = {}

    async def score_group(self, group: list[Rollout]) -> None:
        by_role: dict[str | None, list[Rollout]] = defaultdict(list)
        for rollout in group:
            by_role[rollout.role].append(rollout)
        for role, members in by_role.items():
            mean = sum(m.reward for m in members) / len(members)
            baseline = self.baselines.get(role, mean)
            for member in members:
                member.assign_advantages(member.reward - baseline)
            self.baselines[role] = self.alpha * baseline + (1 - self.alpha) * mean
