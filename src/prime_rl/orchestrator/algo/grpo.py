from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from prime_rl.configs.algorithm import GRPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout
    from prime_rl.utils.client import InferencePool


class GRPOAlgorithm(Algorithm):
    """Group Relative Policy Optimization: sample a group of rollouts from the
    policy per example; credit = reward minus the group mean (optionally
    length-shaped); action tokens feed the ``rl`` loss."""

    def __init__(self, config: GRPOAlgoConfig, policy_pool: InferencePool):
        super().__init__(config, policy_pool)
        self.length_penalty = config.length_penalty

    async def score_group(self, group: list[Rollout]) -> None:
        rewards = torch.tensor([rollout.reward for rollout in group], dtype=torch.float32)
        lp = self.length_penalty
        if lp is None:
            advantages = rewards - rewards.mean()
        else:
            # Linear pass_rate-scaled penalty subtracted from each reward before the baseline:
            # coef * completion + context_coef * (total - completion) over the group's longest
            # sequence, plus turns_coef * turns over the group's most turns.
            completion = torch.tensor([rollout.completion_len for rollout in group], dtype=rewards.dtype)
            total = torch.tensor([rollout.total_tokens for rollout in group], dtype=rewards.dtype)
            penalty_frac = (lp.coef * completion + lp.context_coef * (total - completion)) / total.max().clamp(min=1)
            if lp.turns_coef:
                turns = torch.tensor([rollout.num_turns for rollout in group], dtype=rewards.dtype)
                penalty_frac = penalty_frac + lp.turns_coef * (turns / turns.max().clamp(min=1))
            penalty = rewards.mean() * penalty_frac
            if lp.gate_by_correctness:
                penalty = penalty * rewards
            shaped_rewards = rewards - penalty
            advantages = shaped_rewards - shaped_rewards.mean()
        for rollout, advantage in zip(group, advantages.tolist(), strict=True):
            rollout.assign_advantages(advantage)
