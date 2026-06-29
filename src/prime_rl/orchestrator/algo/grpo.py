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
        self.length_pen = config.length_pen

    async def score_group(self, group: list[Rollout]) -> None:
        rewards = torch.tensor([rollout.reward for rollout in group], dtype=torch.float32)
        length_pen = self.length_pen
        if length_pen is None:
            advantages = rewards - rewards.mean()
        else:
            completion = torch.tensor([rollout.num_output_tokens for rollout in group], dtype=rewards.dtype)
            total = torch.tensor([rollout.num_total_tokens for rollout in group], dtype=rewards.dtype)
            turns = torch.tensor([rollout.num_turns for rollout in group], dtype=rewards.dtype)
            input = total - completion
            penalty_frac = (
                length_pen.completion_pen * (completion / completion.max().clamp(min=1))
                + length_pen.input_pen * (input / input.max().clamp(min=1))
                + length_pen.turns_pen * (turns / turns.max().clamp(min=1))
            )
            penalty = rewards.mean() * penalty_frac
            shaped_rewards = rewards - penalty
            advantages = shaped_rewards - shaped_rewards.mean()
        for rollout, advantage in zip(group, advantages.tolist(), strict=True):
            rollout.assign_advantages(advantage)
