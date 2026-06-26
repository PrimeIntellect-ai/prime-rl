from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from prime_rl.configs.algorithm import GRPOAlgorithmConfig, TokensLengthPenaltyConfig
from prime_rl.orchestrator.algo.advantage import efficiency_shaping
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.orchestrator.utils import get_tool_response_len

if TYPE_CHECKING:
    from renderers.base import Renderer

    from prime_rl.orchestrator.types import Rollout
    from prime_rl.utils.client import InferencePool


class GRPOAlgorithm(Algorithm):
    """Group Relative Policy Optimization: sample a group of rollouts from the
    policy per example; credit = reward minus the group mean (optionally
    length-shaped); action tokens feed the ``rl`` loss."""

    def __init__(self, config: GRPOAlgorithmConfig, policy_pool: InferencePool, renderer: Renderer | None):
        super().__init__(config, policy_pool, renderer)
        self.length_penalty = config.length_penalty

    async def score_group(self, group: list[Rollout]) -> None:
        rewards = torch.tensor([rollout.reward for rollout in group], dtype=torch.float32)
        lp = self.length_penalty
        if lp is None:
            advantages = rewards - rewards.mean()
        elif isinstance(lp, TokensLengthPenaltyConfig):
            costs = torch.tensor(
                [
                    lp.completion_weight * rollout.completion_len
                    + lp.tool_response_weight * get_tool_response_len(rollout)
                    for rollout in group
                ],
                dtype=rewards.dtype,
            )
            advantages = efficiency_shaping(rewards, costs)
        else:  # TurnsLengthPenaltyConfig
            costs = torch.tensor([rollout.num_turns for rollout in group], dtype=rewards.dtype)
            advantages = efficiency_shaping(rewards, costs)
        for rollout, advantage in zip(group, advantages.tolist(), strict=True):
            rollout.assign_advantages(advantage)
