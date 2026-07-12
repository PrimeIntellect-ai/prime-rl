from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from prime_rl.configs.algorithm import GRPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import AgentGraph
    from prime_rl.utils.client import InferencePool


class GRPOAlgorithm(Algorithm):
    """Group Relative Policy Optimization: sample a group of rollouts from the
    policy per example; credit = reward minus the group mean (optionally
    length-shaped); action tokens feed the ``rl`` loss."""

    def __init__(self, config: GRPOAlgoConfig, policy_pool: InferencePool):
        super().__init__(config, policy_pool)
        self.length_penalty = config.length_penalty

    async def score_group(self, group: list[AgentGraph]) -> None:
        traces = [self.training_trace(graph) for graph in group]
        rewards = torch.tensor([trace.reward for trace in traces], dtype=torch.float32)
        length_penalty = self.length_penalty
        if length_penalty is None:
            advantages = rewards - rewards.mean()
        else:
            output = torch.tensor([trace.num_output_tokens for trace in traces], dtype=rewards.dtype)
            total = torch.tensor([trace.num_total_tokens for trace in traces], dtype=rewards.dtype)
            turns = torch.tensor([trace.num_turns for trace in traces], dtype=rewards.dtype)
            input = total - output
            penalty_frac = (
                length_penalty.num_output_tokens_weight * (output / output.max().clamp(min=1))
                + length_penalty.num_input_tokens_weight * (input / input.max().clamp(min=1))
                + length_penalty.num_turns_weight * (turns / turns.max().clamp(min=1))
            )
            penalty = rewards.mean() * penalty_frac
            shaped_rewards = rewards - penalty
            advantages = shaped_rewards - shaped_rewards.mean()
        for trace, advantage in zip(traces, advantages.tolist(), strict=True):
            trace.assign_advantages(advantage)
