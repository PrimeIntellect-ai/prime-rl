from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from prime_rl.orchestrator.algo.base import Algorithm

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import AgentGraph


class MaxRLAlgorithm(Algorithm):
    """Maximum-likelihood RL (arXiv:2602.02710): the GRPO pipeline with
    mean-normalized advantages — ``(reward − group mean) / group mean`` instead
    of plain centering. Normalizing by the mean instead of the standard
    deviation makes the policy gradient unbiased for the order-``group_size``
    truncation of the maximum-likelihood objective (low-pass-rate examples get
    ~1/p weight; ``group_size`` interpolates REINFORCE at 1 → exact maximum
    likelihood as it grows).

    Assumes non-negative (canonically binary) rewards; a group with mean reward
    <= 0 carries no signal and gets zero advantages (the zero-advantage filter
    drops it, matching the paper's no-success convention)."""

    async def score_group(self, group: list[AgentGraph]) -> None:
        traces = [self.training_trace(graph) for graph in group]
        rewards = torch.tensor([trace.reward for trace in traces], dtype=torch.float32)
        mean = rewards.mean()
        advantages = torch.zeros_like(rewards) if mean <= 0 else (rewards - mean) / mean
        for trace, advantage in zip(traces, advantages.tolist(), strict=True):
            trace.assign_advantages(advantage)
