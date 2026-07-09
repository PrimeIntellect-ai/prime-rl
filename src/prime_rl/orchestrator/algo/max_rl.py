from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from prime_rl.orchestrator.algo.base import Algorithm

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout


class MaxRLAlgorithm(Algorithm):
    """Maximum-likelihood RL (arXiv:2602.02710): the GRPO pipeline with
    mean-normalized advantages — ``(reward − group mean) / group mean`` instead
    of plain centering. Normalizing by the mean instead of the standard
    deviation makes the policy gradient unbiased for the order-``group_size``
    truncation of the maximum-likelihood objective (low-pass-rate examples get
    ~1/p weight; ``group_size`` interpolates REINFORCE at 1 → exact maximum
    likelihood as it grows).

    A singleton group uses its reward directly, recovering REINFORCE as the
    paper requires. Assumes non-negative (canonically binary) rewards; larger
    groups with mean reward <= 0 carry no signal and get zero advantages (the
    zero-advantage filter drops them, matching the paper's no-success
    convention)."""

    async def score_group(self, group: list[Rollout]) -> None:
        rewards = torch.tensor([rollout.reward for rollout in group], dtype=torch.float32)
        if len(group) == 1:
            advantages = rewards
        else:
            mean = rewards.mean()
            advantages = torch.zeros_like(rewards) if mean <= 0 else (rewards - mean) / mean
        for rollout, advantage in zip(group, advantages.tolist(), strict=True):
            rollout.assign_advantages(advantage)
