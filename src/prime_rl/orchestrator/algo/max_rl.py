from __future__ import annotations

import torch
import verifiers.v1 as vf

from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import assign_advantages


class MaxRLAlgorithm(Algorithm):
    """Maximum-likelihood RL (arXiv:2602.02710): the GRPO pipeline with
    mean-normalized advantages — ``(reward − group mean) / group mean`` instead
    of plain centering. Normalizing by the mean instead of the standard
    deviation makes the policy gradient unbiased for the order-``group_size``
    truncation of the maximum-likelihood objective (low-pass-rate examples get
    ~1/p weight; ``group_size`` interpolates REINFORCE at 1 → exact maximum
    likelihood as it grows).

    Assumes non-negative (canonically binary) rewards; a group with mean reward
    <= 0 carries no signal and gets zero advantages."""

    async def score_group(self, episodes: list[vf.Episode]) -> None:
        traces = [trace for _, trace in iter_trainable_traces(episodes)]
        rewards = torch.tensor([trace.reward for trace in traces], dtype=torch.float32)
        mean = rewards.mean()
        advantages = torch.zeros_like(rewards) if mean <= 0 else (rewards - mean) / mean
        for trace, advantage in zip(traces, advantages.tolist(), strict=True):
            assign_advantages(trace, advantage)
