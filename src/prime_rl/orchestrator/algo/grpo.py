from __future__ import annotations

import torch
import verifiers.v1 as vf

from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import assign_advantages, training_reward


class GRPOAlgorithm(Algorithm):
    """Group Relative Policy Optimization: sample a group of rollouts from the
    policy per example; credit = training reward minus the group mean; action
    tokens feed the ``rl`` loss."""

    async def score_group(self, episodes: list[vf.Episode]) -> None:
        traces = [trace for _, trace in iter_trainable_traces(episodes)]
        rewards = torch.tensor([training_reward(trace) for trace in traces], dtype=torch.float32)
        advantages = rewards - rewards.mean()
        for trace, advantage in zip(traces, advantages.tolist(), strict=True):
            assign_advantages(trace, advantage)
