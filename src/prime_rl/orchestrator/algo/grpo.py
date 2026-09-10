from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import verifiers.v1 as vf

from prime_rl.configs.algorithm import GRPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import assign_advantages

if TYPE_CHECKING:
    from prime_rl.orchestrator.clients import InferenceClient


def num_trainable_tokens(trace: vf.Trace) -> int:
    """Trainable (mask-True) tokens across the trace's trainable branches."""
    trainable_nodes = {id(node) for branch in trace.branches if branch.trainable for node in branch.nodes}
    return sum(sum(node.mask) for node in trace.nodes if id(node) in trainable_nodes)


class GRPOAlgorithm(Algorithm):
    """Group Relative Policy Optimization: sample a group of rollouts from the
    policy per example; credit = reward minus the group baseline — the group
    mean, or the trainable-token-weighted mean (SWE-2) — optionally length-
    shaped first; action tokens feed the ``rl`` loss."""

    def __init__(self, config: GRPOAlgoConfig, clients: InferenceClient):
        super().__init__(config, clients)
        self.length_penalty = config.length_penalty
        self.length_weighted_baseline = config.length_weighted_baseline

    async def score_group(self, episodes: list[vf.Episode]) -> None:
        traces = [trace for _, trace in iter_trainable_traces(episodes)]
        rewards = torch.tensor([trace.reward for trace in traces], dtype=torch.float32)
        shaped_rewards = self._apply_length_penalty(rewards, traces)
        advantages = shaped_rewards - self._baseline(shaped_rewards, traces)
        for trace, advantage in zip(traces, advantages.tolist(), strict=True):
            assign_advantages(trace, advantage)

    def _apply_length_penalty(self, rewards: torch.Tensor, traces: list[vf.Trace]) -> torch.Tensor:
        length_penalty = self.length_penalty
        if length_penalty is None:
            return rewards
        output = torch.tensor([trace.num_output_tokens for trace in traces], dtype=rewards.dtype)
        total = torch.tensor([trace.num_total_tokens for trace in traces], dtype=rewards.dtype)
        turns = torch.tensor([trace.num_turns for trace in traces], dtype=rewards.dtype)
        input = total - output
        penalty_frac = (
            length_penalty.num_output_tokens_weight * (output / output.max().clamp(min=1))
            + length_penalty.num_input_tokens_weight * (input / input.max().clamp(min=1))
            + length_penalty.num_turns_weight * (turns / turns.max().clamp(min=1))
        )
        return rewards - rewards.mean() * penalty_frac

    def _baseline(self, rewards: torch.Tensor, traces: list[vf.Trace]) -> torch.Tensor:
        if not self.length_weighted_baseline:
            return rewards.mean()
        lengths = torch.tensor([num_trainable_tokens(trace) for trace in traces], dtype=rewards.dtype)
        return (rewards * lengths).sum() / lengths.sum()
