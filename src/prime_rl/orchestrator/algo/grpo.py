from __future__ import annotations

from typing import TYPE_CHECKING

import verifiers.v1 as vf

from prime_rl.configs.algorithm import GRPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import assign_advantages, trainable_nodes

if TYPE_CHECKING:
    from prime_rl.orchestrator.clients import InferenceClient

# EMA decay of the mean group token count used by prompt-mean loss aggregation (~100-group window).
GROUP_TOKENS_DECAY = 0.99


def num_trainable_tokens(trace: vf.Trace) -> int:
    """Sampled (mask-True) tokens on the trace's trainable branches: the tokens its advantage is assigned to."""
    on_trainable_branch = {id(node) for branch in trace.branches if branch.trainable for node in branch.nodes}
    return sum(sum(node.mask) for node in trace.nodes if id(node) in on_trainable_branch)


class GRPOAlgorithm(Algorithm):
    """Group Relative Policy Optimization: sample a group of rollouts from the
    policy per example; credit = reward minus the group mean (optionally
    length-shaped); action tokens feed the ``rl`` loss."""

    def __init__(self, config: GRPOAlgoConfig, clients: InferenceClient):
        super().__init__(config, clients)
        self.length_penalty = config.length_penalty
        self.length_weighted_baseline = config.length_weighted_baseline
        self.loss_aggregation = config.loss_aggregation
        self.mean_group_tokens: float | None = None

    async def score_group(self, episodes: list[vf.Episode]) -> None:
        import torch  # only the trainer-side extras ship torch; an eval process never scores a group

        traces = [trace for _, trace in iter_trainable_traces(episodes)]
        rewards = torch.tensor([trace.reward for trace in traces], dtype=torch.float32)
        length_penalty = self.length_penalty
        if length_penalty is None:
            shaped_rewards = rewards
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
        baseline = shaped_rewards.mean()
        if self.length_weighted_baseline:
            lengths = torch.tensor(
                [sum(sum(node.mask) for node in trainable_nodes(trace)) for trace in traces], dtype=rewards.dtype
            )
            baseline = (lengths * shaped_rewards).sum() / lengths.sum()
        advantages = shaped_rewards - baseline
        if self.loss_aggregation == "prompt" and advantages.any():
            # Scale by T̄/T_q so the trainer's global token-mean becomes a per-prompt mean. All-zero groups
            # are skipped: the train sink drops their tokens from the rl denominator. T̄ is not checkpointed;
            # it re-warms within ~100 groups after a restart and until then only shifts the effective lr.
            group_tokens = sum(num_trainable_tokens(trace) for trace in traces)
            if self.mean_group_tokens is None:
                self.mean_group_tokens = float(group_tokens)
            else:
                self.mean_group_tokens = (
                    GROUP_TOKENS_DECAY * self.mean_group_tokens + (1.0 - GROUP_TOKENS_DECAY) * group_tokens
                )
            advantages = advantages * (self.mean_group_tokens / group_tokens)
        for trace, advantage in zip(traces, advantages.tolist(), strict=True):
            assign_advantages(trace, advantage)
