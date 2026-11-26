from __future__ import annotations

from typing import TYPE_CHECKING

import verifiers.v1 as vf
from verifiers.v1.trace import min_new_input_tokens

from prime_rl.configs.algorithm import GRPOAlgoConfig, LinearSubagentLengthPenaltyConfig
from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import assign_advantage, assign_rl_weight

if TYPE_CHECKING:
    from prime_rl.env.heads import EpisodeHead
    from prime_rl.orchestrator.clients import InferenceClient


def length_penalty_tokens(trace: vf.Trace) -> tuple[float, float, float, float]:
    """Output and new input tokens on trainable branches, counted once per call: those of
    the main agent's calls, then those of sub-agent calls."""
    subagents: set[int] = set()
    successors: dict[int, list[int]] = {}
    for node_id, node in enumerate(trace.nodes):
        for link in node.semantic_parents:
            if link.type == "subagent_call":
                subagents.add(node_id)
            # Return, cancel, and message edges cross agent boundaries.
            if link.type in {"subagent_call", "continuation", "compaction_attempt", "compaction"}:
                successors.setdefault(link.node, []).append(node_id)
    pending = list(subagents)
    while pending:
        for child in successors.get(pending.pop(), []):
            if child not in subagents:
                subagents.add(child)
                pending.append(child)

    output = input = subagent_output = subagent_input = 0.0
    seen: set[int] = set()
    for branch in trace.branches:
        if not branch.trainable:
            continue
        # Count increments before weighting: a discounted predecessor still
        # contributes its full history to the next call's prompt.
        for call, increment in min_new_input_tokens(branch.calls):
            if id(call) in seen:
                continue
            seen.add(id(call))
            if call.node in subagents:
                subagent_output += call.usage.completion_tokens
                subagent_input += increment
            else:
                output += call.usage.completion_tokens
                input += increment
    return output, input, subagent_output, subagent_input


class GRPOAlgorithm(Algorithm):
    """Group Relative Policy Optimization: sample a group of rollouts from the
    policy per example; credit = reward minus the group mean (optionally
    length-shaped); action tokens feed the ``rl`` loss."""

    def __init__(self, config: GRPOAlgoConfig, clients: InferenceClient):
        super().__init__(config, clients)
        self.length_penalty = config.length_penalty
        self.length_weighted_baseline = config.length_weighted_baseline
        self.loss_aggregation = config.loss_aggregation

    def reconfigure(self, config: GRPOAlgoConfig) -> None:
        self.length_penalty = config.length_penalty

    async def score_group(self, episodes: list[EpisodeHead]) -> None:
        import torch  # only the trainer-side extras ship torch; an eval process never scores a group

        traces = [trace for _, trace in iter_trainable_traces(episodes)]
        rewards = torch.tensor([trace.reward for trace in traces], dtype=torch.float32)
        length_penalty = self.length_penalty
        if length_penalty is None:
            shaped_rewards = rewards
        else:
            subagent_weight = (
                length_penalty.subagent_token_weight
                if isinstance(length_penalty, LinearSubagentLengthPenaltyConfig)
                else 1.0
            )
            lengths = [trace.length_tokens for trace in traces]
            output = torch.tensor(
                [length.output + subagent_weight * length.subagent_output for length in lengths], dtype=rewards.dtype
            )
            input = torch.tensor(
                [length.input + subagent_weight * length.subagent_input for length in lengths], dtype=rewards.dtype
            )
            turns = torch.tensor([trace.trainable_turns for trace in traces], dtype=rewards.dtype)
            penalty_frac = (
                length_penalty.num_output_tokens_weight * (output / output.max().clamp(min=1))
                + length_penalty.num_input_tokens_weight * (input / input.max().clamp(min=1))
                + length_penalty.num_turns_weight * (turns / turns.max().clamp(min=1))
            )
            penalty = rewards.mean() * penalty_frac
            shaped_rewards = rewards - penalty
        baseline = shaped_rewards.mean()
        if self.length_weighted_baseline:
            lengths = torch.tensor([trace.trainable_tokens for trace in traces], dtype=rewards.dtype)
            baseline = (lengths * shaped_rewards).sum() / lengths.sum()
        advantages = shaped_rewards - baseline
        for trace, advantage in zip(traces, advantages.tolist(), strict=True):
            assign_advantage(trace, advantage)
        num_tokens = sum(trace.trainable_tokens for trace in traces)
        if self.loss_aggregation == "prompt" and num_tokens:
            # rl weight 1/T_q per token: each group's weights sum to 1, and the trainer divides the
            # rl loss by the sum of rl weights, i.e. the number of groups.
            for trace in traces:
                assign_rl_weight(trace, 1.0 / num_tokens)
