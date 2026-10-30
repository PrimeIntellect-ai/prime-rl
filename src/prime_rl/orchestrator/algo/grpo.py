from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import verifiers.v1 as vf
from verifiers.v1.trace import min_new_input_tokens

from prime_rl.configs.algorithm import GRPOAlgoConfig, LinearSubagentLengthPenaltyConfig, PowerLengthPenaltyConfig
from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import assign_advantages, trainable_nodes

if TYPE_CHECKING:
    from prime_rl.orchestrator.clients import InferenceClient


def _length_penalty_tokens(trace: vf.Trace, subagent_token_weight: float) -> tuple[float, float]:
    """Weighted output and new input tokens for branches with no shared prefixes."""
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

    output_tokens = input_tokens = 0.0
    for branch in trace.branches:
        # Compute increments before weighting: a discounted predecessor still
        # contributes its full history to the next call's prompt.
        for call, increment in min_new_input_tokens(branch.calls):
            weight = subagent_token_weight if call.node in subagents else 1.0
            output_tokens += weight * call.usage.completion_tokens
            input_tokens += weight * increment
    return output_tokens, input_tokens


def _agent_lengths(trace: vf.Trace) -> list[tuple[bool, float]]:
    """Lifetime output plus new input tokens of each agent, and whether it was spawned. An agent is the
    calls of its branches, joined across compactions and turns by continuation and compaction edges;
    a ``subagent_call`` edge starts a spawned agent."""
    owner = list(range(len(trace.nodes)))

    def find(node: int) -> int:
        while owner[node] != node:
            owner[node] = owner[owner[node]]
            node = owner[node]
        return node

    spawned_nodes = []
    for node_id, node in enumerate(trace.nodes):
        for link in node.semantic_parents:
            if link.type in {"continuation", "compaction_attempt", "compaction"}:
                owner[find(node_id)] = find(link.node)
            elif link.type == "subagent_call":
                spawned_nodes.append(node_id)
    branches = [list(min_new_input_tokens(branch.calls)) for branch in trace.branches]
    for calls in branches:
        for call, _ in calls:
            owner[find(call.node)] = find(calls[0][0].node)
    tokens: dict[int, float] = defaultdict(float)
    for calls in branches:
        for call, increment in calls:
            tokens[find(call.node)] += call.usage.completion_tokens + increment
    spawned = {find(node) for node in spawned_nodes}
    return [(agent in spawned, total) for agent, total in tokens.items()]


def _power_length_cost(trace: vf.Trace, config: PowerLengthPenaltyConfig) -> float:
    return sum(
        (config.subagent_weight if spawned else 1.0) * length**config.alpha for spawned, length in _agent_lengths(trace)
    )


class GRPOAlgorithm(Algorithm):
    """Group Relative Policy Optimization: sample a group of rollouts from the
    policy per example; credit = reward minus the group mean (optionally
    length-shaped); action tokens feed the ``rl`` loss."""

    def __init__(self, config: GRPOAlgoConfig, clients: InferenceClient):
        super().__init__(config, clients)
        self.length_penalty = config.length_penalty
        self.length_weighted_baseline = config.length_weighted_baseline
        self.loss_aggregation = config.loss_aggregation

    async def score_group(self, episodes: list[vf.Episode]) -> None:
        import torch  # only the trainer-side extras ship torch; an eval process never scores a group

        traces = [trace for _, trace in iter_trainable_traces(episodes)]
        rewards = torch.tensor([trace.reward for trace in traces], dtype=torch.float32)
        length_penalty = self.length_penalty
        if length_penalty is None:
            shaped_rewards = rewards
        elif isinstance(length_penalty, PowerLengthPenaltyConfig):
            cost = torch.tensor([_power_length_cost(trace, length_penalty) for trace in traces], dtype=torch.float64)
            penalty_frac = length_penalty.weight * cost / cost.max().clamp(min=1)
            shaped_rewards = rewards - rewards.mean() * penalty_frac.to(rewards.dtype)
        else:
            if isinstance(length_penalty, LinearSubagentLengthPenaltyConfig):
                token_counts = [_length_penalty_tokens(trace, length_penalty.subagent_token_weight) for trace in traces]
                output = torch.tensor([output for output, _ in token_counts], dtype=rewards.dtype)
                input = torch.tensor([input for _, input in token_counts], dtype=rewards.dtype)
            else:
                output = torch.tensor([trace.num_output_tokens for trace in traces], dtype=rewards.dtype)
                total = torch.tensor([trace.num_total_tokens for trace in traces], dtype=rewards.dtype)
                input = total - output
            turns = torch.tensor([trace.num_turns for trace in traces], dtype=rewards.dtype)
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
                [sum(int(np.count_nonzero(node.mask)) for node in trainable_nodes(trace)) for trace in traces],
                dtype=rewards.dtype,
            )
            baseline = (lengths * shaped_rewards).sum() / lengths.sum()
        advantages = shaped_rewards - baseline
        for trace, advantage in zip(traces, advantages.tolist(), strict=True):
            assign_advantages(trace, advantage)
        nodes = [node for trace in traces for node in trainable_nodes(trace)]
        if self.loss_aggregation == "prompt" and nodes:
            # rl weight 1/T_q per token: each group's weights sum to 1, and the trainer divides the
            # rl loss by the sum of rl weights, i.e. the number of groups.
            weight = 1.0 / sum(int(np.count_nonzero(node.mask)) for node in nodes)
            for node in nodes:
                rl = np.where(np.asarray(node.mask, dtype=bool), weight, 0.0).astype(np.float32)
                node.loss_weights = {**(node.loss_weights or {}), "rl": rl}
