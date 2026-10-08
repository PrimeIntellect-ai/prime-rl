from __future__ import annotations

from typing import TYPE_CHECKING

import verifiers.v1 as vf

from prime_rl.configs.algorithm import CostPenaltyConfig, GRPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import assign_advantages, trainable_nodes

if TYPE_CHECKING:
    from prime_rl.orchestrator.clients import InferenceClient


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
        else:
            if length_penalty.type == "cost":
                costs = [rollout_cost(trace, length_penalty) for trace in traces]
                for trace, cost in zip(traces, costs, strict=True):
                    trace.record_metrics({f"cost_penalty/{name}": value for name, value in cost.items()})
                usd_per_s = length_penalty.latency.usd_per_hour / 3600 if length_penalty.latency else 0.0
                usd = torch.tensor([c["token_usd"] + usd_per_s * c.get("latency_s", 0.0) for c in costs])
                ungated_penalty = length_penalty.reward_per_usd * usd
            else:
                output = torch.tensor([trace.num_output_tokens for trace in traces], dtype=rewards.dtype)
                total = torch.tensor([trace.num_total_tokens for trace in traces], dtype=rewards.dtype)
                turns = torch.tensor([trace.num_turns for trace in traces], dtype=rewards.dtype)
                input = total - output
                ungated_penalty = (
                    length_penalty.num_output_tokens_weight * (output / output.max().clamp(min=1))
                    + length_penalty.num_input_tokens_weight * (input / input.max().clamp(min=1))
                    + length_penalty.num_turns_weight * (turns / turns.max().clamp(min=1))
                )
            penalty = rewards.mean() * ungated_penalty
            shaped_rewards = rewards - penalty
        baseline = shaped_rewards.mean()
        if self.length_weighted_baseline:
            lengths = torch.tensor(
                [sum(sum(node.mask) for node in trainable_nodes(trace)) for trace in traces], dtype=rewards.dtype
            )
            baseline = (lengths * shaped_rewards).sum() / lengths.sum()
        advantages = shaped_rewards - baseline
        for trace, advantage in zip(traces, advantages.tolist(), strict=True):
            assign_advantages(trace, advantage)
        nodes = [node for trace in traces for node in trainable_nodes(trace)]
        if self.loss_aggregation == "prompt" and nodes:
            # rl weight 1/T_q per token: each group's weights sum to 1, and the trainer divides the
            # rl loss by the sum of rl weights, i.e. the number of groups.
            weight = 1.0 / sum(sum(node.mask) for node in nodes)
            for node in nodes:
                node.loss_weights = {**(node.loss_weights or {}), "rl": [weight if m else 0.0 for m in node.mask]}


def rollout_cost(trace: vf.Trace, penalty: CostPenaltyConfig) -> dict[str, float]:
    """Token cost (USD) and prefix cache hit rate of one rollout and, with ``latency`` set,
    its modelled latency and measured tool time (s).

    Prefix cache at node granularity: a call's input is cached up to the deepest node of
    its path that was on an earlier call's path (all agents, by call start). Calls without
    token ids (non-policy models) and failed calls cost nothing and count as tool time."""
    nodes = trace.nodes
    prefix_len: list[int] = []
    for node in nodes:
        prefix_len.append(len(node.token_ids) + (prefix_len[node.parent] if node.parent is not None else 0))
    seen: set[int] = set()
    uncached = cached = output = 0
    policy_calls = [c for c in trace.calls if c.node is not None and nodes[c.node].token_ids]
    for call in sorted(policy_calls, key=lambda c: c.time.start):
        num_sampled = sum(nodes[call.node].mask)
        num_input = prefix_len[call.node] - num_sampled
        n = call.node
        while n is not None and n not in seen:
            seen.add(n)
            n = nodes[n].parent
        hit = min(prefix_len[n], num_input) if n is not None else 0
        cached += hit
        uncached += num_input - hit
        output += num_sampled
    token_usd = (
        penalty.input_usd_per_mtok * uncached
        + penalty.cached_input_usd_per_mtok * cached
        + penalty.output_usd_per_mtok * output
    ) / 1e6
    cost = {"token_usd": token_usd, "cache_hit_rate": cached / max(1, cached + uncached)}
    if penalty.latency is None:
        return cost

    intervals = sorted((c.time.start, c.time.end) for c in policy_calls if c.time.duration > 0)
    busy = sum(end - start for start, end in intervals)
    union, reach = 0.0, float("-inf")
    for start, end in intervals:
        union += max(0.0, end - max(start, reach))
        reach = max(reach, end)
    parallelism = union / busy if busy else 1.0
    tool_s = max(0.0, trace.timing.agent.duration - union)
    latency = penalty.latency
    latency_s = parallelism * (uncached / latency.prefill_tokens_per_s + output / latency.decode_tokens_per_s) + tool_s
    return {**cost, "latency_s": latency_s, "tool_s": tool_s}
