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
                penalty_frac = torch.tensor(
                    [
                        length_penalty.cost_weight * c["cost_usd"] + length_penalty.time_weight * c["time_s"]
                        for c in costs
                    ],
                    dtype=rewards.dtype,
                )
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
        for trace, advantage in zip(traces, advantages.tolist(), strict=True):
            assign_advantages(trace, advantage)
        nodes = [node for trace in traces for node in trainable_nodes(trace)]
        if self.loss_aggregation == "prompt" and nodes:
            # rl weight 1/T_q per token: each group's weights sum to 1, and the trainer divides the
            # rl loss by the sum of rl weights, i.e. the number of groups.
            weight = 1.0 / sum(sum(node.mask) for node in nodes)
            for node in nodes:
                node.loss_weights = {**(node.loss_weights or {}), "rl": [weight if m else 0.0 for m in node.mask]}


def _common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def rollout_cost(trace: vf.Trace, penalty: CostPenaltyConfig) -> dict[str, float]:
    """Modelled deployment cost (USD) and wait time (s) of one rollout, with their parts.

    A call's input is cached up to its longest common prefix with any earlier call's
    prompt + completion in the trace (all agents, by call start). Graph nodes dedup
    identical prefixes, so the prefix is the leading already-seen nodes on the call's
    path, extended token-wise into the first unseen node against its seen siblings."""
    nodes = trace.nodes
    seen: set[int] = set()
    seen_children: dict[int | None, list[int]] = {}
    uncached = cached = output = 0
    for call in sorted((c for c in trace.calls if c.node is not None), key=lambda c: c.time.start):
        path = [call.node]
        while (parent := nodes[path[-1]].parent) is not None:
            path.append(parent)
        path.reverse()
        node = nodes[call.node]
        num_sampled = sum(node.mask)
        prompt_tail = node.token_ids[: node.mask.index(True)] if num_sampled else node.token_ids
        spans = [nodes[n].token_ids for n in path[:-1]] + [prompt_tail]
        k = 0
        while k < len(path) - 1 and path[k] in seen:
            k += 1
        siblings = seen_children.get(path[k - 1] if k else None, [])
        hit = sum(map(len, spans[:k])) + max(
            (_common_prefix(spans[k], nodes[s].token_ids) for s in siblings), default=0
        )
        cached += hit
        uncached += sum(map(len, spans)) - hit
        output += num_sampled
        for parent, n in zip([None, *path], path):
            if n not in seen:
                seen.add(n)
                seen_children.setdefault(parent, []).append(n)

    intervals = sorted((c.time.start, c.time.end) for c in trace.calls if c.time.duration > 0)
    busy = sum(end - start for start, end in intervals)
    union, reach = 0.0, float("-inf")
    for start, end in intervals:
        union += max(0.0, end - max(start, reach))
        reach = max(reach, end)
    parallelism = union / busy if busy else 1.0
    model_time = (uncached / penalty.prefill_tokens_per_s + output / penalty.decode_tokens_per_s) * parallelism
    timing = trace.timing
    tool_time = max(0.0, timing.agent.duration - union)
    sandbox_hours = sum(s.duration for s in (timing.boot, timing.setup, timing.agent, timing.finalize)) / 3600
    sandbox_cost = penalty.sandbox_usd_per_hour * sandbox_hours
    token_cost = (
        penalty.input_usd_per_mtok * uncached
        + penalty.cached_input_usd_per_mtok * cached
        + penalty.output_usd_per_mtok * output
    ) / 1e6
    cost = token_cost + sandbox_cost
    return {
        "cost_usd": cost,
        "time_s": model_time + tool_time,
        "model_time_s": model_time,
        "tool_time_s": tool_time,
        "parallelism": parallelism,
        "prefix_cache_hit": cached / max(1, cached + uncached),
        "sandbox_cost_frac": sandbox_cost / cost if cost else 0.0,
    }
