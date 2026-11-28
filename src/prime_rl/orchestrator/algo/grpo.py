from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING

import numpy as np
import verifiers.v1 as vf
from verifiers.v1.trace import min_new_input_tokens

from prime_rl.configs.algorithm import (
    CosineLengthPenaltyScheduleConfig,
    GRPOAlgoConfig,
    LinearSubagentLengthPenaltyConfig,
)
from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import assign_advantages, trainable_nodes
from prime_rl.orchestrator.trajectories import iter_trainable_branches
from prime_rl.transports.batch.types import PAYLOAD_INFO_KEY
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from renderers.base import Renderer

    from prime_rl.orchestrator.clients import InferenceClient


def _length_penalty_tokens(trace: vf.Trace, subagent_token_weight: float) -> tuple[float, float]:
    """Weighted output and new input tokens, counted once on trainable branches."""
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
    seen: set[int] = set()
    for branch in trace.branches:
        if not branch.trainable:
            continue
        # Compute increments before weighting: a discounted predecessor still
        # contributes its full history to the next call's prompt.
        for call, increment in min_new_input_tokens(branch.calls):
            if id(call) in seen:
                continue
            seen.add(id(call))
            weight = subagent_token_weight if call.node in subagents else 1.0
            output_tokens += weight * call.usage.completion_tokens
            input_tokens += weight * increment
    return output_tokens, input_tokens


def schedule_scale(schedule: CosineLengthPenaltyScheduleConfig | None, episodes: list[vf.Episode]) -> float:
    """The length penalty's multiplier for a group: 1 without a schedule, else the schedule at the step that
    dispatched the group's rollouts (the earliest, should they differ). A group with no dispatch step on record
    gets the full penalty."""
    if schedule is None:
        return 1.0
    steps = [work.step for episode in episodes if (work := getattr(episode.run, "work", None)) is not None]
    return schedule.scale(min(steps)) if steps else schedule.max_scale


class GRPOAlgorithm(Algorithm):
    """Group Relative Policy Optimization: sample a group of rollouts from the
    policy per example; credit = reward minus the group mean (optionally
    length-shaped); action tokens feed the ``rl`` loss."""

    def __init__(self, config: GRPOAlgoConfig, clients: InferenceClient):
        super().__init__(config, clients)
        self.length_penalty = config.length_penalty
        self.fallback_reward = config.fallback_reward
        self.conditional_prefix = config.conditional_prefix
        self.zero_on_timeout = config.zero_on_timeout
        self.length_weighted_baseline = config.length_weighted_baseline
        self.loss_aggregation = config.loss_aggregation
        self._binary_group_ema: float = 0.0
        self._binary_group_count: int = 0
        self._prefixed_pos_ema: float = 0.0
        self._prefixed_pos_groups: int = 0
        self.renderer: Renderer | None = None

    async def setup(self) -> None:
        if self.conditional_prefix is not None:
            self._ensure_renderer()

    def _ensure_renderer(self) -> Renderer:
        if self.renderer is None:
            from renderers.base import create_renderer, load_tokenizer

            self.renderer = create_renderer(
                load_tokenizer(self.clients.model_name), self.clients.train_client.renderer
            )
        return self.renderer

    def reconfigure(self, config: GRPOAlgoConfig) -> None:
        self.length_penalty = config.length_penalty
        self.fallback_reward = config.fallback_reward
        self.conditional_prefix = config.conditional_prefix

    async def score_group(self, episodes: list[vf.Episode]) -> None:
        import torch  # only the trainer-side extras ship torch; an eval process never scores a group

        traces = [trace for _, trace in iter_trainable_traces(episodes)]
        if self.zero_on_timeout:
            for trace in traces:
                if trace.is_timeout:
                    trace.rewards.clear()
                    trace.record_reward("reward", 0.0, weight=1.0)
        rewards = torch.tensor([trace.reward for trace in traces], dtype=torch.float32)
        fallback = self.fallback_reward
        use_fallback = False
        if fallback is not None and traces:
            use_fallback = not any(trace.reward >= fallback.threshold for trace in traces)
            is_binary_group = 0.0 if use_fallback else 1.0
            if self._binary_group_count == 0:
                self._binary_group_ema = is_binary_group
            else:
                decay = fallback.anneal_ema_decay
                self._binary_group_ema = decay * self._binary_group_ema + (1.0 - decay) * is_binary_group
            self._binary_group_count += 1
            effective_weight = fallback.weight * ((1.0 - self._binary_group_ema) ** fallback.anneal_power)

            raw_scores = [
                0.0
                if (self.zero_on_timeout and trace.is_timeout) or trace.metrics.get(fallback.metric) is None
                else float(trace.metrics[fallback.metric])
                for trace in traces
            ]
            snr_pass = (
                max(raw_scores) >= fallback.min_max_score
                and (max(raw_scores) - min(raw_scores)) >= fallback.min_spread
            )
            for trace, raw in zip(traces, raw_scores, strict=True):
                trace.record_metric("binary_reward", trace.reward)
                trace.record_metric("fallback_reward_active", 1.0 if use_fallback else 0.0)
                trace.record_metric("fallback_reward_weight", effective_weight)
                trace.record_metric("fallback_binary_group_ema", self._binary_group_ema)
                if use_fallback:
                    if snr_pass:
                        rescaled = max(0.0, (raw - fallback.floor) / (1.0 - fallback.floor))
                        score = rescaled ** fallback.exponent
                    else:
                        score = 0.0
                    trace.rewards.clear()
                    trace.record_reward("reward", score, weight=effective_weight)
            if use_fallback:
                rewards = torch.tensor([trace.reward for trace in traces], dtype=torch.float32)
        length_penalty = (
            None
            if (use_fallback and fallback is not None and fallback.disable_length_penalty)
            else self.length_penalty
        )
        if length_penalty is None:
            shaped_rewards = rewards
        else:
            subagent_weight = (
                length_penalty.subagent_token_weight
                if isinstance(length_penalty, LinearSubagentLengthPenaltyConfig)
                else 1.0
            )
            token_counts = [_length_penalty_tokens(trace, subagent_weight) for trace in traces]
            output = torch.tensor([output for output, _ in token_counts], dtype=rewards.dtype)
            input = torch.tensor([input for _, input in token_counts], dtype=rewards.dtype)
            turns = torch.tensor(
                [
                    len(
                        {
                            id(node)
                            for branch in trace.branches
                            if branch.trainable
                            for node in branch.nodes
                            if node.sampled
                        }
                    )
                    for trace in traces
                ],
                dtype=rewards.dtype,
            )
            penalty_frac = (
                length_penalty.num_output_tokens_weight * (output / output.max().clamp(min=1))
                + length_penalty.num_input_tokens_weight * (input / input.max().clamp(min=1))
                + length_penalty.num_turns_weight * (turns / turns.max().clamp(min=1))
            )
            scale = schedule_scale(length_penalty.schedule, episodes)
            penalty = scale * rewards.mean() * penalty_frac
            shaped_rewards = rewards - penalty
            for trace, pen in zip(traces, penalty.tolist(), strict=True):
                trace.info["length_penalty_scale"] = scale
                trace.record_metric("length_penalty_scale", scale)
                trace.record_metric("length_penalty", pen)
        baseline = shaped_rewards.mean()
        if self.length_weighted_baseline:
            lengths = torch.tensor(
                [sum(int(np.count_nonzero(node.mask)) for node in trainable_nodes(trace)) for trace in traces],
                dtype=rewards.dtype,
            )
            baseline = (lengths * shaped_rewards).sum() / lengths.sum()
        advantages = shaped_rewards - baseline
        advantage_list = advantages.tolist()
        promoted_zero_group: set[int] = set()
        cfg_cp = self.conditional_prefix
        if (
            cfg_cp is not None
            and cfg_cp.zero_group_advantage is not None
            and traces
            and all(abs(a) == 0.0 for a in advantage_list)
            and max(trace.reward for trace in traces) < cfg_cp.zero_group_reward_threshold
        ):
            under_cap = (
                cfg_cp.max_global_prefixed_positive_fraction is None
                or self._prefixed_pos_groups == 0
                or self._prefixed_pos_ema <= cfg_cp.max_global_prefixed_positive_fraction
            )
            if under_cap:
                zero_cands: list[tuple[float, int]] = []
                for idx_t, trace in enumerate(traces):
                    if trace.is_timeout:
                        continue
                    raw = trace.info.get(cfg_cp.info_key)
                    if not (isinstance(raw, str) and raw.strip()):
                        continue
                    score = 0.0
                    if cfg_cp.zero_group_rank_metric is not None:
                        val = trace.metrics.get(cfg_cp.zero_group_rank_metric)
                        score = float(val) if val is not None else float("-inf")
                    zero_cands.append((score, idx_t))
                zero_cands.sort(key=lambda item: (-item[0], item[1]))
                for _, idx_t in zero_cands[: cfg_cp.zero_group_max_traces]:
                    advantage_list[idx_t] = float(cfg_cp.zero_group_advantage)
                    promoted_zero_group.add(idx_t)

        for trace, advantage in zip(traces, advantage_list, strict=True):
            assign_advantages(trace, advantage)
        if cfg_cp is not None and traces:
            await self._apply_conditional_prefixes(traces, advantage_list, promoted_zero_group=promoted_zero_group)
        nodes = [node for trace in traces for node in trainable_nodes(trace)]
        if self.loss_aggregation == "prompt" and nodes:
            # rl weight 1/T_q per token: each group's weights sum to 1, and the trainer divides the
            # rl loss by the sum of rl weights, i.e. the number of groups.
            weight = 1.0 / sum(int(np.count_nonzero(node.mask)) for node in nodes)
            for node in nodes:
                rl = np.where(np.asarray(node.mask, dtype=bool), weight, 0.0).astype(np.float32)
                node.loss_weights = {**(node.loss_weights or {}), "rl": rl}

    async def _apply_conditional_prefixes(
        self,
        traces: list[vf.Trace],
        advantages: list[float],
        *,
        promoted_zero_group: set[int] | None = None,
    ) -> None:
        promoted_zero_group = promoted_zero_group or set()
        cfg = self.conditional_prefix
        assert cfg is not None
        renderer = self._ensure_renderer()

        has_prefix: set[int] = set()
        candidates: list[tuple[int, str]] = []
        positive_indices = [i for i, adv in enumerate(advantages) if adv > 0.0]

        for i, (trace, adv) in enumerate(zip(traces, advantages, strict=True)):
            raw = trace.info.get(cfg.info_key)
            if isinstance(raw, str) and raw.strip():
                has_prefix.add(i)
                if not cfg.positive_only or adv > 0.0:
                    candidates.append((i, raw.strip()))

        if cfg.max_positive_fraction is not None and cfg.positive_only and positive_indices:
            cap = math.floor(len(positive_indices) * cfg.max_positive_fraction)
            candidates.sort(key=lambda item: (-advantages[item[0]], item[0]))
            candidates = candidates[:cap]

        skipped_length: set[int] = set()
        skipped_error: set[int] = set()
        applied: set[int] = set()

        # Prepare per-trace (prefix_ids, branches) before concurrent prefill scoring.
        work: list[tuple[int, list[int], list[vf.Branch]]] = []
        for i, prefix_text in candidates:
            trace = traces[i]
            branches = [branch for branch, _ in iter_trainable_branches(trace)]
            if not branches or any(not b.nodes or b.nodes[0].sampled for b in branches):
                continue
            prefix_ids = list(
                renderer.render_ids([{"role": "system", "content": prefix_text}], add_generation_prompt=False)
            )
            if not prefix_ids:
                continue
            if cfg.max_total_tokens is not None and any(
                len(prefix_ids) + len(b.token_ids) > cfg.max_total_tokens for b in branches
            ):
                skipped_length.add(i)
                continue
            work.append((i, prefix_ids, branches))

        async def _score_trace(prefix_ids: list[int], branches: list[vf.Branch]) -> list[list[float]]:
            return list(
                await asyncio.gather(
                    *(self.clients.score([*prefix_ids, *b.token_ids.tolist()]) for b in branches)
                )
            )

        if work:
            results = await asyncio.gather(
                *(_score_trace(prefix_ids, branches) for _, prefix_ids, branches in work),
                return_exceptions=True,
            )
            for (i, prefix_ids, branches), res in zip(work, results, strict=True):
                trace = traces[i]
                if isinstance(res, BaseException):
                    skipped_error.add(i)
                    get_logger().warning(
                        f"conditional_prefix: prefill score failed for trace {trace.id!r}; keeping un-prefixed: {res}"
                    )
                    continue
                if any(len(lp) != len(prefix_ids) + len(b.token_ids) for b, lp in zip(branches, res, strict=True)):
                    skipped_error.add(i)
                    get_logger().warning(
                        f"conditional_prefix: logprob length mismatch for trace {trace.id!r}; keeping un-prefixed"
                    )
                    continue

                prefix_arr = np.asarray(prefix_ids, dtype=np.int32)
                prefix_mask = np.zeros(len(prefix_ids), dtype=np.bool_)
                seen_roots: set[int] = set()
                for branch in branches:
                    root = branch.nodes[0]
                    if id(root) in seen_roots:
                        continue
                    seen_roots.add(id(root))
                    root.token_ids = np.concatenate([prefix_arr, np.asarray(root.token_ids, dtype=np.int32)])
                    root.mask = np.concatenate([prefix_mask, np.asarray(root.mask, dtype=np.bool_)])
                    if len(root.is_content):
                        root.is_content = np.concatenate([prefix_mask, np.asarray(root.is_content, dtype=np.bool_)])
                    if root.loss_weights:
                        root.loss_weights = {
                            name: np.concatenate(
                                [np.zeros(len(prefix_ids), dtype=np.float32), np.asarray(w, dtype=np.float32)]
                            )
                            for name, w in root.loss_weights.items()
                        }

                updated_nodes: set[int] = set()
                for branch, full_logprobs in zip(branches, res, strict=True):
                    vals = np.asarray(full_logprobs, dtype=np.float32)
                    offset = 0
                    for node in branch.nodes:
                        end = offset + len(node.token_ids)
                        mask = np.asarray(node.mask, dtype=bool)
                        if mask.any() and id(node) not in updated_nodes:
                            node.logprobs = vals[offset:end][mask]
                            updated_nodes.add(id(node))
                        offset = end

                if PAYLOAD_INFO_KEY in trace.info and isinstance(trace.info[PAYLOAD_INFO_KEY], dict):
                    payload_info = dict(trace.info[PAYLOAD_INFO_KEY])
                    if isinstance(payload_info.get("branches"), dict):
                        rem_branches = {
                            k: v
                            for k, v in payload_info["branches"].items()
                            if k not in {str(b.index) for b in branches}
                        }
                        payload_info["branches"] = rem_branches
                        trace.info[PAYLOAD_INFO_KEY] = payload_info

                applied.add(i)

        # If a zero-group-promoted trace failed prefixing (length/error), revert its advantage to 0.0
        # so an un-prefixed failing rollout is never reinforced on the bare prompt.
        for i in promoted_zero_group:
            if i not in applied:
                advantages[i] = 0.0
                assign_advantages(traces[i], 0.0)

        pos_after = [i for i, adv in enumerate(advantages) if adv > 0.0]
        if pos_after:
            group_prefixed_pos_frac = sum(1 for i in pos_after if i in applied) / float(len(pos_after))
            if self._prefixed_pos_groups == 0:
                self._prefixed_pos_ema = group_prefixed_pos_frac
            else:
                self._prefixed_pos_ema = (
                    cfg.ema_decay * self._prefixed_pos_ema + (1.0 - cfg.ema_decay) * group_prefixed_pos_frac
                )
            self._prefixed_pos_groups += 1

        for i, (trace, adv) in enumerate(zip(traces, advantages, strict=True)):
            was_applied = i in applied
            trace.record_metric("conditional_prefix/applied", 1.0 if was_applied else 0.0)
            trace.record_metric("conditional_prefix/has_prefix", 1.0 if i in has_prefix else 0.0)
            trace.record_metric("conditional_prefix/skipped_length", 1.0 if i in skipped_length else 0.0)
            trace.record_metric("conditional_prefix/skipped_error", 1.0 if i in skipped_error else 0.0)
            trace.record_metric(
                "conditional_prefix/zero_group_promoted",
                1.0 if (i in promoted_zero_group and was_applied) else 0.0,
            )
            trace.record_metric("conditional_prefix/prefixed_pos_ema", float(self._prefixed_pos_ema))
            if was_applied:
                trace.record_metric("conditional_prefix/adv_prefixed", float(adv))
            else:
                trace.record_metric("conditional_prefix/adv_bare", float(adv))
