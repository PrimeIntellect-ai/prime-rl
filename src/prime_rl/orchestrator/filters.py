"""Orchestrator-side rollout filters for detecting degenerate generations.

Filters run after rollouts complete, inspecting token IDs and logprobs to
detect gibberish or repetition. Detection metrics are always tracked.
When enforce=True, detected rollouts are skipped entirely during training and
are not sent to the trainer. Reward is kept as-is for baseline calculation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from prime_rl.configs.orchestrator import FilterConfig
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import AgentGraph, TrainingTrace


@dataclass
class FilterResult:
    detected: bool


class GraphFilter(Protocol):
    name: str
    enforce: bool

    def check(self, trace: TrainingTrace) -> FilterResult: ...


@dataclass
class GibberishFilter:
    """Flags rollouts containing rare tokens generated at high entropy.

    A token is flagged when both:
      - id(token) > token_id_threshold  (rare BPE token)
      - logprob(token) < -log(vocab_size) - logprob_offset  (high entropy)

    References:
      Section 5.2, https://arxiv.org/abs/2510.02387
    """

    name: str
    token_id_threshold: int
    logprob_threshold: float
    enforce: bool = False

    def check(self, trace: TrainingTrace) -> FilterResult:
        for branch in trace.branches:
            # branch.{token_ids,logprobs,sampled_mask} are flat and mutually aligned; the raw
            # node arrays are not (node.logprobs covers only the sampled suffix, not the
            # generation-prompt scaffold that token_ids/mask also span).
            for token_id, logprob, sampled in zip(branch.token_ids, branch.logprobs, branch.sampled_mask):
                if not sampled:
                    continue
                if token_id > self.token_id_threshold and logprob < self.logprob_threshold:
                    return FilterResult(detected=True)
        return FilterResult(detected=False)


@dataclass
class RepetitionFilter:
    """Flags rollouts with pathological repetition loops.

    Counts consecutive tokens where logprob > log(prob_threshold), indicating
    the model is generating with very high confidence. When the streak reaches
    the window size, the rollout is flagged.

    References:
      Section 3.2, https://arxiv.org/abs/2506.13585
    """

    name: str
    window: int
    logprob_threshold: float
    enforce: bool = False

    def check(self, trace: TrainingTrace) -> FilterResult:
        for branch in trace.branches:
            # Aligned branch streams (see GibberishFilter), and reset the streak per branch:
            # flat rollout.nodes interleaves distinct root->leaf paths (compaction/subagents),
            # so a per-node walk would run a streak across a branch boundary.
            consecutive = 0
            for logprob, sampled in zip(branch.logprobs, branch.sampled_mask):
                if not sampled:
                    continue
                if logprob > self.logprob_threshold:
                    consecutive += 1
                else:
                    consecutive = 0
                if consecutive >= self.window:
                    return FilterResult(detected=True)
        return FilterResult(detected=False)


@dataclass
class ZeroAdvantageFilter:
    """Flags rollouts whose advantage stream is all zero (e.g. all rollouts in
    a GRPO group earned the same reward, so the centered advantage collapses)."""

    name: str
    enforce: bool = True

    def check(self, trace: TrainingTrace) -> FilterResult:
        if trace.advantages is not None and all(a == 0.0 for a in trace.advantages):
            return FilterResult(detected=True)
        return FilterResult(detected=False)


def setup_filter(config: FilterConfig, vocab_size: int) -> GraphFilter:
    """Create a graph filter from a filter config."""
    if config.type == "gibberish":
        return GibberishFilter(
            name="gibberish",
            token_id_threshold=config.token_id_threshold,
            logprob_threshold=-math.log(vocab_size) - config.logprob_offset,
            enforce=config.enforce,
        )
    elif config.type == "repetition":
        return RepetitionFilter(
            name="repetition",
            window=config.window,
            logprob_threshold=math.log(config.prob_threshold),
            enforce=config.enforce,
        )
    elif config.type == "zero_advantage":
        return ZeroAdvantageFilter(
            name="zero_advantage",
            enforce=config.enforce,
        )
    raise ValueError(f"Unknown filter type: {config.type}")


def setup_filters(configs: list[FilterConfig], vocab_size: int, *, kind: str) -> list[GraphFilter]:
    """Create graph filters from filter configs."""
    filters = [setup_filter(config, vocab_size) for config in configs]
    if filters:
        get_logger().info(f"Configured {len(filters)} {kind} rollout filter(s):")
        for config, filt in zip(configs, filters):
            mode = "Enforcing" if filt.enforce else "Monitoring"
            params = ", ".join(f"{k}={v}" for k, v in config.model_dump().items())
            get_logger().info(f"  {mode} {filt.name} filter ({params})")
    return filters


def apply_filters(filters: list[GraphFilter], graphs: list[AgentGraph]) -> None:
    """Flag graphs in place with per-filter detection and drop decisions.

    Each graph's ``filter_results`` dict records per-filter detection bools;
    ``is_filtered`` is True iff an enforcing filter detected it. First matching
    filter wins per rollout (no double-counting). Reward and trajectory tokens
    are left untouched so the rollout can still contribute to baseline
    calculations and metric aggregation.
    """
    for graph in graphs:
        graph.filter_results = {f.name: False for f in filters}
        graph.is_filtered = False

    if not filters:
        return

    for graph in graphs:
        trace = graph.training_trace
        for filt in filters:
            result = filt.check(trace)
            if result.detected:
                graph.filter_results[filt.name] = True
                if filt.enforce:
                    graph.is_filtered = True
                break
