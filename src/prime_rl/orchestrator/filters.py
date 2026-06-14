"""Orchestrator-side rollout filters for detecting degenerate generations.

Filters run after rollouts complete, inspecting token IDs and logprobs to
detect gibberish or repetition. Detection metrics are always tracked.
When enforce=True, detected rollouts are skipped entirely during training and
are not sent to the trainer. Reward is kept as-is for baseline calculation.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from prime_rl.configs.orchestrator import FilterConfig
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import TrainRollout


@dataclass(frozen=True)
class GeneratedTokenLogprob:
    token_id: int
    logprob: float


def _iter_generated_token_logprobs(rollout: "TrainRollout") -> Iterator[GeneratedTokenLogprob]:
    """Yield only model-generated completion tokens for rollout filters.

    ``TrainSink`` builds ``rollout.samples`` before pre-filters run. Those
    samples survive raw trajectory compaction, so filters prefer them when
    present. Tests and direct callers can still use the raw trajectory path.
    """
    if rollout.samples:
        for sample in rollout.samples:
            for token_id, logprob, is_generated in zip(
                sample.completion_ids,
                sample.completion_logprobs,
                sample.completion_mask,
            ):
                if is_generated:
                    yield GeneratedTokenLogprob(token_id=token_id, logprob=logprob)
        return

    for step in rollout.raw.get("trajectory") or []:
        tokens = step.get("tokens") if isinstance(step, dict) else None
        if tokens is None or "completion_ids" not in tokens or "completion_logprobs" not in tokens:
            continue
        mask = tokens.get("completion_mask")
        if mask is None:
            mask = [True] * len(tokens["completion_ids"])
        for token_id, logprob, is_generated in zip(tokens["completion_ids"], tokens["completion_logprobs"], mask):
            if is_generated:
                yield GeneratedTokenLogprob(token_id=token_id, logprob=logprob)


@dataclass
class FilterResult:
    detected: bool
    detection_index: int | None = None


class RolloutFilter(Protocol):
    name: str
    enforce: bool

    def check(self, rollout: "TrainRollout") -> FilterResult: ...


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

    def check(self, rollout: "TrainRollout") -> FilterResult:
        global_idx = 0
        for token in _iter_generated_token_logprobs(rollout):
            if token.token_id > self.token_id_threshold and token.logprob < self.logprob_threshold:
                return FilterResult(detected=True, detection_index=global_idx)
            global_idx += 1
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

    def check(self, rollout: "TrainRollout") -> FilterResult:
        consecutive = 0
        global_idx = 0
        for token in _iter_generated_token_logprobs(rollout):
            if token.logprob > self.logprob_threshold:
                consecutive += 1
            else:
                consecutive = 0
            if consecutive >= self.window:
                return FilterResult(detected=True, detection_index=global_idx)
            global_idx += 1
        return FilterResult(detected=False)


@dataclass
class ZeroAdvantageFilter:
    """Flags rollouts whose computed advantage is zero (e.g. all rollouts in a
    GRPO group earned the same reward, so the centered advantage collapses)."""

    name: str
    enforce: bool = True

    def check(self, rollout: "TrainRollout") -> FilterResult:
        if rollout.advantage is not None and rollout.advantage == 0.0:
            return FilterResult(detected=True)
        return FilterResult(detected=False)


def setup_filter(config: FilterConfig, vocab_size: int) -> RolloutFilter:
    """Create a RolloutFilter from a filter config."""
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


def setup_filters(configs: list[FilterConfig], vocab_size: int, *, kind: str) -> list[RolloutFilter]:
    """Create RolloutFilters from a list of filter configs."""
    filters = [setup_filter(config, vocab_size) for config in configs]
    if filters:
        get_logger().info(f"Configured {len(filters)} {kind} rollout filter(s):")
        for config, filt in zip(configs, filters):
            mode = "Enforcing" if filt.enforce else "Monitoring"
            params = ", ".join(f"{k}={v}" for k, v in config.model_dump().items())
            get_logger().info(f"  {mode} {filt.name} filter ({params})")
    return filters


def apply_filters(filters: list[RolloutFilter], rollouts: list["TrainRollout"]) -> None:  # noqa: F821 (forward ref)
    """Flag ``TrainRollout``\\ s in place with per-filter detection + drop decision.

    Each rollout's ``filter_results`` dict records per-filter detection bools;
    ``is_filtered`` is True iff an enforcing filter detected it. First matching
    filter wins per rollout (no double-counting). Reward and trajectory tokens
    are left untouched so the rollout can still contribute to baseline
    calculations and metric aggregation.
    """
    for rollout in rollouts:
        rollout.filter_results = {f.name: False for f in filters}
        rollout.is_filtered = False

    if not filters:
        return

    for rollout in rollouts:
        for filt in filters:
            result = filt.check(rollout)
            if result.detected:
                rollout.filter_results[filt.name] = True
                if filt.enforce:
                    rollout.is_filtered = True
                break
