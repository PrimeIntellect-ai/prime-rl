"""Orchestrator-side rollout filters for detecting degenerate generations.

Filters run after rollouts complete, inspecting token IDs and logprobs to
detect gibberish or repetition. Detection metrics are always tracked. Each
filter resolves to one of three actions:

- ``monitor``: only record detection metrics;
- ``drop``: detected rollouts are skipped entirely during training and are
  not sent to the trainer. Reward is kept as-is for baseline calculation;
- ``penalize``: detected rollouts stay trainable, but their reward is capped
  at ``penalty_reward``. Penalties must be applied before advantage
  computation to create negative policy-gradient signal — token/logprob
  based filters are ``pre_advantage`` phase so ``TrainSink.process_group``
  runs them before ``assign_advantages``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias

from prime_rl.configs.orchestrator import FilterAction, FilterConfig
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import TrainRollout


FilterPhase: TypeAlias = Literal["pre_advantage", "post_advantage"]

_ACTION_LOG_NAMES: dict[FilterAction, str] = {
    "monitor": "Monitoring",
    "drop": "Dropping",
    "penalize": "Penalizing",
}


@dataclass
class FilterResult:
    detected: bool
    detection_index: int | None = None


class RolloutFilter(Protocol):
    name: str
    action: FilterAction
    phase: FilterPhase
    penalty_reward: float

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
    action: FilterAction = "monitor"
    penalty_reward: float = -1.0
    phase: FilterPhase = "pre_advantage"

    def check(self, rollout: "TrainRollout") -> FilterResult:
        global_idx = 0
        for step in rollout.raw["trajectory"]:
            tokens = step["tokens"]
            if tokens is None:
                continue
            for token_id, logprob in zip(tokens["completion_ids"], tokens["completion_logprobs"]):
                if token_id > self.token_id_threshold and logprob < self.logprob_threshold:
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
    action: FilterAction = "monitor"
    penalty_reward: float = -1.0
    phase: FilterPhase = "pre_advantage"

    def check(self, rollout: "TrainRollout") -> FilterResult:
        consecutive = 0
        global_idx = 0
        for step in rollout.raw["trajectory"]:
            tokens = step["tokens"]
            if tokens is None:
                continue
            for logprob in tokens["completion_logprobs"]:
                if logprob > self.logprob_threshold:
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
    action: FilterAction = "drop"
    penalty_reward: float = -1.0
    phase: FilterPhase = "post_advantage"

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
            action=config.resolved_action,
            penalty_reward=config.penalty_reward,
        )
    elif config.type == "repetition":
        return RepetitionFilter(
            name="repetition",
            window=config.window,
            logprob_threshold=math.log(config.prob_threshold),
            action=config.resolved_action,
            penalty_reward=config.penalty_reward,
        )
    elif config.type == "zero_advantage":
        return ZeroAdvantageFilter(
            name="zero_advantage",
            action=config.resolved_action,
            penalty_reward=config.penalty_reward,
        )
    raise ValueError(f"Unknown filter type: {config.type}")


def setup_filters(configs: list[FilterConfig], vocab_size: int, *, kind: str) -> list[RolloutFilter]:
    """Create RolloutFilters from a list of filter configs."""
    filters = [setup_filter(config, vocab_size) for config in configs]
    if filters:
        get_logger().info(f"Configured {len(filters)} {kind} rollout filter(s):")
        for config, filt in zip(configs, filters):
            mode = _ACTION_LOG_NAMES[filt.action]
            params = ", ".join(f"{k}={v}" for k, v in config.model_dump().items())
            get_logger().info(f"  {mode} {filt.name} filter ({params})")
    return filters


def split_filters(filters: list[RolloutFilter]) -> tuple[list[RolloutFilter], list[RolloutFilter]]:
    """Split filters into ``(pre_advantage, post_advantage)`` phase lists."""
    return (
        [f for f in filters if f.phase == "pre_advantage"],
        [f for f in filters if f.phase == "post_advantage"],
    )


def penalize_reward(
    rollout: "TrainRollout", filter_name: str, penalty_reward: float, detection_index: int | None
) -> None:
    """Cap the rollout's reward at ``penalty_reward`` and record penalty metadata.

    Uses ``min(...)`` so the penalty is a cap: rewards already below
    ``penalty_reward`` are never improved. The original env reward is
    preserved in ``rollout.raw_reward`` (first penalty wins) and per-filter
    details in ``rollout.reward_penalties``.
    """
    raw_reward = rollout.reward
    penalized_reward = min(raw_reward, penalty_reward)
    if rollout.raw_reward is None:
        rollout.raw_reward = raw_reward
    rollout.raw["reward"] = penalized_reward
    rollout.reward_penalties[filter_name] = {
        "raw_reward": raw_reward,
        "penalized_reward": penalized_reward,
        "detection_index": detection_index,
    }


def apply_filters(filters: list[RolloutFilter], rollouts: list["TrainRollout"]) -> None:  # noqa: F821 (forward ref)
    """Flag ``TrainRollout``\\ s in place with per-filter detection + action.

    Each rollout's ``filter_results`` dict records per-filter detection bools;
    ``is_filtered`` is True iff a ``drop`` filter detected it. A ``penalize``
    filter caps the rollout's reward at its ``penalty_reward`` but leaves the
    rollout trainable. First matching filter wins per rollout within a call
    (no double-counting). Trajectory tokens are left untouched so the rollout
    can still contribute to baseline calculations and metric aggregation.

    Safe to call more than once on the same rollouts (e.g. once per filter
    phase): missing ``filter_results`` keys are initialized without wiping
    results, drops, or penalties recorded by an earlier call.
    """
    for rollout in rollouts:
        for filt in filters:
            rollout.filter_results.setdefault(filt.name, False)

    if not filters:
        return

    for rollout in rollouts:
        for filt in filters:
            result = filt.check(rollout)
            if result.detected:
                rollout.filter_results[filt.name] = True
                if filt.action == "drop":
                    rollout.is_filtered = True
                elif filt.action == "penalize":
                    penalize_reward(rollout, filt.name, filt.penalty_reward, result.detection_index)
                break
