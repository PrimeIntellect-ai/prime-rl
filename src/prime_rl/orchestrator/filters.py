"""Orchestrator-side rollout filters for detecting degenerate generations.

Filters run after rollouts complete, inspecting token IDs and logprobs to
detect gibberish or repetition. Detection metrics are always tracked. Each
filter resolves to one of three actions:

- ``monitor``: only record detection metrics;
- ``drop``: detected rollouts are skipped entirely during training and are
  not sent to the trainer. Reward is kept as-is for baseline calculation;
- ``penalize``: detected rollouts stay trainable, but an explicit penalty
  strategy transforms their reward. Penalties must be applied before
  advantage computation to create negative policy-gradient signal —
  token/logprob based filters are ``pre_advantage`` phase so
  ``TrainSink.process_group`` runs them before ``assign_advantages``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal, Protocol, TypeAlias

import verifiers as vf

from prime_rl.configs.orchestrator import (
    CustomRewardPenaltyConfig,
    FilterAction,
    FilterActionConfig,
    FilterConfig,
    PenalizeFilterActionConfig,
    RewardPenaltyConfig,
    SetRewardPenaltyConfig,
)
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import import_object

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import TrainRollout


FilterPhase: TypeAlias = Literal["pre_advantage", "post_advantage"]


@dataclass
class FilterResult:
    detected: bool
    detection_index: int | None = None


@dataclass
class RewardPenaltyInputs:
    raw_reward: float
    rollout: vf.RolloutOutput
    filter_name: str
    detection_index: int | None = None


RewardPenaltyFn = Callable[[RewardPenaltyInputs], float]


class RolloutFilter(Protocol):
    name: str
    action: FilterAction
    phase: FilterPhase
    reward_penalty_fn: RewardPenaltyFn | None

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
    reward_penalty_fn: RewardPenaltyFn | None = None
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
    reward_penalty_fn: RewardPenaltyFn | None = None
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
    reward_penalty_fn: RewardPenaltyFn | None = None
    phase: FilterPhase = "post_advantage"

    def check(self, rollout: "TrainRollout") -> FilterResult:
        if rollout.advantage is not None and rollout.advantage == 0.0:
            return FilterResult(detected=True)
        return FilterResult(detected=False)


def setup_reward_penalty_fn(config: RewardPenaltyConfig) -> RewardPenaltyFn:
    if isinstance(config, SetRewardPenaltyConfig):

        def reward_penalty_fn(_inputs: RewardPenaltyInputs) -> float:
            return config.reward

        return reward_penalty_fn
    if isinstance(config, CustomRewardPenaltyConfig):
        custom_fn = import_object(config.import_path)
        kwargs = config.kwargs

        def reward_penalty_fn(inputs: RewardPenaltyInputs) -> float:
            return custom_fn(inputs, **kwargs)

        return reward_penalty_fn
    raise ValueError(f"Unknown reward penalty type: {config.type}")


def resolve_filter_action(config: FilterActionConfig) -> tuple[FilterAction, RewardPenaltyFn | None]:
    if isinstance(config, PenalizeFilterActionConfig):
        return config.type, setup_reward_penalty_fn(config.penalty)
    return config.type, None


def setup_filter(config: FilterConfig, vocab_size: int) -> RolloutFilter:
    """Create a RolloutFilter from a filter config."""
    action, reward_penalty_fn = resolve_filter_action(config.action)
    if config.type == "gibberish":
        return GibberishFilter(
            name="gibberish",
            token_id_threshold=config.token_id_threshold,
            logprob_threshold=-math.log(vocab_size) - config.logprob_offset,
            action=action,
            reward_penalty_fn=reward_penalty_fn,
        )
    elif config.type == "repetition":
        return RepetitionFilter(
            name="repetition",
            window=config.window,
            logprob_threshold=math.log(config.prob_threshold),
            action=action,
            reward_penalty_fn=reward_penalty_fn,
        )
    elif config.type == "zero_advantage":
        return ZeroAdvantageFilter(
            name="zero_advantage",
            action=action,
            reward_penalty_fn=reward_penalty_fn,
        )
    raise ValueError(f"Unknown filter type: {config.type}")


def setup_filters(configs: list[FilterConfig], vocab_size: int, *, kind: str) -> list[RolloutFilter]:
    """Create RolloutFilters from a list of filter configs."""
    filters = [setup_filter(config, vocab_size) for config in configs]
    if filters:
        get_logger().info(f"Configured {len(filters)} {kind} rollout filter(s):")
        for config, filt in zip(configs, filters):
            mode = {"monitor": "Monitoring", "drop": "Dropping", "penalize": "Penalizing"}[filt.action]
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
    rollout: "TrainRollout",
    filter_name: str,
    penalty_fn: RewardPenaltyFn,
    detection_index: int | None,
) -> None:
    """Transform the rollout's reward and record penalty metadata."""
    raw_reward = rollout.reward
    penalized_reward = float(
        penalty_fn(
            RewardPenaltyInputs(
                raw_reward=raw_reward,
                rollout=rollout.raw,
                filter_name=filter_name,
                detection_index=detection_index,
            )
        )
    )
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
    filter transforms the rollout's reward via its penalty strategy but leaves the
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
                    assert filt.reward_penalty_fn is not None
                    penalize_reward(rollout, filt.name, filt.reward_penalty_fn, result.detection_index)
                break
