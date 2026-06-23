from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Sequence

import torch
import verifiers as vf

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import TrainRollout

from prime_rl.configs.orchestrator import (
    AdvantageConfig,
    CustomAdvantageConfig,
    LinearLengthPenaltyConfig,
)
from prime_rl.orchestrator.utils import get_model_completion_len
from prime_rl.utils.utils import import_object


@dataclass
class AdvantageInputs:
    """Inputs for advantage computation of a single group (one example × N rollouts)."""

    rollouts: list[vf.RolloutOutput]


@dataclass
class AdvantageOutputs:
    """Outputs from advantage computation of a single group."""

    advantages: list[float]
    length_penalty_annotations: list[dict[str, Any]] | None = None


@dataclass
class LengthPenaltyResult:
    """Centered linear length penalty components for one advantage group."""

    penalties: list[float]
    aux: list[float]
    costs: list[float]
    weights: list[float]


AdvantageFn = Callable[..., AdvantageOutputs]
"""Type for an advantage function.

Expected signature:
    def my_advantage(inputs: AdvantageInputs, **kwargs) -> AdvantageOutputs:
        ...

The function receives a single group and returns a list of advantages with one
entry per rollout. `assign_advantages` calls it on one already-grouped cohort.
"""


def centered_linear_length_penalty(
    *,
    lengths: Sequence[float],
    max_seq_len: int,
    coef: float,
    scale: float,
    weights: Sequence[float],
    truncated: Sequence[bool] | None = None,
) -> LengthPenaltyResult:
    """Compute linear length costs and their group-centered advantage delta.

    ``weights`` is the estimator-specific eligibility/strength vector. A
    truncated row pays max cost regardless of the observed emitted token count.
    """
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive when length_penalty is enabled, got {max_seq_len}")
    if len(lengths) != len(weights):
        raise ValueError(f"lengths/weights length mismatch: {len(lengths)} != {len(weights)}")
    if truncated is None:
        truncated = [False] * len(lengths)
    if len(lengths) != len(truncated):
        raise ValueError(f"lengths/truncated length mismatch: {len(lengths)} != {len(truncated)}")

    costs = [
        1.0 if is_truncated else min(max(float(length), 0.0) / max_seq_len, 1.0)
        for length, is_truncated in zip(lengths, truncated)
    ]
    weight_values = [float(weight) for weight in weights]
    penalties = [coef * float(scale) * weight * cost for weight, cost in zip(weight_values, costs)]
    center = sum(penalties) / len(penalties) if penalties else 0.0
    aux = [center - penalty for penalty in penalties]
    return LengthPenaltyResult(penalties=penalties, aux=aux, costs=costs, weights=weight_values)


def length_penalty_annotations(
    *,
    result: LengthPenaltyResult,
    base_advantages: Sequence[float],
    adjusted_advantages: Sequence[float],
) -> list[dict[str, Any]]:
    if len(result.penalties) != len(base_advantages) or len(result.penalties) != len(adjusted_advantages):
        raise ValueError("length penalty annotations require one base and adjusted advantage per penalty")

    annotations: list[dict[str, Any]] = []
    for penalty, base, adjusted, cost, weight in zip(
        result.penalties,
        base_advantages,
        adjusted_advantages,
        result.costs,
        result.weights,
    ):
        sign_flipped = (base < 0.0 < adjusted) or (base > 0.0 > adjusted)
        annotations.append(
            {
                "eligible": weight > 0.0,
                "penalty": penalty,
                "aux": adjusted - base,
                "cost": cost,
                "sign_flipped": sign_flipped,
            }
        )
    return annotations


def default_advantage_fn(
    inputs: AdvantageInputs,
    length_penalty: LinearLengthPenaltyConfig | None = None,
    max_seq_len: int | None = None,
    length_weighted_baseline: bool = False,
) -> AdvantageOutputs:
    """Default GRPO advantage for a single group: reward minus per-group baseline.

    ``length_penalty`` subtracts ``coef * pass_rate * (completion tokens / max_seq_len)``
    from each reward (``pass_rate`` = group mean reward), optionally gated to correct
    (``reward == 1``) rollouts. ``length_weighted_baseline`` uses the token-length-weighted
    mean reward as the baseline instead of the plain mean.
    """
    rewards = torch.tensor([r["reward"] for r in inputs.rollouts], dtype=torch.float32)
    lengths = torch.tensor([get_model_completion_len(r) for r in inputs.rollouts], dtype=rewards.dtype)
    base_rewards = rewards.clone()
    length_penalty_result: LengthPenaltyResult | None = None

    if length_penalty is not None:
        if max_seq_len is None:
            raise ValueError("max_seq_len is required when length_penalty is enabled")
        weights = rewards.tolist() if length_penalty.gate_by_correctness else [1.0] * len(inputs.rollouts)
        length_penalty_result = centered_linear_length_penalty(
            lengths=lengths.tolist(),
            max_seq_len=max_seq_len,
            coef=length_penalty.coef,
            scale=float(rewards.mean()),
            weights=weights,
            truncated=[bool(r.get("is_truncated", False)) for r in inputs.rollouts],
        )
        penalty = torch.tensor(length_penalty_result.penalties, dtype=rewards.dtype)
        rewards = rewards - penalty

    if length_weighted_baseline:
        baseline = (lengths * rewards).sum() / lengths.sum()
        base_baseline = (lengths * base_rewards).sum() / lengths.sum()
    else:
        baseline = rewards.mean()
        base_baseline = base_rewards.mean()
    advantages = (rewards - baseline).tolist()
    annotations = None
    if length_penalty_result is not None:
        annotations = length_penalty_annotations(
            result=length_penalty_result,
            base_advantages=(base_rewards - base_baseline).tolist(),
            adjusted_advantages=advantages,
        )
    return AdvantageOutputs(advantages=advantages, length_penalty_annotations=annotations)


def maxrl_advantage_fn(inputs: AdvantageInputs, eps: float = 1e-8) -> AdvantageOutputs:
    """MaxRL advantage for binary verifier rewards.

    Implements the on-policy MaxRL estimator from Tajwar et al. (2026): normalize
    rewards by the group mean reward and drop all-zero groups. With fixed group
    size, the omitted 1/N factor is a global learning-rate scale.
    """
    rewards = torch.tensor([r["reward"] for r in inputs.rollouts], dtype=torch.float32)
    mean_reward = rewards.mean()
    has_success = mean_reward > 0
    advantages = rewards / mean_reward.clamp_min(eps) - 1
    return AdvantageOutputs(advantages=torch.where(has_success, advantages, torch.zeros_like(rewards)).tolist())


def reward_advantage_fn(inputs: AdvantageInputs) -> AdvantageOutputs:
    """Raw reward advantage for reward-weighted REINFORCE objectives."""
    return AdvantageOutputs(advantages=[r["reward"] for r in inputs.rollouts])


def setup_advantage_fn(config: AdvantageConfig, max_seq_len: int | None = None) -> AdvantageFn:
    """Setup advantage function from config."""
    if isinstance(config, CustomAdvantageConfig):
        custom_fn = import_object(config.import_path)
        kwargs = config.kwargs

        def advantage_fn(inputs: AdvantageInputs) -> AdvantageOutputs:
            return custom_fn(inputs, **kwargs)

        return advantage_fn

    def advantage_fn(inputs: AdvantageInputs) -> AdvantageOutputs:
        return default_advantage_fn(
            inputs,
            length_penalty=config.length_penalty,
            max_seq_len=max_seq_len,
            length_weighted_baseline=config.length_weighted_baseline,
        )

    return advantage_fn


def assign_advantages(
    rollouts: list["TrainRollout"],  # noqa: F821 (forward ref)
    advantage_fn: AdvantageFn | None,
) -> None:
    """Compute and assign advantages for one finished group of rollouts
    (``TrainSink.process_group`` hands in a single group's surviving rollouts).
    ``advantage_fn=None`` is the trivial case (advantage = reward); a custom
    ``advantage_fn`` receives the raw ``vf.RolloutOutput``\\ s via
    ``AdvantageInputs.rollouts``.
    """
    if advantage_fn is None:
        for rollout in rollouts:
            rollout.advantage = rollout.reward
        return
    result = advantage_fn(AdvantageInputs(rollouts=[r.raw for r in rollouts]))
    if result.length_penalty_annotations is not None and len(result.length_penalty_annotations) != len(
        result.advantages
    ):
        raise ValueError("length penalty annotations length must match advantages length")
    for idx, (rollout, advantage) in enumerate(zip(rollouts, result.advantages)):
        rollout.advantage = advantage
        if result.length_penalty_annotations is not None:
            rollout.raw["length_penalty"] = result.length_penalty_annotations[idx]
