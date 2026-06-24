from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch

if TYPE_CHECKING:
    import verifiers.v1 as vf

    from prime_rl.orchestrator.types import Rollout

from prime_rl.configs.orchestrator import (
    AdvantageConfig,
    CustomAdvantageConfig,
    LinearLengthPenaltyConfig,
)
from prime_rl.utils.utils import import_object


@dataclass
class AdvantageInputs:
    """Inputs for advantage computation of a single group (one example × N rollouts)."""

    rollouts: list[vf.Trace]


@dataclass
class AdvantageOutputs:
    """Outputs from advantage computation of a single group."""

    advantages: list[float]


AdvantageFn = Callable[..., AdvantageOutputs]
"""Type for an advantage function.

Expected signature:
    def my_advantage(inputs: AdvantageInputs, **kwargs) -> AdvantageOutputs:
        ...

The function receives a single group and returns a list of advantages with one
entry per rollout. `assign_advantages` calls it on one already-grouped cohort.
"""


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
    rewards = torch.tensor([r.reward for r in inputs.rollouts], dtype=torch.float32)
    lengths = torch.tensor([r.completion_len for r in inputs.rollouts], dtype=rewards.dtype)

    if length_penalty is not None:
        if max_seq_len is None:
            raise ValueError("max_seq_len is required when length_penalty is enabled")
        penalty = length_penalty.coef * rewards.mean() * (lengths / max_seq_len)
        if length_penalty.gate_by_correctness:
            penalty = penalty * rewards
        rewards = rewards - penalty

    if length_weighted_baseline:
        baseline = (lengths * rewards).sum() / lengths.sum()
    else:
        baseline = rewards.mean()
    return AdvantageOutputs(advantages=(rewards - baseline).tolist())


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
    rollouts: list[Rollout],
    advantage_fn: AdvantageFn | None,
) -> None:
    """Compute and assign advantages for one finished group of rollouts
    (``TrainSink.process_group`` hands in a single group's surviving rollouts).
    ``advantage_fn=None`` is the trivial case (advantage = reward); a custom
    ``advantage_fn`` receives the ``vf.Trace``\\ s via ``AdvantageInputs.rollouts``.
    """
    if advantage_fn is None:
        for rollout in rollouts:
            rollout.advantage = rollout.reward
        return
    result = advantage_fn(AdvantageInputs(rollouts=[r for r in rollouts]))
    for rollout, advantage in zip(rollouts, result.advantages):
        rollout.advantage = advantage
