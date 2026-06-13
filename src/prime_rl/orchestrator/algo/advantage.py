from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
from jaxtyping import Float
from torch import Tensor

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import TrainRollout

from prime_rl.configs.algorithm import (
    LengthPenaltyConfig,
    TokensLengthPenaltyConfig,
    TurnsLengthPenaltyConfig,
)
from prime_rl.orchestrator.utils import get_model_completion_len, get_tool_response_len

AdvantageFn = Callable[..., list[list[float]]]
"""Type for an advantage function.

Expected signature:
    def my_advantage(rollouts: list[TrainRollout], **kwargs) -> list[list[float]]:
        ...

The function receives one finalized group — the same ``TrainRollout``\\ s the
algorithm hooks see (``raw`` in step coordinates, ``samples`` in merged token
coordinates) — and returns per-token advantages: one list per rollout,
aligned to its samples' completion tokens. There is no scalar advantage
anywhere — uniform group credit goes through :func:`broadcast`.
`apply_advantage_fn` calls the function on one already-grouped cohort.
"""


def broadcast(rollouts: list["TrainRollout"], values: list[float]) -> list[list[float]]:
    """Spread one value per rollout over that rollout's completion tokens —
    scalar group credit (e.g. reward minus baseline) becomes a uniform
    per-token stream."""
    return [[float(v)] * sum(len(s.completion_ids) for s in r.samples) for r, v in zip(rollouts, values, strict=True)]


def default_advantage_fn(
    rollouts: list["TrainRollout"],
    length_penalty: LengthPenaltyConfig | None = None,
) -> list[list[float]]:
    """Default GRPO advantage for a single group: reward minus per-group baseline.

    `length_penalty` enables correctness-gated efficiency shaping over a per-rollout
    cost: tokens (weighted completion + tool-response) or trajectory turn count.
    """
    rewards = torch.tensor([r.reward for r in rollouts], dtype=torch.float32)

    if isinstance(length_penalty, TokensLengthPenaltyConfig):
        w_c = length_penalty.completion_weight
        w_t = length_penalty.tool_response_weight
        costs = torch.tensor(
            [w_c * get_model_completion_len(r.raw) + w_t * get_tool_response_len(r.raw) for r in rollouts],
            dtype=rewards.dtype,
        )
        return broadcast(rollouts, _efficiency_shaping(rewards, costs).tolist())
    if isinstance(length_penalty, TurnsLengthPenaltyConfig):
        costs = torch.tensor([len(r.raw["trajectory"]) for r in rollouts], dtype=rewards.dtype)
        return broadcast(rollouts, _efficiency_shaping(rewards, costs).tolist())

    return broadcast(rollouts, (rewards - rewards.mean()).tolist())


def max_rl_advantage_fn(rollouts: list["TrainRollout"]) -> list[list[float]]:
    """MaxRL advantage for a single group (arXiv:2602.02710): reward minus the
    per-group mean, divided by that mean — equivalent to averaging score
    functions over successful rollouts only, which makes the policy gradient
    unbiased for the order-``group_size`` truncation of the maximum-likelihood
    objective instead of pass@1. Assumes non-negative (canonically binary)
    rewards; a group with mean reward <= 0 carries no signal and gets zero
    advantages (the zero-advantage filter drops it, matching the paper's
    no-success convention)."""
    rewards = torch.tensor([r.reward for r in rollouts], dtype=torch.float32)
    mean = rewards.mean()
    if mean <= 0:
        return broadcast(rollouts, torch.zeros_like(rewards).tolist())
    return broadcast(rollouts, ((rewards - mean) / mean).tolist())


def _efficiency_shaping(
    rewards: Float[Tensor, "group_size"],
    costs: Float[Tensor, "group_size"],
) -> Float[Tensor, "group_size"]:
    """Correctness-gated efficiency shaping with bounded advantages.

    Shapes rewards with a bounded efficiency bonus before standard GRPO subtraction,
    preserving zero-mean advantages within the group. `costs` is a per-rollout cost
    (e.g., completion length in tokens or number of turns).

    Correct rollouts get reward amplified by up to 2x based on relative efficiency.
    Incorrect rollouts are untouched. Lower-cost correct rollouts get higher advantage.
    """
    max_reward = rewards.max()
    correct_mask = rewards >= max_reward
    num_correct = correct_mask.sum()

    # No shaping when max reward is 0 — no correct rollouts to differentiate
    if max_reward <= 0:
        return rewards - rewards.mean()

    # Mean cost of correct rollouts
    mean_correct_cost = (costs * correct_mask).sum() / num_correct.clamp(min=1)

    # Bounded efficiency bonus: [0, 1], positive for below-average cost, zero for above.
    # When mean_correct_cost is 0 (e.g. tool-only shaping with no harness metric, or
    # all-zero turn counts), no rollouts can be differentiated — fall back to no bonus.
    if mean_correct_cost <= 0:
        return rewards - rewards.mean()

    bonus = (1 - costs / mean_correct_cost).clamp(0, 1)

    # Shape rewards: correct rollouts amplified by up to 2x, incorrect untouched
    shaped_rewards = rewards * (1 + bonus * correct_mask)
    return shaped_rewards - shaped_rewards.mean()


def apply_advantage_fn(
    rollouts: list["TrainRollout"],
    advantage_fn: AdvantageFn | None,
) -> None:
    """Run an advantage function over one finished group of rollouts and
    write the resulting per-token streams (the algorithm's
    ``assign_advantages`` hands in one finalized group's survivors).
    ``advantage_fn=None`` is the trivial case (advantage = reward,
    broadcast); a custom ``advantage_fn`` receives the ``TrainRollout``\\ s
    themselves.
    """
    advantages = broadcast(rollouts, [r.reward for r in rollouts]) if advantage_fn is None else advantage_fn(rollouts)
    for rollout, advs in zip(rollouts, advantages, strict=True):
        rollout.advantages = advs


def assign_group_norm(rollouts: list["TrainRollout"], length_penalty: LengthPenaltyConfig | None) -> None:
    """Group-norm credit (the GRPO default), optionally length-shaped — shared
    by the algorithms whose ``assign_advantages`` is plain group normalization."""
    apply_advantage_fn(rollouts, lambda rollouts: default_advantage_fn(rollouts, length_penalty=length_penalty))
