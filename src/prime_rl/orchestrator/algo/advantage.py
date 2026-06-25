from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
from jaxtyping import Float
from torch import Tensor

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout

from prime_rl.configs.algorithm import (
    LengthPenaltyConfig,
    TokensLengthPenaltyConfig,
    TurnsLengthPenaltyConfig,
)
from prime_rl.orchestrator.utils import get_model_completion_len, get_tool_response_len

AdvantageFn = Callable[..., list[float | list[float]]]
"""Type for an advantage function.

Expected signature:
    def my_advantage(group: list[Rollout], **kwargs) -> list[float | list[float]]:
        ...

The function receives one finalized group — the same ``Rollout``\\ s the
``score_group`` hook sees (the trace in step coordinates, ``samples`` in merged
token coordinates) — and returns one value per rollout: a scalar (broadcast
over the rollout's completion tokens) or a per-token list aligned to them.
`apply_advantage_fn` writes each through ``Rollout.assign_advantages``.
"""


def default_advantage_fn(
    group: list["Rollout"],
    length_penalty: LengthPenaltyConfig | None = None,
) -> list[float]:
    """Default GRPO advantage for a single group: reward minus per-group baseline.

    `length_penalty` enables correctness-gated efficiency shaping over a per-rollout
    cost: tokens (weighted completion + tool-response) or trajectory turn count.
    """
    rewards = torch.tensor([v.reward for v in group], dtype=torch.float32)

    if isinstance(length_penalty, TokensLengthPenaltyConfig):
        w_c = length_penalty.completion_weight
        w_t = length_penalty.tool_response_weight
        costs = torch.tensor(
            [w_c * get_model_completion_len(v) + w_t * get_tool_response_len(v) for v in group],
            dtype=rewards.dtype,
        )
        return _efficiency_shaping(rewards, costs).tolist()
    if isinstance(length_penalty, TurnsLengthPenaltyConfig):
        costs = torch.tensor([v.num_turns for v in group], dtype=rewards.dtype)
        return _efficiency_shaping(rewards, costs).tolist()

    return (rewards - rewards.mean()).tolist()


def max_rl_advantage_fn(group: list["Rollout"]) -> list[float]:
    """MaxRL advantage for a single group (arXiv:2602.02710): reward minus the
    per-group mean, divided by that mean — equivalent to averaging score
    functions over successful rollouts only, which makes the policy gradient
    unbiased for the order-``group_size`` truncation of the maximum-likelihood
    objective instead of pass@1. Assumes non-negative (canonically binary)
    rewards; a group with mean reward <= 0 carries no signal and gets zero
    advantages (the zero-advantage filter drops it, matching the paper's
    no-success convention)."""
    rewards = torch.tensor([v.reward for v in group], dtype=torch.float32)
    mean = rewards.mean()
    if mean <= 0:
        return torch.zeros_like(rewards).tolist()
    return ((rewards - mean) / mean).tolist()


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


def apply_advantage_fn(group: list["Rollout"], advantage_fn: AdvantageFn) -> None:
    """Run an advantage function over one finished group and write each
    rollout's result through :meth:`Rollout.assign_advantages` (scalar
    broadcast or per-token list). The group-relative algorithms' ``score_group``
    hook delegates here."""
    for rollout, advs in zip(group, advantage_fn(group), strict=True):
        rollout.assign_advantages(advs)


def assign_group_norm(group: list["Rollout"], length_penalty: LengthPenaltyConfig | None) -> None:
    """Group-norm credit (the GRPO default), optionally length-shaped — shared
    by the algorithms whose ``score_group`` is plain group normalization."""
    apply_advantage_fn(group, lambda g: default_advantage_fn(g, length_penalty=length_penalty))
