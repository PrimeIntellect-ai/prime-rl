from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
from jaxtyping import Float
from torch import Tensor

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import RolloutView

from prime_rl.configs.algorithm import (
    LinearLengthPenaltyConfig,
    TokensLengthPenaltyConfig,
    TurnsLengthPenaltyConfig,
)
from prime_rl.orchestrator.utils import get_model_completion_len, get_tool_response_len

AdvantageFn = Callable[..., list[float | list[float]]]
"""Type for an advantage function.

Expected signature:
    def my_advantage(group: list[RolloutView], **kwargs) -> list[float | list[float]]:
        ...

The function receives one finalized group — the same ``RolloutView``\\ s the
``score_group`` hook sees (``raw`` in step coordinates, ``samples`` in merged
token coordinates) — and returns one value per rollout: a scalar (broadcast
over the rollout's completion tokens) or a per-token list aligned to them.
`apply_advantage_fn` writes each through ``RolloutView.assign_advantages``.
"""


def grpo_advantage(group: list["RolloutView"], length_weighted_baseline: bool = False) -> list[float]:
    """Plain GRPO advantage for a single group: reward minus the per-group
    baseline (DR-GRPO without std normalization).

    ``length_weighted_baseline`` uses the token-length-weighted mean reward
    (``sum(len_i * reward_i) / sum(len_i)``) as the baseline instead of the plain
    mean, centering advantages by per-token expected reward.
    """
    rewards = torch.tensor([v.reward for v in group], dtype=torch.float32)
    if length_weighted_baseline:
        lengths = torch.tensor([get_model_completion_len(v.raw) for v in group], dtype=rewards.dtype)
        baseline = (lengths * rewards).sum() / lengths.sum()
    else:
        baseline = rewards.mean()
    return (rewards - baseline).tolist()


def length_penalty_advantage(
    group: list["RolloutView"],
    config: LinearLengthPenaltyConfig,
    max_seq_len: int | None,
    length_weighted_baseline: bool = False,
) -> list[float]:
    """The linear length penalty as a standalone additive advantage term.

    Each rollout's penalty is ``coef * pass_rate * (completion tokens / max_seq_len)``
    (``pass_rate`` = group mean reward; optionally gated to correct rollouts), and
    this returns the group-centered negative penalty ``-(penalty_i - baseline)``.
    Summed onto :func:`grpo_advantage` it is *identical* to subtracting the penalty
    from each reward before centering — centering is linear, so
    ``center(reward - penalty) = center(reward) + center(-penalty)`` — provided both
    terms use the same baseline operator, hence ``length_weighted_baseline`` is
    threaded here too (it picks the plain vs token-length-weighted mean, matching
    :func:`grpo_advantage`).
    """
    if max_seq_len is None:
        raise ValueError("max_seq_len is required when the linear length penalty is enabled")
    rewards = torch.tensor([v.reward for v in group], dtype=torch.float32)
    lengths = torch.tensor([get_model_completion_len(v.raw) for v in group], dtype=rewards.dtype)
    penalty = config.coef * rewards.mean() * (lengths / max_seq_len)
    if config.gate_by_correctness:
        penalty = penalty * rewards
    baseline = (lengths * penalty).sum() / lengths.sum() if length_weighted_baseline else penalty.mean()
    return (baseline - penalty).tolist()


def efficiency_shaping_advantage(
    group: list["RolloutView"], config: TokensLengthPenaltyConfig | TurnsLengthPenaltyConfig
) -> list[float]:
    """Correctness-gated efficiency shaping (the ``tokens`` / ``turns`` length
    penalties) over a per-rollout cost: weighted completion + tool-response tokens,
    or trajectory turn count. Unlike :func:`length_penalty_advantage` this is not an
    additive term — it amplifies correct rewards (see :func:`_efficiency_shaping`) and
    returns the full advantage, so it replaces the GRPO baseline rather than summing
    with it.
    """
    rewards = torch.tensor([v.reward for v in group], dtype=torch.float32)
    if isinstance(config, TokensLengthPenaltyConfig):
        w_c, w_t = config.completion_weight, config.tool_response_weight
        costs = torch.tensor(
            [w_c * get_model_completion_len(v.raw) + w_t * get_tool_response_len(v.raw) for v in group],
            dtype=rewards.dtype,
        )
    else:
        costs = torch.tensor([len(v.raw["trajectory"]) for v in group], dtype=rewards.dtype)
    return _efficiency_shaping(rewards, costs).tolist()


def max_rl_advantage_fn(group: list["RolloutView"]) -> list[float]:
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


def apply_advantage_fn(group: list["RolloutView"], advantage_fn: AdvantageFn) -> None:
    """Run an advantage function over one finished group and write each
    rollout's result through :meth:`RolloutView.assign_advantages` (scalar
    broadcast or per-token list). The group-relative algorithms' ``score_group``
    hook delegates here."""
    for view, advs in zip(group, advantage_fn(group), strict=True):
        view.assign_advantages(advs)
