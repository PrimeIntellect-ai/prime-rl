from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import torch
import verifiers as vf
from jaxtyping import Float
from torch import Tensor

from prime_rl.configs.orchestrator import (
    AdvantageConfig,
    CustomAdvantageConfig,
    LengthPenaltyConfig,
    TokensLengthPenaltyConfig,
    TurnsLengthPenaltyConfig,
)
from prime_rl.orchestrator.vf_utils import get_model_completion_len, get_num_turns, get_tool_response_len
from prime_rl.utils.utils import import_object


@dataclass
class AdvantageInputs:
    """Inputs for advantage computation of a single group (one example × N rollouts)."""

    rollouts: list[vf.RolloutOutput]


@dataclass
class AdvantageOutputs:
    """Outputs from advantage computation of a single group."""

    advantages: Float[Tensor, "group_size"]


AdvantageFn = Callable[..., AdvantageOutputs]
"""Type for an advantage function.

Expected signature:
    def my_advantage(inputs: AdvantageInputs, **kwargs) -> AdvantageOutputs:
        ...

The function receives a single group and returns a 1D advantage tensor of the
same length. `compute_advantages` calls it once per group.
"""


def default_advantage_fn(
    inputs: AdvantageInputs,
    length_penalty: LengthPenaltyConfig | None = None,
) -> AdvantageOutputs:
    """Default GRPO advantage for a single group: reward minus per-group baseline.

    `length_penalty` enables correctness-gated efficiency shaping over a per-rollout
    cost: tokens (weighted completion + tool-response) or trajectory turn count.
    """
    rewards = torch.tensor([r["reward"] for r in inputs.rollouts], dtype=torch.float32)

    if isinstance(length_penalty, TokensLengthPenaltyConfig):
        w_c = length_penalty.completion_weight
        w_t = length_penalty.tool_response_weight
        costs = torch.tensor(
            [w_c * get_model_completion_len(r) + w_t * get_tool_response_len(r) for r in inputs.rollouts],
            dtype=rewards.dtype,
        )
        return AdvantageOutputs(advantages=_efficiency_shaping(rewards, costs))
    if isinstance(length_penalty, TurnsLengthPenaltyConfig):
        costs = torch.tensor([get_num_turns(r) for r in inputs.rollouts], dtype=rewards.dtype)
        return AdvantageOutputs(advantages=_efficiency_shaping(rewards, costs))

    return AdvantageOutputs(advantages=rewards - rewards.mean())


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


def setup_advantage_fn(config: AdvantageConfig) -> AdvantageFn:
    """Setup advantage function from config."""
    if isinstance(config, CustomAdvantageConfig):
        custom_fn = import_object(config.import_path)
        kwargs = config.kwargs

        def advantage_fn(inputs: AdvantageInputs) -> AdvantageOutputs:
            return custom_fn(inputs, **kwargs)

        return advantage_fn

    def advantage_fn(inputs: AdvantageInputs) -> AdvantageOutputs:
        return default_advantage_fn(inputs, length_penalty=config.length_penalty)

    return advantage_fn


def compute_advantages(
    rollouts: list[vf.RolloutOutput],
    advantage_config: AdvantageConfig | None,
) -> None:
    """Computes advantages from rollouts, grouped by (env_name, example_id), and
    stores them in-place on the rollouts.

    `advantage_fn` is called once per group, so groups may have varying sizes
    (partial-group training drops failed rollouts rather than rescheduling them).
    """
    if not advantage_config:
        for rollout in rollouts:
            rollout["advantage"] = rollout["reward"]
        return

    advantage_fn = setup_advantage_fn(advantage_config)

    groups_by_example: dict[tuple[str, int], list[vf.RolloutOutput]] = defaultdict(list)
    for rollout in rollouts:
        groups_by_example[(rollout["env_name"], rollout["example_id"])].append(rollout)

    for group in groups_by_example.values():
        result = advantage_fn(AdvantageInputs(rollouts=group))
        for rollout, advantage in zip(group, result.advantages.tolist()):
            rollout["advantage"] = advantage
