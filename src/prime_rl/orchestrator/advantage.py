from dataclasses import dataclass
from typing import Callable, Literal

import torch
import verifiers as vf
from jaxtyping import Float, Int
from torch import Tensor

from prime_rl.configs.orchestrator import AdvantageConfig, CustomAdvantageConfig
from prime_rl.orchestrator.vf_utils import get_model_completion_len, get_num_turns
from prime_rl.utils.utils import import_object


ShapingMetric = Literal["length", "num_turns"]


@dataclass
class AdvantageInputs:
    """Inputs for advantage computation."""

    rewards: Float[Tensor, "num_problems rollouts_per_example"]
    completion_lengths: Int[Tensor, "num_problems rollouts_per_example"]
    num_turns: Int[Tensor, "num_problems rollouts_per_example"] | None = None


@dataclass
class AdvantageOutputs:
    """Outputs from advantage computation."""

    advantages: Float[Tensor, "num_problems rollouts_per_example"]


AdvantageFn = Callable[..., AdvantageOutputs]
"""Type for an advantage function.

Expected signature:
    def my_advantage(inputs: AdvantageInputs, **kwargs) -> AdvantageOutputs:
        ...
"""


def default_advantage_fn(
    inputs: AdvantageInputs,
    shaping_metric: ShapingMetric | None = None,
) -> AdvantageOutputs:
    """Default GRPO advantage: reward minus per-problem baseline."""
    rewards = inputs.rewards

    if shaping_metric == "length":
        costs = inputs.completion_lengths.to(dtype=rewards.dtype)
        return AdvantageOutputs(advantages=_efficiency_shaping(rewards, costs))
    if shaping_metric == "num_turns":
        assert inputs.num_turns is not None, "num_turns required for num_turns shaping"
        costs = inputs.num_turns.to(dtype=rewards.dtype)
        return AdvantageOutputs(advantages=_efficiency_shaping(rewards, costs))

    baseline = rewards.mean(dim=1, keepdim=True)
    return AdvantageOutputs(advantages=rewards - baseline)


def _efficiency_shaping(
    rewards: Float[Tensor, "num_problems rollouts_per_example"],
    costs: Float[Tensor, "num_problems rollouts_per_example"],
) -> Float[Tensor, "num_problems rollouts_per_example"]:
    """Correctness-gated efficiency shaping with bounded advantages.

    Shapes rewards with a bounded efficiency bonus before standard GRPO subtraction,
    preserving zero-mean advantages per group. `costs` is a per-rollout cost (e.g.,
    completion length in tokens or number of turns).

    Correct rollouts get reward amplified by up to 2x based on relative efficiency.
    Incorrect rollouts are untouched. Lower-cost correct rollouts get higher advantage.
    """
    max_reward = rewards.max(dim=1, keepdim=True).values
    correct_mask = rewards >= max_reward
    num_correct = correct_mask.sum(dim=1, keepdim=True)

    # No shaping when max reward is 0 — no correct rollouts to differentiate
    has_correct = max_reward > 0

    # Mean cost of correct rollouts per problem
    correct_costs = costs * correct_mask
    mean_correct_cost = correct_costs.sum(dim=1, keepdim=True) / num_correct.clamp(min=1)

    # Bounded efficiency bonus: [0, 1], positive for below-average cost, zero for above
    bonus = (1 - costs / mean_correct_cost).clamp(0, 1)

    # Shape rewards: correct rollouts amplified by up to 2x, incorrect untouched
    shaped_rewards = rewards * (1 + bonus * correct_mask)
    baseline = shaped_rewards.mean(dim=1, keepdim=True)

    shaped = shaped_rewards - baseline
    unshaped = rewards - rewards.mean(dim=1, keepdim=True)
    return torch.where(has_correct, shaped, unshaped)


def setup_advantage_fn(config: AdvantageConfig) -> AdvantageFn:
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
            shaping_metric=config.shaping_metric,
        )

    return advantage_fn


def compute_advantages(
    rollouts: list[vf.RolloutOutput],
    samples_per_problem: int,
    advantage_config: AdvantageConfig | None,
) -> None:
    """
    Computes advantages from rollouts, grouped by problem.
    Stores advantages in-place on the rollouts.

    Args:
        rollouts: List of rollouts to store advantages on
        samples_per_problem: Number of samples (and thus, rewards) per problem
        advantage_config: Configuration for advantage computation (DefaultAdvantageConfig or CustomAdvantageConfig)
    """
    rewards = [r["reward"] for r in rollouts]

    if not advantage_config:
        for rollout, reward in zip(rollouts, rewards):
            rollout["advantage"] = reward
        return

    advantage_fn = setup_advantage_fn(advantage_config)
    completion_lengths = [get_model_completion_len(r) for r in rollouts]
    num_turns = [get_num_turns(r) for r in rollouts]

    inputs = AdvantageInputs(
        rewards=torch.tensor(rewards).view(-1, samples_per_problem),
        completion_lengths=torch.tensor(completion_lengths).view(-1, samples_per_problem),
        num_turns=torch.tensor(num_turns).view(-1, samples_per_problem),
    )

    result = advantage_fn(inputs)
    advantages = result.advantages.flatten().tolist()

    for rollout, advantage in zip(rollouts, advantages):
        rollout["advantage"] = advantage
