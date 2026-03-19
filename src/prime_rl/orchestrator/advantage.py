from dataclasses import dataclass
from typing import Callable

import torch
from jaxtyping import Float, Int
from torch import Tensor

from prime_rl.configs.orchestrator import AdvantageConfig, CustomAdvantageConfig
from prime_rl.utils.utils import import_object


@dataclass
class AdvantageInputs:
    """Inputs for advantage computation."""

    rewards: Float[Tensor, "num_problems rollouts_per_example"]
    completion_lengths: Int[Tensor, "num_problems rollouts_per_example"]


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
    length_shaping_alpha: float | None = None,
) -> AdvantageOutputs:
    """Default GRPO advantage: reward minus per-problem baseline."""
    rewards = inputs.rewards

    if length_shaping_alpha is not None:
        completion_lengths = inputs.completion_lengths.to(dtype=rewards.dtype)
        lengths_normalized = completion_lengths / completion_lengths.mean(dim=1, keepdim=True)
        length_shaping = (1 + length_shaping_alpha * lengths_normalized) ** -1
        rewards = rewards * length_shaping
    baseline = rewards.mean(dim=1, keepdim=True)

    return AdvantageOutputs(advantages=rewards - baseline)


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
            length_shaping_alpha=config.length_shaping_alpha,
        )

    return advantage_fn


def compute_advantages(
    rewards: list[float],
    completion_lengths: list[int],
    samples_per_problem: int,
    advantage_config: AdvantageConfig | None,
) -> list[float]:
    """
    Computes advantages from a flattened list of rewards, grouped by problem.

    Args:
        rewards: Flattened list of rewards where first `samples_per_problem` rewards are for the first problem
        completion_lengths: List of completion lengths for each reward
        samples_per_problem: Number of samples (and thus, rewards) per problem
        advantage_config: Configuration for advantage computation (DefaultAdvantageConfig or CustomAdvantageConfig)
    """
    if not advantage_config:
        return rewards

    advantage_fn = setup_advantage_fn(advantage_config)

    inputs = AdvantageInputs(
        rewards=torch.tensor(rewards).view(-1, samples_per_problem),
        completion_lengths=torch.tensor(completion_lengths).view(-1, samples_per_problem),
    )

    result = advantage_fn(inputs)
    return result.advantages.flatten().tolist()
