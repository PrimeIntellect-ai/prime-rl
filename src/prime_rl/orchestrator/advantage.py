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

    rewards: Float[Tensor, "num_groups group_size"]
    completion_lengths: Int[Tensor, "num_groups group_size"]


@dataclass
class AdvantageOutputs:
    """Outputs from advantage computation."""

    advantages: Float[Tensor, "num_groups group_size"]


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
    group_sizes: list[int],
    advantage_config: AdvantageConfig | None,
) -> list[float]:
    """
    Computes advantages from a flattened list of rewards, grouped by completed rollout group.

    Args:
        rewards: Flattened rewards in scheduler-emitted group order.
        completion_lengths: List of completion lengths for each reward
        group_sizes: Number of accepted rollouts in each completed group
        advantage_config: Configuration for advantage computation (DefaultAdvantageConfig or CustomAdvantageConfig)
    """
    if not advantage_config:
        return rewards
    if not rewards:
        return []
    if sum(group_sizes) != len(rewards) or len(completion_lengths) != len(rewards):
        raise ValueError("Rewards, completion_lengths, and group_sizes must describe the same flattened groups")

    advantage_fn = setup_advantage_fn(advantage_config)
    advantages: list[float] = []
    offset = 0
    for group_size in group_sizes:
        if group_size <= 0:
            continue
        next_offset = offset + group_size
        inputs = AdvantageInputs(
            rewards=torch.tensor(rewards[offset:next_offset]).view(1, group_size),
            completion_lengths=torch.tensor(completion_lengths[offset:next_offset]).view(1, group_size),
        )
        result = advantage_fn(inputs)
        advantages.extend(result.advantages.flatten().tolist())
        offset = next_offset
    return advantages
