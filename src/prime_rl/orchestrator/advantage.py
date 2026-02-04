import importlib
from dataclasses import dataclass
from typing import Any, Callable

import torch
from jaxtyping import Float, Int
from torch import Tensor

from prime_rl.orchestrator.config import AdvantageConfigType, CustomAdvantageConfig


@dataclass
class AdvantageInputs:
    """Inputs for advantage computation."""

    rewards: Float[Tensor, "num_problems samples_per_problem"]
    completion_lengths: Int[Tensor, "num_problems samples_per_problem"]


@dataclass
class AdvantageOutputs:
    """Outputs from advantage computation."""

    advantages: Float[Tensor, "num_problems samples_per_problem"]


AdvantageFn = Callable[..., AdvantageOutputs]
"""Type for an advantage function.

Expected signature:
    def my_advantage(inputs: AdvantageInputs, **kwargs) -> AdvantageOutputs:
        ...
"""


def _import_object(path: str) -> Any:
    """Import an object from a dotted path."""
    module_path, _, name = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, name)


def grpo_advantage(inputs: AdvantageInputs, length_weighted_mean: bool = False) -> AdvantageOutputs:
    """Default GRPO advantage: reward minus per-problem baseline."""
    if length_weighted_mean:
        baseline = (inputs.rewards * inputs.completion_lengths).sum(
            dim=1, keepdim=True
        ) / inputs.completion_lengths.sum(dim=1, keepdim=True)
    else:
        baseline = inputs.rewards.mean(dim=1, keepdim=True)

    return AdvantageOutputs(advantages=inputs.rewards - baseline)


def setup_advantage_fn(config: AdvantageConfigType) -> AdvantageFn:
    """Setup advantage function from config."""
    if isinstance(config, CustomAdvantageConfig):
        custom_fn = _import_object(config.path)
        kwargs = config.kwargs

        def advantage_fn(inputs: AdvantageInputs) -> AdvantageOutputs:
            return custom_fn(inputs, **kwargs)

        return advantage_fn

    def advantage_fn(inputs: AdvantageInputs) -> AdvantageOutputs:
        return grpo_advantage(inputs, length_weighted_mean=config.length_weighted_mean)

    return advantage_fn


def compute_advantages(
    rewards: list[float],
    completion_lengths: list[int],
    samples_per_problem: int,
    advantage_config: AdvantageConfigType | None,
) -> list[float]:
    """
    Computes advantages from a flattened list of rewards, grouped by problem.

    Args:
        rewards: Flattened list of rewards where first `samples_per_problem` rewards are for the first problem
        completion_lengths: List of completion lengths for each reward
        samples_per_problem: Number of samples (and thus, rewards) per problem
        advantage_config: Configuration for advantage computation (AdvantageConfig or CustomAdvantageConfig)
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
