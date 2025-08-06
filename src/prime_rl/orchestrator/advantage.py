from typing import Callable, Literal

import torch
from beartype import beartype as typechecker
from jaxtyping import Float, jaxtyped
from torch import Tensor


@jaxtyped(typechecker=typechecker)
def compute_advantage_drgrpo(rewards: Float[Tensor, "group"]) -> Float[Tensor, "group"]:
    """
    Computes DR.GRPO advantages for a single group.
    For example:
    - `[0.0, 0.0, 1.0, 1.0]` -> `[-0.5, -0.5, 0.5, 0.5]`
    - `[0.0, 0.0, 0.0, 0.0]` -> `[0.0, 0.0, 0.0, 0.0]`
    - `[1.0, 1.0, 1.0, 1.0]` -> `[0.0, 0.0, 0.0, 0.0]`
    """
    return rewards - rewards.mean()


@jaxtyped(typechecker=typechecker)
def compute_advantage_drgrpo_negclipped(rewards: Float[Tensor, "group"]) -> Float[Tensor, "group"]:
    """
    Computes DR.GRPO advantages for a single group, but clips all negative advantages to zero.
    For example:
    - `[0.0, 0.0, 1.0, 1.0]` -> `[0.0, 0.0, 0.5, 0.5]`
    - `[0.0, 0.0, 0.0, 0.0]` -> `[0.0, 0.0, 0.0, 0.0]`
    - `[1.0, 1.0, 1.0, 1.0]` -> `[0.0, 0.0, 0.0, 0.0]`
    """
    return torch.maximum(rewards - rewards.mean(), torch.zeros_like(rewards))


AdvantageType = Literal["drgrpo", "drgrpo-negclipped"]

# Map of advantage types to their corresponding functions
REGISTRY: dict[AdvantageType, Callable[[Float[Tensor, "group"]], Float[Tensor, "group"]]] = {
    "drgrpo": compute_advantage_drgrpo,
    "drgrpo-negclipped": compute_advantage_drgrpo_negclipped,
}


def compute_advantages(rewards: list[float], samples_per_problem: int, advantage_type: AdvantageType) -> list[float]:
    """
    Computes advantages and statistics for logging from a flattened list of rewards for a given advantage type.

    Args:
        rewards: Flattened list of rewards where first `samples_per_problem` rewards are for the first problem
        samples_per_problem: Number of samples (and thus, rewards) per problem
        advantage_type: Type of advantage computation to use

    Returns:
        Tuple of (advantages, advantage_stats)
    """
    advantages = []
    assert len(rewards) % samples_per_problem == 0
    all_group_rewards = [rewards[i : i + samples_per_problem] for i in range(0, len(rewards), samples_per_problem)]
    compute_advantage = REGISTRY[advantage_type]
    for group_rewards in all_group_rewards:
        group_rewards_tensor = torch.tensor(group_rewards).float()
        group_advantages_tensor = compute_advantage(group_rewards_tensor)
        assert len(group_advantages_tensor) == len(group_rewards_tensor)
        advantages.extend(group_advantages_tensor.tolist())
    assert len(rewards) == len(advantages)
    return advantages
