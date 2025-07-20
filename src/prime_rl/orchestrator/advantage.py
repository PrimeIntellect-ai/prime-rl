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


def compute_advantages(
    rewards: list[float], samples_per_problem: int, advantage_type: AdvantageType
) -> tuple[list[float], dict[str, float]]:
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
    solve_none, solve_all = 0, 0
    assert len(rewards) % samples_per_problem == 0
    problem_rewards = [rewards[i : i + samples_per_problem] for i in range(0, len(rewards), samples_per_problem)]
    compute_advantage = REGISTRY[advantage_type]
    for rewards in problem_rewards:
        rewards_tensor = torch.tensor(rewards)
        advantages_tensor = compute_advantage(rewards_tensor)
        assert len(advantages_tensor) == len(rewards_tensor)
        advantages.extend(advantages_tensor.tolist())
        if torch.all(rewards_tensor == 0):
            solve_none += 1 / len(problem_rewards)
        if torch.all(rewards_tensor == 1):
            solve_all += 1 / len(problem_rewards)
    assert len(rewards) == len(advantages)
    effective_batch_size = 1 - solve_none - solve_all
    advantage_stats = {"solve_none": solve_none, "solve_all": solve_all, "effective_batch_size": effective_batch_size}
    return advantages, advantage_stats
