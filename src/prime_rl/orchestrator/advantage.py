from typing import Callable, Literal

import torch
from beartype import beartype as typechecker
from jaxtyping import Float, jaxtyped
from torch import Tensor


@jaxtyped(typechecker=typechecker)
def compute_advantage_drgrpo(rewards: Float[Tensor, "group"]) -> Float[Tensor, "group"]:
    """
    Compute DR.GRPO advantages for a single group.
    For example:
    - `[0.0, 0.0, 1.0, 1.0]` -> `[-0.5, -0.5, 0.5, 0.5]`
    - `[0.0, 0.0, 0.0, 0.0]` -> `[0.0, 0.0, 0.0, 0.0]`
    - `[1.0, 1.0, 1.0, 1.0]` -> `[0.0, 0.0, 0.0, 0.0]`
    """
    return rewards - rewards.mean()


@jaxtyped(typechecker=typechecker)
def compute_advantage_drgrpo_negclipped(rewards: Float[Tensor, "group"]) -> Float[Tensor, "group"]:
    """
    Compute DR.GRPO advantages for a single group, but clips all negative advantages to zero.
    For example:
    - `[0.0, 0.0, 1.0, 1.0]` -> `[0.0, 0.0, 0.5, 0.5]`
    - `[0.0, 0.0, 0.0, 0.0]` -> `[0.0, 0.0, 0.0, 0.0]`
    - `[1.0, 1.0, 1.0, 1.0]` -> `[0.0, 0.0, 0.0, 0.0]`
    """
    return torch.maximum(rewards - rewards.mean(), torch.zeros_like(rewards))


AdvantageType = Literal["drgrpo", "drgrpo-negclipped"]

# Map of advantage types to their corresponding functions
_ADVANTAGE_REGISTRY: dict[AdvantageType, Callable[[Float[Tensor, "group"]], Float[Tensor, "group"]]] = {
    "drgrpo": compute_advantage_drgrpo,
    "drgrpo-negclipped": compute_advantage_drgrpo_negclipped,
}


def compute_advantages(rewards: list[float], samples_per_problem: int, advantage_type: AdvantageType) -> list[float]:
    advantages = []
    solve_none, solve_all = 0, 0
    problem_rewards = [rewards[i : i + samples_per_problem] for i in range(0, len(rewards), samples_per_problem)]
    compute_advantage = _ADVANTAGE_REGISTRY[advantage_type]
    for rewards in problem_rewards:
        rewards_tensor = torch.tensor(rewards)
        advantages_tensor = compute_advantage(rewards_tensor)
        assert len(advantages_tensor) == len(rewards_tensor)
        advantages.extend(advantages_tensor.tolist())
        if torch.all(rewards_tensor == 0):
            solve_none += 1
        if torch.all(rewards_tensor == 1):
            solve_all += 1
    solve_none_ratio = solve_none / len(problem_rewards)
    solve_all_ratio = solve_all / len(problem_rewards)
    effective_batch_size_ratio = 1 - solve_none_ratio - solve_all_ratio
    return advantages, solve_none_ratio, solve_all_ratio, effective_batch_size_ratio
