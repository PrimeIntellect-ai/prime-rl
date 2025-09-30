from typing import Callable, Literal

import torch
from beartype import beartype as typechecker
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor


@jaxtyped(typechecker=typechecker)
def compute_advantage_drgrpo(rewards: Float[Tensor, "group"], _: Int[Tensor, "group"]) -> Float[Tensor, "group"]:
    """
    Computes DR.GRPO advantages for a single group.
    For example:
    - `[0.0, 0.0, 1.0, 1.0]` -> `[-0.5, -0.5, 0.5, 0.5]`
    - `[0.0, 0.0, 0.0, 0.0]` -> `[0.0, 0.0, 0.0, 0.0]`
    - `[1.0, 1.0, 1.0, 1.0]` -> `[0.0, 0.0, 0.0, 0.0]`
    """
    return rewards - rewards.mean()


@jaxtyped(typechecker=typechecker)
def compute_advantage_drgrpo_negclipped(
    rewards: Float[Tensor, "group"], _: Int[Tensor, "group"]
) -> Float[Tensor, "group"]:
    """
    Computes DR.GRPO advantages for a single group, but clips all negative advantages to zero.
    For example:
    - `[0.0, 0.0, 1.0, 1.0]` -> `[0.0, 0.0, 0.5, 0.5]`
    - `[0.0, 0.0, 0.0, 0.0]` -> `[0.0, 0.0, 0.0, 0.0]`
    - `[1.0, 1.0, 1.0, 1.0]` -> `[0.0, 0.0, 0.0, 0.0]`
    """
    return torch.maximum(rewards - rewards.mean(), torch.zeros_like(rewards))


@jaxtyped(typechecker=typechecker)
def compute_advantage_rloo(rewards: Float[Tensor, "group"], _: Int[Tensor, "group"]) -> Float[Tensor, "group"]:
    """
    Computes RLOO (rescaled leave-one-out) advantages for a single group by
    scaling the standard DR.GRPO advantage by the factor G / (G - 1).
    """
    group_size = rewards.shape[0]
    advantages = rewards - rewards.mean()
    return advantages * group_size / (group_size - 1)


@jaxtyped(typechecker=typechecker)
def compute_advantage_opo(
    rewards: Float[Tensor, "group"], response_lengths: Int[Tensor, "group"]
) -> Float[Tensor, "group"]:
    """
    Computes OPO advantages for a single group.
    The baseline is the *weighted* mean of rewards where each reward is
    weighted by the length of the corresponding model response (in tokens).
    """
    weights = response_lengths.to(dtype=rewards.dtype)
    baseline = (rewards * weights).sum() / weights.sum()
    return rewards - baseline


AdvantageType = Literal["drgrpo", "drgrpo-negclipped", "rloo", "opo"]

# Map of advantage types to their corresponding functions
REGISTRY: dict[AdvantageType, Callable[[Float[Tensor, "group"], Int[Tensor, "group"]], Float[Tensor, "group"]]] = {
    "drgrpo": compute_advantage_drgrpo,
    "drgrpo-negclipped": compute_advantage_drgrpo_negclipped,
    "rloo": compute_advantage_rloo,
    "opo": compute_advantage_opo,
}


def compute_advantages(
    rewards: list[float],
    completion_lengths: list[int],
    samples_per_problem: int,
    advantage_type: AdvantageType,
    keep_n_shortest: int | None = None,
) -> list[float]:
    """
    Computes advantages and statistics for logging from a flattened list of rewards for a given advantage type.

    Args:
        rewards: Flattened list of rewards where first `samples_per_problem` rewards are for the first problem
        samples_per_problem: Number of samples (and thus, rewards) per problem
        advantage_type: Type of advantage computation to use
        completion_lengths: List of completion lengths for each reward. Required for OPO advantage computation.

    Returns:
        List of advantages corresponding to the input rewards. If `keep_n_shortest` is set,
        advantages for rollouts outside the N shortest (per problem group) are set to 0.0.
    """
    advantages = []
    assert len(rewards) % samples_per_problem == 0
    all_group_rewards = [rewards[i : i + samples_per_problem] for i in range(0, len(rewards), samples_per_problem)]
    all_group_lengths = [
        completion_lengths[i : i + samples_per_problem] for i in range(0, len(completion_lengths), samples_per_problem)
    ]
    compute_advantage = REGISTRY[advantage_type]
    for group_rewards, group_lengths in zip(all_group_rewards, all_group_lengths):
        group_rewards_tensor = torch.tensor(group_rewards)
        group_lengths_tensor = torch.tensor(group_lengths)

        if keep_n_shortest is not None:
            k = min(max(keep_n_shortest, 0), group_rewards_tensor.numel())
            keep_indices = torch.topk(group_lengths_tensor, k, largest=False).indices

            kept_rewards = group_rewards_tensor.index_select(0, keep_indices)
            kept_lengths = group_lengths_tensor.index_select(0, keep_indices)
            kept_advantages = compute_advantage(kept_rewards, kept_lengths)
            assert kept_advantages.numel() == keep_indices.numel()

            group_advantages_tensor = torch.zeros_like(group_rewards_tensor, dtype=kept_advantages.dtype)
            group_advantages_tensor.index_copy_(0, keep_indices, kept_advantages)
        else:
            group_advantages_tensor = compute_advantage(group_rewards_tensor, group_lengths_tensor)

        assert len(group_advantages_tensor) == len(group_rewards_tensor)
        advantages.extend(group_advantages_tensor.tolist())
    assert len(rewards) == len(advantages)
    return advantages
