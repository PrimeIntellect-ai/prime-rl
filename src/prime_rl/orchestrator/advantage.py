import torch
from jaxtyping import Float
from torch import Tensor
from prime_rl.utils.pydantic_config import BaseConfig

class AdvantageConfig(BaseConfig):
    global_std_norm: bool = False
    length_weighted_mean: bool = False
    leave_one_out: bool = False
    neg_clipped: bool = False


def compute_advantage(
    rewards: Float[Tensor, "group"],
    lengths: Float[Tensor, "group"],
    global_std: float,
    advantage_config: AdvantageConfig,
) -> Float[Tensor, "group"]:
    """
    Computes advantages for a single group.
    """
    group_size = rewards.shape[0]
    if advantage_config.length_weighted_mean:
        baseline = (rewards * lengths).sum() / lengths.sum()
    else:
        baseline = rewards.mean()
    advantages = rewards - baseline
    if advantage_config.leave_one_out:
        advantages = advantages * group_size / (group_size - 1)
    if advantage_config.neg_clipped:
        advantages = torch.maximum(advantages, torch.zeros_like(advantages))
    if advantage_config.global_std_norm:
        advantages = advantages / global_std
    return advantages


def compute_advantages(
    rewards: list[float],
    completion_lengths: list[int],
    samples_per_problem: int,
    advantage_config: AdvantageConfig,
) -> list[float]:
    """
    Computes advantages and statistics for logging from a flattened list of rewards for a given advantage type.

    Args:
        rewards: Flattened list of rewards where first `samples_per_problem` rewards are for the first problem
        samples_per_problem: Number of samples (and thus, rewards) per problem
        advantage_config: Configuration for advantage computation
        completion_lengths: List of completion lengths for each reward. Required for OPO advantage computation.
        global_std_norm: Whether to normalize the advantages by the global standard deviation of the rewards.
    Returns:
        Tuple of (advantages, advantage_stats)
    """
    advantages = []
    assert len(rewards) % samples_per_problem == 0
    all_group_rewards = [rewards[i : i + samples_per_problem] for i in range(0, len(rewards), samples_per_problem)]
    all_group_lengths = [
        completion_lengths[i : i + samples_per_problem] for i in range(0, len(completion_lengths), samples_per_problem)
    ]
    global_std = torch.tensor(rewards).std().item()
    for group_rewards, group_lengths in zip(all_group_rewards, all_group_lengths):
        group_rewards_tensor = torch.tensor(group_rewards)
        group_lengths_tensor = torch.tensor(group_lengths, dtype=group_rewards_tensor.dtype)
        group_advantages_tensor = compute_advantage(
            group_rewards_tensor, group_lengths_tensor, global_std, advantage_config
        )
        assert len(group_advantages_tensor) == len(group_rewards_tensor)
        advantages.extend(group_advantages_tensor.tolist())
    assert len(rewards) == len(advantages)
    return advantages
