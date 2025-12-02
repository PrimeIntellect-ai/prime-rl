import torch

from prime_rl.orchestrator.config import AdvantageConfig


def compute_advantages(
    rewards: list[float],
    completion_lengths: list[int],
    samples_per_problem: int,
    advantage_config: AdvantageConfig | None,
) -> list[float]:
    """
    Computes advantages from a flattened list of rewards.

    Args:
        rewards: Flattened list of rewards where first `samples_per_problem` rewards are for the first problem
        completion_lengths: List of completion lengths for each reward
        samples_per_problem: Number of samples per problem
        advantage_config: Configuration for advantage computation
    Returns:
        List of advantages
    """
    if not advantage_config:
        return rewards

    rewards = torch.tensor(rewards).view(-1, samples_per_problem)
    lengths = torch.tensor(completion_lengths).view(-1, samples_per_problem)

    if advantage_config.length_weighted_mean:
        baseline = (rewards * lengths).sum(dim=1, keepdim=True) / lengths.sum(dim=1, keepdim=True)
    else:
        baseline = rewards.mean(dim=1, keepdim=True)

    return (rewards - baseline).flatten().tolist()
