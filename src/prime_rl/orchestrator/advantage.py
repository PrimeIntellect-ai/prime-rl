import torch

from prime_rl.orchestrator.config import AdvantageConfig


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
        advantage_config: Configuration for advantage computation
    """
    if not advantage_config:
        return rewards
    rewards = torch.tensor(rewards).view(-1, samples_per_problem)
    lengths = torch.tensor(completion_lengths, dtype=torch.float32).view(-1, samples_per_problem)
    if advantage_config.gr3_alpha:
        lengths_normalized = lengths / lengths.mean(dim=1, keepdim=True)
        length_shaping = (1 + advantage_config.gr3_alpha * lengths_normalized) ** -1
        rewards_corrected = rewards * length_shaping
        baseline = rewards_corrected.mean(dim=1, keepdim=True)
        return (rewards_corrected - baseline).flatten().tolist()
    elif advantage_config.length_weighted_mean:
        baseline = (rewards * lengths).sum(dim=1, keepdim=True) / lengths.sum(dim=1, keepdim=True)
    else:
        baseline = rewards.mean(dim=1, keepdim=True)
    return (rewards - baseline).flatten().tolist()
