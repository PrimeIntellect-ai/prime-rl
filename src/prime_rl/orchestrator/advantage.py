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
    lengths = torch.tensor(completion_lengths).view(-1, samples_per_problem)
    if advantage_config.length_weighted_mean:
        baseline = (rewards * lengths).sum(dim=1, keepdim=True) / lengths.sum(dim=1, keepdim=True)
    else:
        baseline = rewards.mean(dim=1, keepdim=True)
    return (rewards - baseline).flatten().tolist()


def compute_advantages_multi_reward(
    metrics: list[dict[str, float]],
    reward_keys: list[str],
    samples_per_problem: int,
    advantage_config: AdvantageConfig | None,
    reward_weights: list[float] | None = None,
) -> list[float]:
    """
    Computes advantages from multiple reward signals with per-reward normalization.

    Each reward is normalized independently within its problem group (mean and std),
    then the normalized values are combined via weighted sum. Optionally applies
    batch-wise normalization at the end.

    Args:
        metrics: List of metric dicts, one per sample. Each dict contains reward values
                 keyed by reward name (e.g., {"correct_answer": 1.0, "length_reward": 0.0})
        reward_keys: List of metric keys to use as reward signals
        samples_per_problem: Number of samples per problem (group size)
        advantage_config: Configuration for advantage computation
        reward_weights: Weights for each reward when summing. If None, uses equal weights (1.0).

    Returns:
        List of advantage values, one per sample
    """
    num_rewards = len(reward_keys)

    # Extract rewards into tensor: [num_samples, num_rewards]
    reward_values = torch.tensor([[m.get(k, 0.0) for k in reward_keys] for m in metrics], dtype=torch.float32)

    # Reshape to [num_problems, samples_per_problem, num_rewards]
    reward_values = reward_values.view(-1, samples_per_problem, num_rewards)

    # Per-reward normalization within each problem group
    # mean/std computed over samples_per_problem dimension (dim=1)
    mean_per_reward = reward_values.mean(dim=1, keepdim=True)  # [P, 1, R]
    std_per_reward = reward_values.std(dim=1, keepdim=True)  # [P, 1, R]

    # Get epsilon from config
    eps = advantage_config.std_eps if advantage_config else 1e-8

    # Normalize each reward: (r - mean) / (std + eps)
    normalized = (reward_values - mean_per_reward) / (std_per_reward + eps)  # [P, S, R]

    # Weighted sum across rewards
    if reward_weights is not None:
        weights = torch.tensor(reward_weights, dtype=torch.float32)  # [R]
        advantages = (normalized * weights).sum(dim=-1)  # [P, S]
    else:
        advantages = normalized.sum(dim=-1)  # [P, S]

    # Batch-wise normalization (optional but recommended)
    if advantage_config and advantage_config.batch_normalize:
        batch_mean = advantages.mean()
        batch_std = advantages.std()
        advantages = (advantages - batch_mean) / (batch_std + eps)

    return advantages.flatten().tolist()
