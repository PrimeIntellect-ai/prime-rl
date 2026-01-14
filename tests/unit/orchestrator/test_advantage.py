import pytest

from prime_rl.orchestrator.advantage import compute_advantages, compute_advantages_multi_reward
from prime_rl.orchestrator.config import AdvantageConfig

# Single-reward tests


def test_compute_advantages_returns_rewards_when_no_config():
    rewards = [1.0, 0.0, 0.5, 0.5]
    result = compute_advantages(rewards, [10, 10, 10, 10], 2, None)
    assert result == rewards


def test_compute_advantages_subtracts_mean_within_groups():
    # 2 problems, 2 samples each
    rewards = [1.0, 0.0, 0.5, 0.5]
    config = AdvantageConfig()
    result = compute_advantages(rewards, [10, 10, 10, 10], 2, config)

    # Problem 1: mean=0.5, so [1.0-0.5, 0.0-0.5] = [0.5, -0.5]
    # Problem 2: mean=0.5, so [0.5-0.5, 0.5-0.5] = [0.0, 0.0]
    assert result == [0.5, -0.5, 0.0, 0.0]


def test_compute_advantages_length_weighted_mean():
    # 1 problem, 2 samples with different lengths
    rewards = [1.0, 0.0]
    lengths = [100, 10]  # First sample much longer
    config = AdvantageConfig(length_weighted_mean=True)
    result = compute_advantages(rewards, lengths, 2, config)

    # Weighted mean = (1.0*100 + 0.0*10) / (100+10) = 100/110 ≈ 0.909
    # Advantages: [1.0 - 0.909, 0.0 - 0.909] ≈ [0.091, -0.909]
    assert len(result) == 2
    assert abs(result[0] - 0.0909) < 0.001
    assert abs(result[1] - (-0.9091)) < 0.001


# Multi-reward tests


@pytest.fixture
def simple_metrics():
    """2 problems, 3 samples each, 2 rewards."""
    return [
        # Problem 1
        {"correct_answer": 1.0, "length_reward": 1.0},
        {"correct_answer": 1.0, "length_reward": 0.0},
        {"correct_answer": 0.0, "length_reward": 1.0},
        # Problem 2
        {"correct_answer": 0.0, "length_reward": 0.0},
        {"correct_answer": 1.0, "length_reward": 1.0},
        {"correct_answer": 0.0, "length_reward": 0.0},
    ]


@pytest.fixture
def uniform_metrics():
    """All samples have the same reward values (edge case for std=0)."""
    return [
        {"correct_answer": 1.0, "length_reward": 1.0},
        {"correct_answer": 1.0, "length_reward": 1.0},
        {"correct_answer": 1.0, "length_reward": 1.0},
    ]


def test_multi_reward_returns_correct_length(simple_metrics):
    config = AdvantageConfig()
    result = compute_advantages_multi_reward(
        simple_metrics,
        ["correct_answer", "length_reward"],
        3,
        config,
    )
    assert len(result) == 6


def test_multi_reward_batch_normalized_sum_is_zero(simple_metrics):
    config = AdvantageConfig(batch_normalize=True)
    result = compute_advantages_multi_reward(
        simple_metrics,
        ["correct_answer", "length_reward"],
        3,
        config,
    )
    assert abs(sum(result)) < 1e-5


def test_multi_reward_no_batch_normalization(simple_metrics):
    config = AdvantageConfig(batch_normalize=False)
    result = compute_advantages_multi_reward(
        simple_metrics,
        ["correct_answer", "length_reward"],
        3,
        config,
    )
    assert len(result) == 6
    assert all(isinstance(v, float) for v in result)


def test_multi_reward_handles_uniform_rewards(uniform_metrics):
    """When all rewards are the same, std=0, should not crash due to eps."""
    config = AdvantageConfig()
    result = compute_advantages_multi_reward(
        uniform_metrics,
        ["correct_answer", "length_reward"],
        3,
        config,
    )
    assert len(result) == 3
    assert all(abs(v) < 1e-5 for v in result)


def test_multi_reward_different_combinations_produce_different_advantages(simple_metrics):
    """Key property: different reward combos should not collapse to same advantage."""
    config = AdvantageConfig(batch_normalize=False)
    result = compute_advantages_multi_reward(
        simple_metrics,
        ["correct_answer", "length_reward"],
        3,
        config,
    )
    # First 3 samples (problem 1) should have different advantages
    problem1_advantages = result[:3]
    assert len(set(round(a, 6) for a in problem1_advantages)) > 1


def test_multi_reward_weighted_rewards(simple_metrics):
    config = AdvantageConfig(batch_normalize=False)

    result_equal = compute_advantages_multi_reward(
        simple_metrics,
        ["correct_answer", "length_reward"],
        3,
        config,
        reward_weights=None,
    )

    result_weighted = compute_advantages_multi_reward(
        simple_metrics,
        ["correct_answer", "length_reward"],
        3,
        config,
        reward_weights=[2.0, 1.0],
    )

    assert result_equal != result_weighted


def test_multi_reward_single_reward_key():
    """Multi-reward with single key should work."""
    metrics = [
        {"correct_answer": 1.0},
        {"correct_answer": 0.0},
        {"correct_answer": 1.0},
    ]
    config = AdvantageConfig()
    result = compute_advantages_multi_reward(
        metrics,
        ["correct_answer"],
        3,
        config,
    )
    assert len(result) == 3


def test_multi_reward_missing_metric_key_defaults_to_zero():
    """If a metric key is missing, it should default to 0.0."""
    metrics = [
        {"correct_answer": 1.0},  # Missing length_reward
        {"correct_answer": 0.0, "length_reward": 1.0},
        {"correct_answer": 1.0, "length_reward": 0.0},
    ]
    config = AdvantageConfig()
    result = compute_advantages_multi_reward(
        metrics,
        ["correct_answer", "length_reward"],
        3,
        config,
    )
    assert len(result) == 3


def test_multi_reward_std_eps_prevents_division_by_zero():
    """Verify std_eps is used to prevent division by zero."""
    uniform_metrics = [
        {"r": 1.0},
        {"r": 1.0},
    ]
    config = AdvantageConfig(std_eps=1e-8)
    result = compute_advantages_multi_reward(
        uniform_metrics,
        ["r"],
        2,
        config,
    )
    assert len(result) == 2


def test_multi_reward_no_config_returns_weighted_sum():
    """When advantage_config is None, return weighted sum of rewards."""
    metrics = [
        {"a": 1.0, "b": 2.0},
        {"a": 0.5, "b": 1.0},
    ]
    result = compute_advantages_multi_reward(
        metrics,
        ["a", "b"],
        2,
        None,
        reward_weights=[1.0, 0.5],
    )
    assert result[0] == 1.0 * 1.0 + 2.0 * 0.5  # 2.0
    assert result[1] == 0.5 * 1.0 + 1.0 * 0.5  # 1.0


def test_multi_reward_samples_per_problem_one_no_nan():
    """With samples_per_problem=1, should not produce NaN."""
    metrics = [
        {"r": 1.0},
        {"r": 0.0},
        {"r": 0.5},
    ]
    config = AdvantageConfig()
    result = compute_advantages_multi_reward(
        metrics,
        ["r"],
        1,
        config,
    )
    assert len(result) == 3
    assert all(not (v != v) for v in result)  # NaN check
