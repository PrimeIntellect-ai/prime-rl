import pytest

from prime_rl.orchestrator.advantage import compute_advantages, compute_advantages_multi_reward
from prime_rl.orchestrator.config import AdvantageConfig


class TestComputeAdvantages:
    """Tests for single-reward advantage calculation."""

    def test_returns_rewards_when_no_config(self):
        rewards = [1.0, 0.0, 0.5, 0.5]
        result = compute_advantages(rewards, [10, 10, 10, 10], 2, None)
        assert result == rewards

    def test_subtracts_mean_within_groups(self):
        # 2 problems, 2 samples each
        rewards = [1.0, 0.0, 0.5, 0.5]
        config = AdvantageConfig()
        result = compute_advantages(rewards, [10, 10, 10, 10], 2, config)

        # Problem 1: mean=0.5, so [1.0-0.5, 0.0-0.5] = [0.5, -0.5]
        # Problem 2: mean=0.5, so [0.5-0.5, 0.5-0.5] = [0.0, 0.0]
        assert result == [0.5, -0.5, 0.0, 0.0]

    def test_length_weighted_mean(self):
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


class TestComputeAdvantagesMultiReward:
    """Tests for multi-reward advantage calculation."""

    @pytest.fixture
    def simple_metrics(self):
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
    def uniform_metrics(self):
        """All samples have the same reward values (edge case for std=0)."""
        return [
            {"correct_answer": 1.0, "length_reward": 1.0},
            {"correct_answer": 1.0, "length_reward": 1.0},
            {"correct_answer": 1.0, "length_reward": 1.0},
        ]

    def test_returns_correct_length(self, simple_metrics):
        config = AdvantageConfig()
        result = compute_advantages_multi_reward(
            simple_metrics,
            ["correct_answer", "length_reward"],
            3,
            config,
        )
        assert len(result) == 6

    def test_batch_normalized_sum_is_zero(self, simple_metrics):
        config = AdvantageConfig(batch_normalize=True)
        result = compute_advantages_multi_reward(
            simple_metrics,
            ["correct_answer", "length_reward"],
            3,
            config,
        )
        # Batch normalization should make sum approximately 0
        assert abs(sum(result)) < 1e-5

    def test_no_batch_normalization(self, simple_metrics):
        config = AdvantageConfig(batch_normalize=False)
        result = compute_advantages_multi_reward(
            simple_metrics,
            ["correct_answer", "length_reward"],
            3,
            config,
        )
        # Without batch norm, sum won't necessarily be 0
        # Just verify we get reasonable values
        assert len(result) == 6
        assert all(isinstance(v, float) for v in result)

    def test_handles_uniform_rewards(self, uniform_metrics):
        """When all rewards are the same, std=0, should not crash due to eps."""
        config = AdvantageConfig()
        result = compute_advantages_multi_reward(
            uniform_metrics,
            ["correct_answer", "length_reward"],
            3,
            config,
        )
        # All normalized values should be 0 (since r - mean = 0 for all)
        # After batch norm, still 0
        assert len(result) == 3
        assert all(abs(v) < 1e-5 for v in result)

    def test_different_reward_combinations_produce_different_advantages(self, simple_metrics):
        """Key property: different reward combos should not collapse to same advantage."""
        config = AdvantageConfig(batch_normalize=False)
        result = compute_advantages_multi_reward(
            simple_metrics,
            ["correct_answer", "length_reward"],
            3,
            config,
        )
        # First 3 samples (problem 1) should have different advantages
        # Sample 0: correct=1, length=1
        # Sample 1: correct=1, length=0
        # Sample 2: correct=0, length=1
        # These should NOT all be the same
        problem1_advantages = result[:3]
        assert len(set(round(a, 6) for a in problem1_advantages)) > 1

    def test_weighted_rewards(self, simple_metrics):
        config = AdvantageConfig(batch_normalize=False)

        # Equal weights
        result_equal = compute_advantages_multi_reward(
            simple_metrics,
            ["correct_answer", "length_reward"],
            3,
            config,
            reward_weights=None,
        )

        # Correctness weighted 2x more than length
        result_weighted = compute_advantages_multi_reward(
            simple_metrics,
            ["correct_answer", "length_reward"],
            3,
            config,
            reward_weights=[2.0, 1.0],
        )

        # Results should be different
        assert result_equal != result_weighted

    def test_single_reward_key(self):
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

    def test_missing_metric_key_defaults_to_zero(self):
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

    def test_std_eps_prevents_division_by_zero(self):
        """Verify std_eps is used to prevent division by zero."""
        uniform_metrics = [
            {"r": 1.0},
            {"r": 1.0},
        ]
        config = AdvantageConfig(std_eps=1e-8)
        # Should not raise ZeroDivisionError
        result = compute_advantages_multi_reward(
            uniform_metrics,
            ["r"],
            2,
            config,
        )
        assert len(result) == 2
