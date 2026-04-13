import torch

from prime_rl.configs.orchestrator import CustomAdvantageConfig, DefaultAdvantageConfig
from prime_rl.orchestrator.advantage import (
    AdvantageInputs,
    AdvantageOutputs,
    compute_advantages,
    default_advantage_fn,
    setup_advantage_fn,
)


def test_default_advantage_fn_simple_mean():
    inputs = AdvantageInputs(
        rewards=torch.tensor([[1.0, 0.5, 0.8], [0.2, 0.9, 0.1]]),
        completion_lengths=torch.tensor([[10, 12, 8], [15, 11, 9]]),
    )
    result = default_advantage_fn(inputs)

    assert result.advantages.shape == (2, 3)
    # Check that mean is subtracted per row
    assert torch.allclose(result.advantages.mean(dim=1), torch.zeros(2), atol=1e-6)


def test_efficiency_mixed_group():
    """Mixed group: bounded amplification for short correct rollouts, others unchanged."""
    inputs = AdvantageInputs(
        rewards=torch.tensor([[1.0, 1.0, 0.0, 1.0]]),
        completion_lengths=torch.tensor([[10, 30, 20, 20]]),
    )
    result = default_advantage_fn(inputs, length_shaping=True)

    # mean_correct_len = (10+30+20)/3 = 20
    # bonus = clamp(1 - [10,30,20,20]/20, 0, 1) = [0.5, 0, 0, 0]
    # baseline = 0.75
    # A = (R - 0.75) * (1 + bonus * correct_mask)
    #   = [0.25, 0.25, -0.75, 0.25] * [1.5, 1, 1, 1]
    expected = torch.tensor([[0.375, 0.25, -0.75, 0.25]])
    assert torch.allclose(result.advantages, expected, atol=1e-6)

    # All correct rollouts have positive advantage
    correct_mask = inputs.rewards[0] >= 1.0
    assert (result.advantages[0][correct_mask] > 0).all()


def test_efficiency_all_correct_group():
    """All-correct group: bounded [0, 1] advantages, zero for above-average length."""
    inputs = AdvantageInputs(
        rewards=torch.tensor([[1.0, 1.0, 1.0]]),
        completion_lengths=torch.tensor([[10, 20, 40]]),
    )
    result = default_advantage_fn(inputs, length_shaping=True)

    # mean_len = 70/3 ≈ 23.33
    # bonus = clamp(1 - [10, 20, 40] / (70/3), 0, 1) = [4/7, 1/7, 0]
    expected = torch.tensor([[4.0 / 7, 1.0 / 7, 0.0]])
    assert torch.allclose(result.advantages, expected, atol=1e-6)

    # Shortest has highest advantage, longest gets zero
    assert result.advantages[0, 0] > result.advantages[0, 1] > result.advantages[0, 2]
    assert result.advantages[0, 2] == 0.0

    # All advantages bounded in [0, 1]
    assert (result.advantages >= 0).all()
    assert (result.advantages <= 1).all()


def test_efficiency_all_zero_rewards():
    """When all rewards are 0, no length shaping — falls back to standard GRPO."""
    inputs = AdvantageInputs(
        rewards=torch.tensor([[0.0, 0.0, 0.0]]),
        completion_lengths=torch.tensor([[10, 20, 15]]),
    )
    result_with = default_advantage_fn(inputs, length_shaping=True)
    result_without = default_advantage_fn(inputs)

    assert torch.allclose(result_with.advantages, result_without.advantages, atol=1e-6)


def test_efficiency_single_correct():
    """Single correct rollout: w=1, no length differentiation, same as standard GRPO."""
    inputs = AdvantageInputs(
        rewards=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        completion_lengths=torch.tensor([[100, 50, 200, 150]]),
    )
    result = default_advantage_fn(inputs, length_shaping=True)

    expected = torch.tensor([[0.75, -0.25, -0.25, -0.25]])
    assert torch.allclose(result.advantages, expected, atol=1e-6)


def test_efficiency_shorter_correct_higher_advantage():
    """Among correct rollouts in a mixed group, shorter always gets higher advantage."""
    inputs = AdvantageInputs(
        rewards=torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]]),
        completion_lengths=torch.tensor([[50, 100, 200, 80, 120]]),
    )
    result = default_advantage_fn(inputs, length_shaping=True)

    advs = result.advantages[0]
    assert advs[0] > advs[1] > advs[2]
    assert (advs[:3] > 0).all()
    assert (advs[3:] < 0).all()


def test_efficiency_incorrect_unchanged():
    """Incorrect rollouts get exactly the same advantage as standard GRPO."""
    inputs = AdvantageInputs(
        rewards=torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
        completion_lengths=torch.tensor([[10, 30, 20, 40]]),
    )
    result_eff = default_advantage_fn(inputs, length_shaping=True)
    result_std = default_advantage_fn(inputs)

    assert torch.allclose(result_eff.advantages[0, 2:], result_std.advantages[0, 2:], atol=1e-6)


def test_efficiency_amplification_bounded():
    """Even with extreme length outliers, amplification is capped at 2x."""
    inputs = AdvantageInputs(
        rewards=torch.tensor([[1.0, 1.0, 0.0]]),
        completion_lengths=torch.tensor([[1, 10000, 5000]]),
    )
    result = default_advantage_fn(inputs, length_shaping=True)

    # Standard advantage for correct = 1 - 2/3 = 1/3, max amplified = 1/3 * 2 = 2/3
    assert result.advantages[0, 0] < 2.0 / 3 + 1e-6


def test_efficiency_multiple_problems():
    """Handles multiple problems independently."""
    inputs = AdvantageInputs(
        rewards=torch.tensor(
            [
                [1.0, 1.0, 0.0],  # mixed
                [1.0, 1.0, 1.0],  # all correct
            ]
        ),
        completion_lengths=torch.tensor(
            [
                [10, 20, 15],
                [10, 20, 40],
            ]
        ),
    )
    result = default_advantage_fn(inputs, length_shaping=True)

    # Row 0: mixed group
    assert result.advantages[0, 0] > result.advantages[0, 1]
    assert (result.advantages[0, :2] > 0).all()
    assert result.advantages[0, 2] < 0

    # Row 1: all-correct group — bounded [0, 1], longest gets 0
    assert result.advantages[1, 0] > result.advantages[1, 1] > result.advantages[1, 2]
    assert result.advantages[1, 2] == 0.0
    assert (result.advantages[1] >= 0).all()
    assert (result.advantages[1] <= 1).all()


def _make_rollout(reward: float, completion_len: int) -> dict:
    """Create a minimal rollout dict for advantage testing."""
    return {
        "reward": reward,
        "trajectory": [{"tokens": {"prompt_ids": [0], "completion_ids": list(range(completion_len))}}],
    }


def test_compute_advantages_with_config():
    rewards = [1.0, 0.5, 0.8, 0.2, 0.9, 0.1]
    lengths = [10, 12, 8, 15, 11, 9]
    rollouts = [_make_rollout(r, l) for r, l in zip(rewards, lengths)]

    compute_advantages(rollouts, samples_per_problem=3, advantage_config=DefaultAdvantageConfig())

    advantages = [r["advantage"] for r in rollouts]
    assert len(advantages) == 6
    assert abs(sum(advantages[:3])) < 1e-5
    assert abs(sum(advantages[3:])) < 1e-5


def test_compute_advantages_without_config():
    rewards = [1.0, 0.5, 0.8]
    lengths = [10, 12, 8]
    rollouts = [_make_rollout(r, l) for r, l in zip(rewards, lengths)]

    compute_advantages(rollouts, samples_per_problem=3, advantage_config=None)

    advantages = [r["advantage"] for r in rollouts]
    assert advantages == rewards


def test_setup_advantage_fn_with_custom_config():
    config = CustomAdvantageConfig(
        import_path="tests.unit.orchestrator.test_advantage._dummy_custom_advantage",
        kwargs={"scale": 2.0},
    )
    advantage_fn = setup_advantage_fn(config)

    inputs = AdvantageInputs(
        rewards=torch.tensor([[1.0, 0.5, 0.8]]),
        completion_lengths=torch.tensor([[10, 12, 8]]),
    )

    result = advantage_fn(inputs)
    assert isinstance(result, AdvantageOutputs)
    assert torch.allclose(result.advantages, torch.tensor([[2.0, 1.0, 1.6]]))


def _dummy_custom_advantage(inputs: AdvantageInputs, scale: float = 1.0) -> AdvantageOutputs:
    """A simple custom advantage for testing."""
    return AdvantageOutputs(advantages=inputs.rewards * scale)
