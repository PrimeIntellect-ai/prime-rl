import torch

from prime_rl.orchestrator.advantage import (
    AdvantageInputs,
    AdvantageOutputs,
    compute_advantages,
    default_advantage_fn,
    setup_advantage_fn,
)
from prime_rl.orchestrator.config import AdvantageConfig, CustomAdvantageConfig


def test_default_advantage_fn_simple_mean():
    inputs = AdvantageInputs(
        rewards=torch.tensor([[1.0, 0.5, 0.8], [0.2, 0.9, 0.1]]),
        completion_lengths=torch.tensor([[10, 12, 8], [15, 11, 9]]),
    )
    result = default_advantage_fn(inputs, length_weighted_mean=False)

    assert result.advantages.shape == (2, 3)
    # Check that mean is subtracted per row
    assert torch.allclose(result.advantages.mean(dim=1), torch.zeros(2), atol=1e-6)


def test_default_advantage_fn_length_weighted():
    inputs = AdvantageInputs(
        rewards=torch.tensor([[1.0, 0.5, 0.8]]),
        completion_lengths=torch.tensor([[10, 20, 10]]),
    )
    result = default_advantage_fn(inputs, length_weighted_mean=True)

    # Length-weighted mean: (1.0*10 + 0.5*20 + 0.8*10) / (10+20+10) = 28/40 = 0.7
    expected_baseline = 0.7
    expected = torch.tensor([[1.0 - expected_baseline, 0.5 - expected_baseline, 0.8 - expected_baseline]])
    assert torch.allclose(result.advantages, expected, atol=1e-6)


def test_compute_advantages_with_config():
    rewards = [1.0, 0.5, 0.8, 0.2, 0.9, 0.1]
    lengths = [10, 12, 8, 15, 11, 9]

    result = compute_advantages(rewards, lengths, samples_per_problem=3, advantage_config=AdvantageConfig())

    assert len(result) == 6
    # First 3 should sum to ~0 (mean subtracted)
    assert abs(sum(result[:3])) < 1e-5
    # Last 3 should sum to ~0
    assert abs(sum(result[3:])) < 1e-5


def test_compute_advantages_without_config():
    rewards = [1.0, 0.5, 0.8]
    lengths = [10, 12, 8]

    result = compute_advantages(rewards, lengths, samples_per_problem=3, advantage_config=None)

    # Without config, returns raw rewards
    assert result == rewards


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
    # Dummy just multiplies rewards by scale
    assert torch.allclose(result.advantages, torch.tensor([[2.0, 1.0, 1.6]]))


def _dummy_custom_advantage(inputs: AdvantageInputs, scale: float = 1.0) -> AdvantageOutputs:
    """A simple custom advantage for testing."""
    return AdvantageOutputs(advantages=inputs.rewards * scale)
