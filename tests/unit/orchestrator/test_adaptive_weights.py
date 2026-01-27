import pytest

from prime_rl.orchestrator.adaptive_weights import AdaptiveWeightManager
from prime_rl.orchestrator.config import AdaptiveWeightConfig


@pytest.fixture
def default_config():
    return AdaptiveWeightConfig(enabled=True)


@pytest.fixture
def manager(default_config):
    return AdaptiveWeightManager(
        config=default_config,
        reward_keys=["correct_answer", "length_reward"],
        base_weights=[1.0, 1.0],
    )


def test_initial_weights_match_base_weights(manager):
    weights = manager.get_weights_dict()
    assert weights == {"correct_answer": 1.0, "length_reward": 1.0}


def test_ema_starts_at_zero(manager):
    ema = manager.get_ema_dict()
    assert ema == {"correct_answer": 0.0, "length_reward": 0.0}


def test_update_returns_weights_list(manager):
    weights = manager.update({"correct_answer": 0.5, "length_reward": 0.3})
    assert isinstance(weights, list)
    assert len(weights) == 2


def test_ema_updates_on_batch(manager):
    manager.update({"correct_answer": 1.0, "length_reward": 0.5})
    ema = manager.get_ema_dict()
    # With alpha=0.1: EMA = 0.1 * batch_mean + 0.9 * 0.0
    assert ema["correct_answer"] == pytest.approx(0.1)
    assert ema["length_reward"] == pytest.approx(0.05)


def test_single_reward_decays_without_min_weight_protection():
    """Single reward decays if min_weight < 1.0."""
    config = AdaptiveWeightConfig(enabled=True, saturation_threshold=1.0, decay_exponent=1.0)
    mgr = AdaptiveWeightManager(config=config, reward_keys=["r"], base_weights=[1.0])

    # Simulate high reward for multiple steps
    for _ in range(50):
        mgr.update({"r": 1.0})

    weights = mgr.get_weights_dict()
    # Default min_weight is 0.1, so it should decay
    assert weights["r"] < 0.5


def test_single_reward_no_decay_with_min_weight_one():
    """Single reward doesn't decay if min_weight = 1.0."""
    config = AdaptiveWeightConfig(enabled=True, min_weights=[1.0], saturation_threshold=1.0, decay_exponent=1.0)
    mgr = AdaptiveWeightManager(config=config, reward_keys=["r"], base_weights=[1.0])

    # Simulate high reward for multiple steps
    for _ in range(50):
        mgr.update({"r": 1.0})

    weights = mgr.get_weights_dict()
    # min_weight=1.0 means no decay
    assert weights["r"] == 1.0


def test_per_reward_min_weights():
    """Per-reward min_weights: first reward protected, second decays."""
    config = AdaptiveWeightConfig(enabled=True, min_weights=[1.0, 0.1], saturation_threshold=1.0, decay_exponent=1.0)
    mgr = AdaptiveWeightManager(config=config, reward_keys=["primary", "auxiliary"], base_weights=[1.0, 1.0])

    # Simulate high reward for multiple steps
    for _ in range(50):
        mgr.update({"primary": 1.0, "auxiliary": 1.0})

    weights = mgr.get_weights_dict()
    # Primary has min_weight=1.0, should NOT decay
    assert weights["primary"] == 1.0
    # Auxiliary has min_weight=0.1, SHOULD decay significantly
    assert weights["auxiliary"] < 0.5


def test_min_weight_floor_respected():
    """Rewards should not decay below min_weight * base_weight."""
    config = AdaptiveWeightConfig(enabled=True, min_weights=[0.2, 0.2], saturation_threshold=0.5)
    mgr = AdaptiveWeightManager(config=config, reward_keys=["a", "b"], base_weights=[1.0, 1.0])

    # Push EMA past saturation
    for _ in range(100):
        mgr.update({"a": 1.0, "b": 1.0})

    weights = mgr.get_weights_dict()
    # Both should not go below min_weight * base_weight = 0.2
    assert weights["a"] >= 0.2
    assert weights["b"] >= 0.2


def test_checkpoint_save_and_restore(manager):
    # Update state
    manager.update({"correct_answer": 0.8, "length_reward": 0.4})
    manager.update({"correct_answer": 0.9, "length_reward": 0.5})

    # Save state
    state = manager.get_state()

    # Create new manager and restore
    new_manager = AdaptiveWeightManager(
        config=AdaptiveWeightConfig(enabled=True),
        reward_keys=["correct_answer", "length_reward"],
        base_weights=[1.0, 1.0],
    )
    new_manager.load_state(state)

    assert new_manager.get_ema_dict() == manager.get_ema_dict()
    assert new_manager.get_weights_dict() == manager.get_weights_dict()


def test_missing_reward_key_skipped(manager):
    # Only update one key
    manager.update({"correct_answer": 0.5})
    ema = manager.get_ema_dict()
    assert ema["correct_answer"] == pytest.approx(0.05)
    assert ema["length_reward"] == 0.0  # Unchanged
