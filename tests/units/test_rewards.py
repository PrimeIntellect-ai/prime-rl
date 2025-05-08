import pickle
import pytest

from zeroband.inference.rewards import compute_rewards


@pytest.fixture
def precomputed_rewards(path):
    with open(path, "rb") as f:
        return pickle.load(f)


@pytest.mark.parametrize("path", ["tests/units/rewards.pkl"])
def test_compute_rewards(precomputed_rewards):
    # Get inputs and outputs
    request_outputs = precomputed_rewards["request_outputs"]
    verification_infos = precomputed_rewards["verification_infos"]
    task_types = precomputed_rewards["task_types"]
    config = precomputed_rewards["config"]
    ground_truth_rewards = precomputed_rewards["rewards"]
    ground_truth_task_rewards = precomputed_rewards["task_rewards"]
    ground_truth_length_penalties = precomputed_rewards["length_penalties"]
    ground_truth_advantages = precomputed_rewards["advantages"]

    # Re-compute rewards
    task_types = ["verifiable_math"] * len(request_outputs)
    rewards, task_rewards, length_penalties, advantages = compute_rewards(
        request_outputs,
        verification_infos,
        task_types,
        config=config,
    )
    print("rewards", rewards)
    print("task_rewards", task_rewards)
    print("length_penalties", length_penalties)
    print("advantages", advantages)

    # Assert type
    for reward_or_advantage in [rewards, task_rewards, length_penalties, advantages]:
        assert isinstance(reward_or_advantage, dict)
        assert all(isinstance(key, str) for key in reward_or_advantage.keys())
        assert all(isinstance(value, list) for value in reward_or_advantage.values())

    # Assert computation
    assert rewards == ground_truth_rewards
    assert task_rewards == ground_truth_task_rewards
    assert length_penalties == ground_truth_length_penalties
    assert advantages == ground_truth_advantages
