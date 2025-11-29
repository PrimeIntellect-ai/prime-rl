import json
import random
from copy import deepcopy

import pytest
import verifiers as vf
from datasets import Dataset

from prime_rl.orchestrator.buffer import Buffer
from prime_rl.orchestrator.config import BufferConfig


@pytest.fixture(autouse=True)
def set_seed():
    random.seed(42)


@pytest.fixture
def dataset() -> Dataset:
    return Dataset.from_list(
        [
            {"example_id": 0, "problem": "0"},
            {"example_id": 1, "problem": "1"},
            {"example_id": 2, "problem": "2"},
            {"example_id": 3, "problem": "3"},
            {"example_id": 4, "problem": "4"},
        ]
    )


@pytest.fixture
def difficulty_dataset(dataset: Dataset) -> Dataset:
    difficulty_dataset = deepcopy(dataset)
    difficulties = ["easy", "easy", "normal", "normal", "hard"]
    difficulty_dataset = difficulty_dataset.map(
        lambda x, i: {"metadata": json.dumps({"difficulty": difficulties[i]}), "rollouts": json.dumps([])},
        with_indices=True,
    )
    return difficulty_dataset


@pytest.fixture
def make_rollouts():
    """Factory fixture that creates rollouts for any given dataset."""

    def _make_rollouts(
        dataset: Dataset, rewards: list[float] | None = None, advantages: list[float] | None = None
    ) -> list[vf.State]:
        rollouts = []
        rewards = rewards or [1.0] * len(dataset)
        advantages = advantages or [1.0] * len(dataset)
        for i, (reward, advantage) in enumerate(zip(rewards, advantages)):
            problem_rollouts = [
                vf.State(
                    example_id=i,
                    task="default",
                    reward=reward,
                    advantage=advantage,
                )
            ] * 2
            rollouts.extend(problem_rollouts)
        return rollouts

    return _make_rollouts


def test_buffer_init(dataset):
    buffer_config = BufferConfig()
    Buffer(dataset, buffer_config)


def test_buffer_sample_problems(dataset):
    buffer_config = BufferConfig()
    buffer = Buffer(dataset, buffer_config)
    sampled_problems = buffer.sample_problems(2)
    assert sampled_problems[0] == {"example_id": 0, "problem": "0"}
    assert sampled_problems[1] == {"example_id": 4, "problem": "4"}


def test_buffer_sample_problems_with_difficulty_pools(difficulty_dataset, make_rollouts):
    buffer_config = BufferConfig(easy_fraction=0.5, hard_fraction=0.5, easy_threshold=1.0, hard_threshold=0.0)
    buffer = Buffer(difficulty_dataset, buffer_config)
    # First, set up difficulties by updating with rollouts
    # Set problems 0,1 to easy (advantage=0, reward=1.0), problem 4 to hard (advantage=0, reward=0.0)
    rollouts = make_rollouts(
        difficulty_dataset, rewards=[1.0, 1.0, 0.5, 0.5, 0.0], advantages=[0.0, 0.0, 1.0, 1.0, 0.0]
    )
    buffer.update(rollouts)
    sampled_problems = buffer.sample_problems(3)
    # Should sample from easy (0,1) and hard (4) pools
    assert len(sampled_problems) == 3
    # Verify we got problems from the right difficulty pools
    sampled_ids = [p["example_id"] for p in sampled_problems]
    assert 0 in sampled_ids or 1 in sampled_ids  # At least one easy
    assert 4 in sampled_ids  # At least one hard


def test_buffer_sample_rollouts(dataset, make_rollouts):
    buffer_config = BufferConfig(online_difficulty_filtering=False)
    buffer = Buffer(dataset, buffer_config)
    # Use rewards that won't be filtered (0.5 instead of 1.0)
    rollouts = make_rollouts(dataset, rewards=[0.5] * len(dataset))
    buffer.update(rollouts)
    sampled_rollouts = buffer.sample_rollouts(10)
    assert sampled_rollouts == rollouts
    assert len(sampled_rollouts) == 10


def test_buffer_sample_rollouts_more_than_available(dataset, make_rollouts):
    buffer_config = BufferConfig(online_difficulty_filtering=False)
    buffer = Buffer(dataset, buffer_config)
    # Use rewards that won't be filtered (0.5 instead of 1.0)
    rollouts = make_rollouts(dataset, rewards=[0.5] * len(dataset))
    buffer.update(rollouts)
    sampled_rollouts = buffer.sample_rollouts(20)
    assert len(sampled_rollouts) == 10
    assert len(buffer.rollout_buffer) == 0


def test_buffer_update_with_advantage_nonzero(difficulty_dataset, make_rollouts):
    buffer_config = BufferConfig()
    buffer = Buffer(difficulty_dataset, buffer_config)
    # Rollouts with advantage != 0 should be added to buffer and marked as normal
    rollouts = make_rollouts(
        difficulty_dataset, rewards=[0.5, 0.5, 0.5, 0.5, 0.5], advantages=[1.0, 1.0, 1.0, 1.0, 1.0]
    )
    buffer.update(rollouts)
    sampled_rollouts = buffer.sample_rollouts(10)
    assert sampled_rollouts == rollouts
    assert len(sampled_rollouts) == 10
    # All should be marked as normal (advantage != 0)
    assert all(metadata["difficulty"] == "normal" for metadata in buffer.metadata.values())


def test_buffer_online_difficulty_filtering(dataset, make_rollouts):
    """Test that only rollouts with avg_reward == 0.0 or 1.0 are filtered."""
    buffer_config = BufferConfig(online_difficulty_filtering=True, easy_threshold=1.0, hard_threshold=0.0)
    buffer = Buffer(dataset, buffer_config)
    # Mix of rewards: 1.0 (filtered), 0.5 (kept), 0.0 (filtered), 0.5 (kept), 0.5 (kept)
    rollouts = make_rollouts(dataset, rewards=[1.0, 0.5, 0.0, 0.5, 0.5])
    buffer.update(rollouts)
    # Only rollouts with reward 0.5 should be in buffer (3 problems * 2 rollouts = 6)
    assert len(buffer.rollout_buffer) == 6
    # Check that difficulties are set correctly
    assert buffer.metadata[0]["difficulty"] == "easy"
    assert buffer.metadata[1]["difficulty"] == "normal"
    assert buffer.metadata[2]["difficulty"] == "hard"
    assert buffer.metadata[3]["difficulty"] == "normal"
    assert buffer.metadata[4]["difficulty"] == "normal"
