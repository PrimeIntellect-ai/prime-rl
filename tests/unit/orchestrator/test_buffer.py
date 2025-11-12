import json
import random
from copy import deepcopy

import pytest
from datasets import Dataset

from prime_rl.orchestrator.buffer import Buffer
from prime_rl.orchestrator.config import BufferConfig
from prime_rl.utils.vf import Rollout


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
    ) -> list[Rollout]:
        rollouts = []
        rewards = rewards or [1.0] * len(dataset)
        advantages = advantages or [1.0] * len(dataset)
        for i, (reward, advantage) in enumerate(zip(rewards, advantages)):
            problem_rollouts = [
                Rollout(
                    example_id=i,
                    task="default",
                    prompt_ids=[0],
                    prompt_mask=[1],
                    completion_ids=[1],
                    completion_mask=[1],
                    completion_logprobs=[0.0],
                    is_truncated=False,
                    reward=reward,
                    advantage=advantage,
                    metrics={},
                )
            ] * 2
            rollouts.extend(problem_rollouts)
        return rollouts

    return _make_rollouts


def test_buffer_init(dataset):
    Buffer(dataset, BufferConfig())


def test_buffer_init_with_difficulty_dataset(difficulty_dataset):
    Buffer(difficulty_dataset, BufferConfig())


def test_buffer_sample_problems(dataset):
    buffer = Buffer(dataset, BufferConfig())
    sampled_problems = buffer.sample_problems(2)
    assert sampled_problems[0] == {"example_id": 0, "problem": "0"}
    assert sampled_problems[1] == {"example_id": 4, "problem": "4"}


def test_buffer_sample_problems_with_difficulty_pools(difficulty_dataset, make_rollouts):
    buffer = Buffer(difficulty_dataset, BufferConfig(easy_fraction=0.5, hard_fraction=0.5, easy_threshold=1.0, hard_threshold=0.0))
    # First, set up difficulties by updating with rollouts
    # Set problems 0,1 to easy (advantage=0, reward=1.0), problem 4 to hard (advantage=0, reward=0.0)
    rollouts = make_rollouts(
        difficulty_dataset,
        rewards=[1.0, 1.0, 0.5, 0.5, 0.0],
        advantages=[0.0, 0.0, 1.0, 1.0, 0.0]
    )
    buffer.update(rollouts)
    sampled_problems = buffer.sample_problems(3)
    # Should sample from easy (0,1) and hard (4) pools
    assert len(sampled_problems) == 3
    # Verify we got problems from the right difficulty pools
    sampled_ids = [p["example_id"] for p in sampled_problems]
    assert 0 in sampled_ids or 1 in sampled_ids  # At least one easy
    assert 4 in sampled_ids  # At least one hard


def test_buffer_sample_problems_only_easy(difficulty_dataset, make_rollouts):
    buffer = Buffer(difficulty_dataset, BufferConfig(easy_fraction=1.0, hard_fraction=0.0, easy_threshold=1.0))
    # Set problems 0,1 to easy (advantage=0, reward=1.0)
    rollouts = make_rollouts(
        difficulty_dataset,
        rewards=[1.0, 1.0, 0.5, 0.5, 0.5],
        advantages=[0.0, 0.0, 1.0, 1.0, 1.0]
    )
    buffer.update(rollouts)
    sampled_problems = buffer.sample_problems(2)
    assert sampled_problems[0]["example_id"] == 0
    assert sampled_problems[0]["problem"] == "0"
    assert sampled_problems[1]["example_id"] == 1
    assert sampled_problems[1]["problem"] == "1"


def test_buffer_sample_problems_only_hard(difficulty_dataset, make_rollouts):
    buffer = Buffer(difficulty_dataset, BufferConfig(easy_fraction=0.0, hard_fraction=1.0, hard_threshold=0.0))
    # Set problems 2,4 to hard (advantage=0, reward=0.0)
    rollouts = make_rollouts(
        difficulty_dataset,
        rewards=[0.5, 0.5, 0.0, 0.5, 0.0],
        advantages=[1.0, 1.0, 0.0, 1.0, 0.0]
    )
    buffer.update(rollouts)
    sampled_problems = buffer.sample_problems(2)
    assert sampled_problems[0]["example_id"] == 2
    assert sampled_problems[0]["problem"] == "2"
    assert sampled_problems[1]["example_id"] == 4
    assert sampled_problems[1]["problem"] == "4"


def test_buffer_sample_problems_multiple_epochs(dataset):
    buffer = Buffer(dataset, BufferConfig())
    sampled_problems = buffer.sample_problems(2)
    assert sampled_problems[0] == {"example_id": 0, "problem": "0"}
    assert sampled_problems[1] == {"example_id": 4, "problem": "4"}
    sampled_problems = buffer.sample_problems(2)
    assert sampled_problems[0] == {"example_id": 2, "problem": "2"}
    assert sampled_problems[1] == {"example_id": 1, "problem": "1"}
    sampled_problems = buffer.sample_problems(2)
    assert sampled_problems[0] == {"example_id": 1, "problem": "1"}
    assert sampled_problems[1] == {"example_id": 4, "problem": "4"}


def test_buffer_sample_rollouts(dataset, make_rollouts):
    buffer = Buffer(dataset, BufferConfig())
    rollouts = make_rollouts(dataset)
    buffer.update(rollouts)
    sampled_rollouts = buffer.sample_rollouts(10)
    assert sampled_rollouts == rollouts
    assert len(sampled_rollouts) == 10


def test_buffer_sample_rollouts_partial(dataset, make_rollouts):
    buffer = Buffer(dataset, BufferConfig())
    rollouts = make_rollouts(dataset)
    buffer.update(rollouts)
    sampled_rollouts = buffer.sample_rollouts(5)
    assert len(sampled_rollouts) == 5
    assert len(buffer.rollout_buffer) == 5


def test_buffer_sample_rollouts_more_than_available(dataset, make_rollouts):
    buffer = Buffer(dataset, BufferConfig())
    rollouts = make_rollouts(dataset)
    buffer.update(rollouts)
    sampled_rollouts = buffer.sample_rollouts(20)
    assert len(sampled_rollouts) == 10
    assert len(buffer.rollout_buffer) == 0


def test_buffer_update_with_advantage_zero(difficulty_dataset, make_rollouts):
    buffer = Buffer(difficulty_dataset, BufferConfig())
    # Rollouts with advantage=0 should set difficulty based on thresholds, but not be added to buffer
    rollouts = make_rollouts(difficulty_dataset, rewards=[1.0, 1.0, 1.0, 1.0, 1.0], advantages=[0.0, 0.0, 0.0, 0.0, 0.0])
    buffer.update(rollouts)
    # All should be marked as normal (thresholds are None by default)
    assert all(metadata["difficulty"] == "normal" for metadata in buffer.metadata.values())
    # But none should be in the rollout buffer (advantage == 0)
    assert len(buffer.rollout_buffer) == 0


def test_buffer_update_with_advantage_nonzero(difficulty_dataset, make_rollouts):
    buffer = Buffer(difficulty_dataset, BufferConfig())
    # Rollouts with advantage != 0 should be added to buffer and marked as normal
    rollouts = make_rollouts(difficulty_dataset, rewards=[0.5, 0.5, 0.5, 0.5, 0.5], advantages=[1.0, 1.0, 1.0, 1.0, 1.0])
    buffer.update(rollouts)
    sampled_rollouts = buffer.sample_rollouts(10)
    assert sampled_rollouts == rollouts
    assert len(sampled_rollouts) == 10
    # All should be marked as normal (advantage != 0)
    assert all(metadata["difficulty"] == "normal" for metadata in buffer.metadata.values())

