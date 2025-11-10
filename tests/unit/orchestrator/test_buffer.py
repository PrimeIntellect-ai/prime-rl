import json
import random
from copy import deepcopy

import pytest
from datasets import Dataset

from prime_rl.orchestrator.buffer import Buffer
from prime_rl.utils.vf import Rollout
from prime_rl.orchestrator.config import BufferConfig


@pytest.fixture(autouse=True)
def set_seed():
    random.seed(42)


@pytest.fixture
def dataset() -> Dataset:
    return Dataset.from_list(
        [
            {"problem": "0"},
            {"problem": "1"},
            {"problem": "2"},
            {"problem": "3"},
            {"problem": "4"},
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
                )
            ] * 2
            rollouts.extend(problem_rollouts)
        return rollouts

    return _make_rollouts


def test_buffer_init(dataset):
    Buffer(dataset, BufferConfig())


def test_buffer_init_with_difficulty(difficulty_dataset):
    Buffer(difficulty_dataset, BufferConfig(from_scratch=False))


def test_buffer_sample_problems_default(dataset):
    buffer = Buffer(dataset, BufferConfig())
    sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problems) == 2
    # All problems start as "normal" difficulty, so we should get normal problems
    assert all("id" in p for p in sampled_problems)


def test_buffer_sample_problems_mix(difficulty_dataset):
    buffer = Buffer(
        difficulty_dataset,
        BufferConfig(easy_fraction=0.5, hard_fraction=0.5, from_scratch=False),
    )
    sampled_problems = buffer.sample_problems(3)
    assert len(sampled_problems) == 3
    # Should have easy, normal, and hard problems
    problem_ids = [p["id"] for p in sampled_problems]
    assert len(set(problem_ids)) == 3


def test_buffer_sample_problems_only_easy(difficulty_dataset):
    buffer = Buffer(
        difficulty_dataset,
        BufferConfig(easy_fraction=1.0, hard_fraction=0.0, from_scratch=False),
    )
    sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problems) == 2
    problem_ids = [p["id"] for p in sampled_problems]
    # Should only sample from easy problems (ids 0 and 1)
    assert all(pid in [0, 1] for pid in problem_ids)


def test_buffer_sample_problems_only_hard(difficulty_dataset):
    buffer = Buffer(
        difficulty_dataset,
        BufferConfig(easy_fraction=0.0, hard_fraction=1.0, from_scratch=False),
    )
    sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problems) == 2
    problem_ids = [p["id"] for p in sampled_problems]
    # Should only sample from hard problems (id 4)
    # Note: only 1 hard problem exists, so we'll get warnings but should still work
    assert all(pid == 4 for pid in problem_ids) or len(problem_ids) == 1


def test_buffer_sample_problems_multiple_epochs(dataset):
    buffer = Buffer(dataset, BufferConfig())
    sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problems) == 2
    sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problems) == 2
    sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problems) == 2


def test_buffer_sample_rollouts(dataset, make_rollouts):
    buffer = Buffer(dataset, BufferConfig())
    rewards = [0.5, 0.5, 0.5, 0.5, 0.5]
    rollouts = make_rollouts(dataset, rewards=rewards)
    buffer.update(rollouts)
    sampled_rollouts = buffer.sample_rollouts(5)
    # All rewards are 0.5, which is within default range [0.01, 0.99]
    assert len(sampled_rollouts) == 10
    assert all(metadata["reward"] == reward for metadata, reward in zip(buffer.metadata.values(), rewards))


def test_buffer_sample_rollouts_outside_range(dataset, make_rollouts):
    buffer = Buffer(dataset, BufferConfig(min_reward=0.1, max_reward=0.0))
    rewards = [0.0, 0.0, 0.0, 1.0, 1.0]
    rollouts = make_rollouts(dataset, rewards=rewards)
    buffer.update(rollouts)
    sampled_rollouts = buffer.sample_rollouts(5)
    # All rewards are outside range [0.1, 0.0] (invalid range, but tests the filtering)
    assert len(sampled_rollouts) == 0
    assert all(metadata["reward"] == reward for metadata, reward in zip(buffer.metadata.values(), rewards))


def test_buffer_sample_rollouts_within_range(dataset, make_rollouts):
    buffer = Buffer(dataset, BufferConfig(min_reward=0.3, max_reward=0.7))
    rewards = [0.2, 0.4, 0.5, 0.6, 0.8]
    rollouts = make_rollouts(dataset, rewards=rewards)
    buffer.update(rollouts)
    sampled_rollouts = buffer.sample_rollouts(5)
    # Only rewards 0.4, 0.5, 0.6 are within range [0.3, 0.7]
    # So we should get 3 problems worth of rollouts = 6 rollouts
    assert len(sampled_rollouts) == 6
    assert all(metadata["reward"] == reward for metadata, reward in zip(buffer.metadata.values(), rewards))


def test_buffer_update_difficulty(difficulty_dataset, make_rollouts):
    buffer = Buffer(difficulty_dataset, BufferConfig(from_scratch=False))
    # Update with high rewards - should move to easy
    rollouts = make_rollouts(difficulty_dataset, rewards=[0.9, 0.9, 0.9, 0.9, 0.9])
    buffer.update(rollouts)
    # All should be easy now (reward 0.9 > easy_border 0.8)
    assert all(metadata["difficulty"] == "easy" for metadata in buffer.metadata.values())


def test_buffer_update_difficulty_hard(difficulty_dataset, make_rollouts):
    buffer = Buffer(difficulty_dataset, BufferConfig(from_scratch=False))
    # Update with low rewards - should move to hard
    rollouts = make_rollouts(difficulty_dataset, rewards=[0.1, 0.1, 0.1, 0.1, 0.1])
    buffer.update(rollouts)
    # All should be hard now (reward 0.1 < hard_border 0.2)
    assert all(metadata["difficulty"] == "hard" for metadata in buffer.metadata.values())


def test_buffer_update_difficulty_normal(difficulty_dataset, make_rollouts):
    buffer = Buffer(difficulty_dataset, BufferConfig(from_scratch=False))
    # Update with medium rewards - should stay/move to normal
    rollouts = make_rollouts(difficulty_dataset, rewards=[0.5, 0.5, 0.5, 0.5, 0.5])
    buffer.update(rollouts)
    # All should be normal now (0.2 <= reward 0.5 <= 0.8)
    assert all(metadata["difficulty"] == "normal" for metadata in buffer.metadata.values())
