import random
from copy import deepcopy

import pytest
from datasets import Dataset

from prime_rl.orchestrator.buffer import DifficultyPoolBuffer, OnlineDifficultyBuffer, Rollout, SimpleBuffer
from prime_rl.orchestrator.config import DifficultyPoolBufferConfig, OnlineDifficultyBufferConfig, SimpleBufferConfig


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
    difficulty_dataset = difficulty_dataset.map(lambda x, i: {"difficulty": difficulties[i]}, with_indices=True)
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
                    problem_id=i,
                    prompt_tokens=[0],
                    prompt_mask=[1],
                    completion_tokens=[1],
                    completion_mask=[1],
                    completion_logprobs=[0.0],
                    reward=reward,
                    advantage=advantage,
                )
            ] * 2
            rollouts.extend(problem_rollouts)
        return rollouts

    return _make_rollouts


def test_simple_buffer_init(dataset):
    SimpleBuffer(dataset, SimpleBufferConfig())


def test_difficulty_pool_buffer_init(difficulty_dataset):
    DifficultyPoolBuffer(difficulty_dataset, DifficultyPoolBufferConfig())


def test_online_difficulty_buffer_init(difficulty_dataset):
    OnlineDifficultyBuffer(difficulty_dataset, OnlineDifficultyBufferConfig())


def test_simple_buffer_sample_problems(dataset):
    buffer = SimpleBuffer(dataset, SimpleBufferConfig())
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [0, 4]
    assert sampled_problems[0] == {"problem": "0"}
    assert sampled_problems[1] == {"problem": "4"}


def test_difficulty_pool_buffer_sample_default_problems(dataset):
    buffer = DifficultyPoolBuffer(dataset, DifficultyPoolBufferConfig())
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [0, 4]
    assert sampled_problems[0] == {"problem": "0"}
    assert sampled_problems[1] == {"problem": "4"}


def test_difficulty_pool_buffer_sample_problems_mix(difficulty_dataset):
    buffer = DifficultyPoolBuffer(
        difficulty_dataset,
        DifficultyPoolBufferConfig(difficulty_field="difficulty", easy_fraction=0.5, hard_fraction=0.5),
    )
    sampled_problem_ids, sampled_problems = buffer.sample_problems(3)
    assert len(sampled_problem_ids) == len(sampled_problems) == 3
    assert sampled_problem_ids == [0, 3, 4]
    assert sampled_problems[0] == {"problem": "0", "difficulty": "easy"}
    assert sampled_problems[1] == {"problem": "3", "difficulty": "normal"}
    assert sampled_problems[2] == {"problem": "4", "difficulty": "hard"}


def test_difficulty_pool_buffer_sample_problems_only_easy(difficulty_dataset):
    buffer = DifficultyPoolBuffer(
        difficulty_dataset,
        DifficultyPoolBufferConfig(difficulty_field="difficulty", easy_fraction=1.0, hard_fraction=0.0),
    )
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [0, 1]
    assert sampled_problems[0] == {"problem": "0", "difficulty": "easy"}
    assert sampled_problems[1] == {"problem": "1", "difficulty": "easy"}


def test_difficulty_pool_buffer_sample_problems_only_hard(difficulty_dataset):
    buffer = DifficultyPoolBuffer(
        difficulty_dataset,
        DifficultyPoolBufferConfig(difficulty_field="difficulty", easy_fraction=0.0, hard_fraction=1.0),
    )
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [2, 4]
    assert sampled_problems[0] == {"problem": "2", "difficulty": "normal"}
    assert sampled_problems[1] == {"problem": "4", "difficulty": "hard"}


def test_online_difficulty_buffer_sample_problems(dataset):
    buffer = OnlineDifficultyBuffer(dataset, OnlineDifficultyBufferConfig())
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [0, 4]
    assert sampled_problems[0] == {"problem": "0"}
    assert sampled_problems[1] == {"problem": "4"}


def test_simple_buffer_sample_problems_multiple_epochs(dataset):
    buffer = SimpleBuffer(dataset, SimpleBufferConfig())
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [0, 4]
    assert sampled_problems[0] == {"problem": "0"}
    assert sampled_problems[1] == {"problem": "4"}
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [2, 1]
    assert sampled_problems[0] == {"problem": "2"}
    assert sampled_problems[1] == {"problem": "1"}
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [1, 4]
    assert sampled_problems[0] == {"problem": "1"}
    assert sampled_problems[1] == {"problem": "4"}


def test_difficulty_pool_buffer_sample_default_problems_multiple_epochs(dataset):
    buffer = DifficultyPoolBuffer(dataset, DifficultyPoolBufferConfig())
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [0, 4]
    assert sampled_problems[0] == {"problem": "0"}
    assert sampled_problems[1] == {"problem": "4"}
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [2, 1]
    assert sampled_problems[0] == {"problem": "2"}
    assert sampled_problems[1] == {"problem": "1"}
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [1, 4]
    assert sampled_problems[0] == {"problem": "1"}
    assert sampled_problems[1] == {"problem": "4"}


def test_difficulty_pool_buffer_sample_problems_multiple_epochs_mix(difficulty_dataset):
    buffer = DifficultyPoolBuffer(
        difficulty_dataset,
        DifficultyPoolBufferConfig(difficulty_field="difficulty", easy_fraction=0.5, hard_fraction=0.5),
    )
    sampled_problem_ids, sampled_problems = buffer.sample_problems(3)
    assert len(sampled_problem_ids) == len(sampled_problems) == 3
    assert sampled_problem_ids == [0, 3, 4]
    assert sampled_problems[0] == {"problem": "0", "difficulty": "easy"}
    assert sampled_problems[1] == {"problem": "3", "difficulty": "normal"}
    assert sampled_problems[2] == {"problem": "4", "difficulty": "hard"}
    sampled_problem_ids, sampled_problems = buffer.sample_problems(3)
    assert len(sampled_problem_ids) == len(sampled_problems) == 3
    assert sampled_problem_ids == [0, 2, 4]
    assert sampled_problems[0] == {"problem": "0", "difficulty": "easy"}
    assert sampled_problems[1] == {"problem": "2", "difficulty": "normal"}
    assert sampled_problems[2] == {"problem": "4", "difficulty": "hard"}


def test_difficulty_pool_buffer_sample_problems_multiple_epochs_only_easy(difficulty_dataset):
    buffer = DifficultyPoolBuffer(
        difficulty_dataset,
        DifficultyPoolBufferConfig(difficulty_field="difficulty", easy_fraction=1.0, hard_fraction=0.0),
    )
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [0, 1]
    assert sampled_problems[0] == {"problem": "0", "difficulty": "easy"}
    assert sampled_problems[1] == {"problem": "1", "difficulty": "easy"}
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [1, 0]
    assert sampled_problems[0] == {"problem": "1", "difficulty": "easy"}
    assert sampled_problems[1] == {"problem": "0", "difficulty": "easy"}


def test_difficulty_pool_buffer_sample_problems_multiple_epochs_only_hard(difficulty_dataset):
    buffer = DifficultyPoolBuffer(
        difficulty_dataset,
        DifficultyPoolBufferConfig(difficulty_field="difficulty", easy_fraction=0.0, hard_fraction=1.0),
    )
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [2, 4]
    assert sampled_problems[0] == {"problem": "2", "difficulty": "normal"}
    assert sampled_problems[1] == {"problem": "4", "difficulty": "hard"}
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [2, 4]
    assert sampled_problems[0] == {"problem": "2", "difficulty": "normal"}
    assert sampled_problems[1] == {"problem": "4", "difficulty": "hard"}


def test_online_difficulty_buffer_sample_problems_multiple_epochs(dataset):
    buffer = OnlineDifficultyBuffer(dataset, OnlineDifficultyBufferConfig())
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [0, 4]
    assert sampled_problems[0] == {"problem": "0"}
    assert sampled_problems[1] == {"problem": "4"}
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [2, 1]
    assert sampled_problems[0] == {"problem": "2"}
    assert sampled_problems[1] == {"problem": "1"}
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [1, 4]
    assert sampled_problems[0] == {"problem": "1"}
    assert sampled_problems[1] == {"problem": "4"}


def test_simple_buffer_sample_rollouts(dataset, make_rollouts):
    buffer = SimpleBuffer(dataset, SimpleBufferConfig())
    rollouts = make_rollouts(dataset)
    buffer.update(rollouts)
    sampled_rollouts = buffer.sample_rollouts(5)
    assert sampled_rollouts == rollouts
    assert len(sampled_rollouts) == 10


@pytest.mark.parametrize("n", [1, 4, 6, 10])
def test_simple_buffer_sample_invalid_rollouts(dataset, make_rollouts, n):
    buffer = SimpleBuffer(dataset, SimpleBufferConfig())
    rollouts = make_rollouts(dataset)
    buffer.update(rollouts)
    with pytest.raises(AssertionError):
        buffer.sample_rollouts(n)


def test_difficulty_pool_buffer_sample_rollouts(difficulty_dataset, make_rollouts):
    buffer = DifficultyPoolBuffer(
        difficulty_dataset,
        DifficultyPoolBufferConfig(),
    )
    rollouts = make_rollouts(difficulty_dataset, rewards=[0.5, 0.5, 0.5, 0.5, 0.5])
    buffer.update(rollouts)
    sampled_rollouts = buffer.sample_rollouts(5)
    assert sampled_rollouts == rollouts
    assert len(sampled_rollouts) == 10
    assert all(metadata["difficulty"] == "normal" for metadata in buffer.metadata.values())


def test_difficulty_pool_buffer_sample_rollouts_easy(difficulty_dataset, make_rollouts):
    buffer = DifficultyPoolBuffer(difficulty_dataset, DifficultyPoolBufferConfig())
    rollouts = make_rollouts(difficulty_dataset, rewards=[1.0, 1.0, 1.0, 1.0, 1.0])
    buffer.update(rollouts)
    sampled_rollouts = buffer.sample_rollouts(5)
    assert sampled_rollouts == rollouts
    assert len(sampled_rollouts) == 10
    assert all(metadata["difficulty"] == "easy" for metadata in buffer.metadata.values())


def test_difficulty_pool_buffer_sample_rollouts_hard(difficulty_dataset, make_rollouts):
    buffer = DifficultyPoolBuffer(
        difficulty_dataset,
        DifficultyPoolBufferConfig(),
    )
    rollouts = make_rollouts(difficulty_dataset, rewards=[0.0, 0.0, 0.0, 0.0, 0.0])
    buffer.update(rollouts)
    sampled_rollouts = buffer.sample_rollouts(5)
    assert sampled_rollouts == rollouts
    assert len(sampled_rollouts) == 10
    assert all(metadata["difficulty"] == "hard" for metadata in buffer.metadata.values())


def test_online_difficulty_buffer_sample_rollouts(dataset, make_rollouts):
    buffer = OnlineDifficultyBuffer(dataset, OnlineDifficultyBufferConfig())
    rewards = [0.5, 0.5, 0.5, 0.5, 0.5]
    rollouts = make_rollouts(dataset, rewards=rewards)
    buffer.update(rollouts)
    sampled_rollouts = buffer.sample_rollouts(5)
    assert sampled_rollouts == rollouts
    assert len(sampled_rollouts) == 10
    assert all(metadata["reward"] == reward for metadata, reward in zip(buffer.metadata.values(), rewards))


def test_online_difficulty_buffer_sample_rollouts_outside_range(dataset, make_rollouts):
    buffer = OnlineDifficultyBuffer(dataset, OnlineDifficultyBufferConfig(min_reward=0.1, max_reward=0.0))
    rewards = [0.0, 0.0, 0.0, 1.0, 1.0]
    rollouts = make_rollouts(dataset, rewards=rewards)
    buffer.update(rollouts)
    sampled_rollouts = buffer.sample_rollouts(5)
    assert sampled_rollouts == []
    assert len(sampled_rollouts) == 0
    assert all(metadata["reward"] == reward for metadata, reward in zip(buffer.metadata.values(), rewards))
