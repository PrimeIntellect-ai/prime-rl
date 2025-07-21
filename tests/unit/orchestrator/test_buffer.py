from copy import deepcopy

import pytest
from datasets import Dataset

from prime_rl.orchestrator.buffer import OnlineDifficultyBuffer, PriorityPoolBuffer, Rollout, SimpleBuffer
from prime_rl.orchestrator.config import OnlineDifficultyBufferConfig, PriorityPoolBufferConfig, SimpleBufferConfig


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
def priority_dataset(dataset: Dataset) -> Dataset:
    priority_dataset = deepcopy(dataset)
    priorities = ["low", "low", "high", "high", "high"]
    priority_dataset = priority_dataset.map(lambda x, i: {"priority": priorities[i]}, with_indices=True)
    return priority_dataset


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


def test_priority_pool_buffer_init(priority_dataset):
    PriorityPoolBuffer(priority_dataset, PriorityPoolBufferConfig())


def test_online_difficulty_buffer_init(priority_dataset):
    OnlineDifficultyBuffer(priority_dataset, OnlineDifficultyBufferConfig())


def test_simple_buffer_sample_problems(dataset):
    buffer = SimpleBuffer(dataset, SimpleBufferConfig())
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [0, 1]
    assert sampled_problems[0] == {"problem": "0"}
    assert sampled_problems[1] == {"problem": "1"}


def test_priority_buffer_sample_default_problems(dataset):
    buffer = PriorityPoolBuffer(dataset, PriorityPoolBufferConfig())
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [0, 1]
    assert sampled_problems[0] == {"problem": "0"}
    assert sampled_problems[1] == {"problem": "1"}


def test_priority_buffer_sample_problems_mix(priority_dataset):
    buffer = PriorityPoolBuffer(
        priority_dataset, PriorityPoolBufferConfig(priority_field="priority", low_priority_fraction=0.5)
    )
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [0, 2]
    assert sampled_problems[0] == {"problem": "0", "priority": "low"}
    assert sampled_problems[1] == {"problem": "2", "priority": "high"}


def test_priority_buffer_sample_problems_only_low(priority_dataset):
    buffer = PriorityPoolBuffer(
        priority_dataset, PriorityPoolBufferConfig(priority_field="priority", low_priority_fraction=1.0)
    )
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [0, 1]
    assert sampled_problems[0] == {"problem": "0", "priority": "low"}
    assert sampled_problems[1] == {"problem": "1", "priority": "low"}


def test_priority_buffer_sample_problems_only_high(priority_dataset):
    buffer = PriorityPoolBuffer(
        priority_dataset, PriorityPoolBufferConfig(priority_field="priority", low_priority_fraction=0.0)
    )
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [2, 3]
    assert sampled_problems[0] == {"problem": "2", "priority": "high"}
    assert sampled_problems[1] == {"problem": "3", "priority": "high"}


def test_difficulty_buffer_sample_problems(dataset):
    buffer = OnlineDifficultyBuffer(dataset, OnlineDifficultyBufferConfig())
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [0, 1]
    assert sampled_problems[0] == {"problem": "0"}
    assert sampled_problems[1] == {"problem": "1"}


def test_simple_buffer_sample_problems_multiple_epochs(dataset):
    buffer = SimpleBuffer(dataset, SimpleBufferConfig())
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [0, 1]
    assert sampled_problems[0] == {"problem": "0"}
    assert sampled_problems[1] == {"problem": "1"}
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [2, 3]
    assert sampled_problems[0] == {"problem": "2"}
    assert sampled_problems[1] == {"problem": "3"}
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [0, 1]
    assert sampled_problems[0] == {"problem": "0"}
    assert sampled_problems[1] == {"problem": "1"}


def test_priority_buffer_sample_default_problems_multiple_epochs(dataset):
    buffer = PriorityPoolBuffer(dataset, PriorityPoolBufferConfig())
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [0, 1]
    assert sampled_problems[0] == {"problem": "0"}
    assert sampled_problems[1] == {"problem": "1"}
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [2, 3]
    assert sampled_problems[0] == {"problem": "2"}
    assert sampled_problems[1] == {"problem": "3"}
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [0, 1]
    assert sampled_problems[0] == {"problem": "0"}
    assert sampled_problems[1] == {"problem": "1"}


def test_priority_buffer_sample_problems_multiple_epochs_mix(priority_dataset):
    buffer = PriorityPoolBuffer(
        priority_dataset, PriorityPoolBufferConfig(priority_field="priority", low_priority_fraction=0.5)
    )
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [0, 2]
    assert sampled_problems[0] == {"problem": "0", "priority": "low"}
    assert sampled_problems[1] == {"problem": "2", "priority": "high"}
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [1, 3]
    assert sampled_problems[0] == {"problem": "1", "priority": "low"}
    assert sampled_problems[1] == {"problem": "3", "priority": "high"}
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [0, 2]
    assert sampled_problems[0] == {"problem": "0", "priority": "low"}
    assert sampled_problems[1] == {"problem": "2", "priority": "high"}


def test_priority_buffer_sample_problems_multiple_epochs_only_low(priority_dataset):
    buffer = PriorityPoolBuffer(
        priority_dataset, PriorityPoolBufferConfig(priority_field="priority", low_priority_fraction=1.0)
    )
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [0, 1]
    assert sampled_problems[0] == {"problem": "0", "priority": "low"}
    assert sampled_problems[1] == {"problem": "1", "priority": "low"}
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [2, 3]
    assert sampled_problems[0] == {"problem": "2", "priority": "high"}
    assert sampled_problems[1] == {"problem": "3", "priority": "high"}
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [0, 1]
    assert sampled_problems[0] == {"problem": "0", "priority": "low"}
    assert sampled_problems[1] == {"problem": "1", "priority": "low"}


def test_priority_buffer_sample_problems_multiple_epochs_only_high(priority_dataset):
    buffer = PriorityPoolBuffer(
        priority_dataset, PriorityPoolBufferConfig(priority_field="priority", low_priority_fraction=0.0)
    )
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [2, 3]
    assert sampled_problems[0] == {"problem": "2", "priority": "high"}
    assert sampled_problems[1] == {"problem": "3", "priority": "high"}
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [2, 3]
    assert sampled_problems[0] == {"problem": "2", "priority": "high"}
    assert sampled_problems[1] == {"problem": "3", "priority": "high"}


def test_difficulty_buffer_sample_problems_multiple_epochs(dataset):
    buffer = OnlineDifficultyBuffer(dataset, OnlineDifficultyBufferConfig())
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [0, 1]
    assert sampled_problems[0] == {"problem": "0"}
    assert sampled_problems[1] == {"problem": "1"}
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [2, 3]
    assert sampled_problems[0] == {"problem": "2"}
    assert sampled_problems[1] == {"problem": "3"}
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [0, 1]
    assert sampled_problems[0] == {"problem": "0"}
    assert sampled_problems[1] == {"problem": "1"}


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


def test_priority_buffer_sample_rollouts_discarded(priority_dataset, make_rollouts):
    buffer = PriorityPoolBuffer(priority_dataset, PriorityPoolBufferConfig())
    rollouts = make_rollouts(priority_dataset, rewards=[1.0, 1.0, 1.0, 1.0, 1.0], advantages=[0.0, 0.0, 0.0, 0.0, 0.0])
    buffer.update(rollouts)
    sampled_rollouts = buffer.sample_rollouts(5)
    assert sampled_rollouts == rollouts
    assert len(sampled_rollouts) == 10
    assert all(metadata["priority"] == "discarded" for metadata in buffer.metadata.values())


def test_priority_buffer_sample_rollouts_low(priority_dataset, make_rollouts):
    buffer = PriorityPoolBuffer(priority_dataset, PriorityPoolBufferConfig())
    rollouts = make_rollouts(priority_dataset, rewards=[0.0, 0.0, 0.0, 0.0, 0.0], advantages=[0.0, 0.0, 0.0, 0.0, 0.0])
    buffer.update(rollouts)
    sampled_rollouts = buffer.sample_rollouts(5)
    assert sampled_rollouts == rollouts
    assert len(sampled_rollouts) == 10
    assert all(metadata["priority"] == "low" for metadata in buffer.metadata.values())


def test_priority_buffer_sample_rollouts_high(priority_dataset, make_rollouts):
    buffer = PriorityPoolBuffer(priority_dataset, PriorityPoolBufferConfig())
    rollouts = make_rollouts(priority_dataset)
    buffer.update(rollouts)
    sampled_rollouts = buffer.sample_rollouts(5)
    assert sampled_rollouts == rollouts
    assert len(sampled_rollouts) == 10
    assert all(metadata["priority"] == "high" for metadata in buffer.metadata.values())


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
