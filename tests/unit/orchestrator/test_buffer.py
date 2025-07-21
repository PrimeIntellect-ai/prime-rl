from copy import deepcopy

import pytest
from datasets import Dataset

from prime_rl.orchestrator.buffer import PriorityPoolBuffer, Rollout, SimpleBuffer
from prime_rl.orchestrator.config import PriorityPoolBufferConfig, SimpleBufferConfig


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

    def _make_rollouts(ds: Dataset) -> list[Rollout]:
        rollouts = []
        for i, _ in enumerate(ds):
            problem_rollouts = [
                Rollout(
                    problem_id=i,
                    prompt_tokens=[0],
                    prompt_mask=[1],
                    completion_tokens=[1],
                    completion_mask=[1],
                    completion_logprobs=[0.0],
                    reward=0.0,
                    advantage=0.0,
                )
            ] * 2
            rollouts.extend(problem_rollouts)
        return rollouts

    return _make_rollouts


def test_simple_buffer_init(dataset):
    SimpleBuffer(dataset, SimpleBufferConfig())


def test_priority_pool_buffer_init(priority_dataset):
    PriorityPoolBuffer(priority_dataset, PriorityPoolBufferConfig())


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


def test_simple_buffer_sample_rollouts(dataset, make_rollouts):
    buffer = SimpleBuffer(dataset, SimpleBufferConfig())
    rollouts = make_rollouts(dataset)
    buffer.update(rollouts)
    assert buffer.sample_rollouts(5) == rollouts


def test_priority_buffer_sample_rollouts(priority_dataset, make_rollouts):
    buffer = PriorityPoolBuffer(priority_dataset, PriorityPoolBufferConfig())
    rollouts = make_rollouts(priority_dataset)
    buffer.update(rollouts)
    assert buffer.sample_rollouts(5) == rollouts


@pytest.mark.parametrize("n", [1, 4, 6, 10])
def test_simple_buffer_sample_invalid_rollouts(dataset, make_rollouts, n):
    buffer = SimpleBuffer(dataset, SimpleBufferConfig())
    rollouts = make_rollouts(dataset)
    buffer.update(rollouts)
    with pytest.raises(AssertionError):
        buffer.sample_rollouts(n)
