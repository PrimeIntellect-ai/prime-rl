import pytest
from datasets import Dataset

from prime_rl.orchestrator.buffer import Rollout, SimpleBuffer
from prime_rl.orchestrator.config import SimpleBufferConfig


@pytest.fixture
def dataset():
    return Dataset.from_list(
        [
            {"problem": "1 + 1", "solution": "2"},
            {"problem": "2 + 2", "solution": "4"},
            {"problem": "3 + 3", "solution": "6"},
        ]
    )


@pytest.fixture
def rollouts():
    return [
        Rollout(
            problem_id=0,
            prompt_tokens=[],
            prompt_mask=[],
            completion_tokens=[],
            completion_mask=[],
            completion_logprobs=[],
            reward=0.0,
            advantage=0.0,
        ),
        Rollout(
            problem_id=0,
            prompt_tokens=[],
            prompt_mask=[],
            completion_tokens=[],
            completion_mask=[],
            completion_logprobs=[],
            reward=1.0,
            advantage=1.0,
        ),
        Rollout(
            problem_id=1,
            prompt_tokens=[],
            prompt_mask=[],
            completion_tokens=[],
            completion_mask=[],
            completion_logprobs=[],
            reward=0.0,
            advantage=0.0,
        ),
        Rollout(
            problem_id=1,
            prompt_tokens=[],
            prompt_mask=[],
            completion_tokens=[],
            completion_mask=[],
            completion_logprobs=[],
            reward=1.0,
            advantage=1.0,
        ),
    ]


def test_simple_buffer_init(dataset):
    SimpleBuffer(dataset, SimpleBufferConfig())


def test_simple_buffer_sample_problems(dataset):
    buffer = SimpleBuffer(dataset, SimpleBufferConfig())
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [0, 1]
    assert sampled_problems[0] == {"problem": "1 + 1", "solution": "2"}
    assert sampled_problems[1] == {"problem": "2 + 2", "solution": "4"}


def test_simple_buffer_sample_problems_multiple_epochs(dataset):
    buffer = SimpleBuffer(dataset, SimpleBufferConfig())
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [0, 1]
    assert sampled_problems[0] == {"problem": "1 + 1", "solution": "2"}
    assert sampled_problems[1] == {"problem": "2 + 2", "solution": "4"}
    sampled_problem_ids, sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problem_ids) == len(sampled_problems) == 2
    assert sampled_problem_ids == [0, 1]
    assert sampled_problems[0] == {"problem": "1 + 1", "solution": "2"}
    assert sampled_problems[1] == {"problem": "2 + 2", "solution": "4"}


def test_simple_buffer_sample_rollouts(dataset, rollouts):
    buffer = SimpleBuffer(dataset, SimpleBufferConfig())
    buffer.update(rollouts)
    assert buffer.sample_rollouts(2) == rollouts


@pytest.mark.parametrize("n", [1, 4])
def test_simple_buffer_sample_invalid_rollouts(dataset, rollouts, n):
    buffer = SimpleBuffer(dataset, SimpleBufferConfig())
    buffer.update(rollouts)
    with pytest.raises(AssertionError):
        buffer.sample_rollouts(n)
