import random

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
            {"example_id": 0, "problem": "0", "task": "env_a"},
            {"example_id": 1, "problem": "1", "task": "env_a"},
            {"example_id": 2, "problem": "2", "task": "env_a"},
            {"example_id": 3, "problem": "3", "task": "env_a"},
            {"example_id": 4, "problem": "4", "task": "env_a"},
        ]
    )


@pytest.fixture
def multi_env_dataset() -> Dataset:
    return Dataset.from_list(
        [
            {"example_id": 0, "problem": "0", "task": "env_a"},
            {"example_id": 1, "problem": "1", "task": "env_a"},
            {"example_id": 2, "problem": "2", "task": "env_a"},
            {"example_id": 3, "problem": "3", "task": "env_b"},
            {"example_id": 4, "problem": "4", "task": "env_b"},
        ]
    )


@pytest.fixture
def make_rollouts():
    def _make_rollouts(dataset: Dataset, rewards: list[float]) -> list[Rollout]:
        rollouts = []
        for i, reward in enumerate(rewards):
            task = dataset[i]["task"]
            problem_rollouts = [
                vf.State(
                    example_id=i,
                    task=task,
                    prompt_ids=[0],
                    prompt_mask=[1],
                    completion_ids=[1],
                    completion_mask=[1],
                    completion_logprobs=[0.0],
                    is_truncated=False,
                    reward=reward,
                    advantage=1.0,
                    metrics={},
                )
            ] * 2
            rollouts.extend(problem_rollouts)
        return rollouts

    return _make_rollouts


def _get_normal_ids(buffer: Buffer) -> set[int]:
    return {pid for env in buffer.problem_buffer.values() for pid in env.keys()}


def test_buffer_init_and_sample(dataset):
    buffer = Buffer(dataset, BufferConfig(), ["env_a"])
    assert len(buffer.problem_buffer["env_a"]) == 5
    samples = buffer.sample_problems(2)
    assert len(samples) == 2


def test_buffer_problem_pool_assignment(dataset, make_rollouts):
    """Problems are moved to easy/hard pools based on reward thresholds."""
    buffer = Buffer(dataset, BufferConfig(easy_threshold=1.0, hard_threshold=0.0), ["env_a"])
    buffer.update(make_rollouts(dataset, rewards=[1.0, 1.0, 0.5, 0.5, 0.0]))

    assert buffer.easy_problems.keys() == {0, 1}
    assert buffer.hard_problems.keys() == {4}
    assert _get_normal_ids(buffer) == {2, 3}


def test_buffer_online_difficulty_filtering(dataset, make_rollouts):
    """With online_difficulty_filtering=True, only partial reward rollouts are kept."""
    buffer = Buffer(
        dataset,
        BufferConfig(online_difficulty_filtering=True),
        ["env_a"],
    )
    buffer.update(make_rollouts(dataset, rewards=[1.0, 0.5, 0.0, 0.5, 0.5]))

    # Only 3 problems with reward 0.5 -> 6 rollouts kept
    assert len(buffer.rollout_buffer) == 6


def test_buffer_no_filtering_by_default(dataset, make_rollouts):
    """With online_difficulty_filtering=False (default), all rollouts are kept."""
    buffer = Buffer(dataset, BufferConfig(), ["env_a"])
    buffer.update(make_rollouts(dataset, rewards=[1.0, 0.5, 0.0, 0.5, 0.5]))

    # All 5 problems -> 10 rollouts kept
    assert len(buffer.rollout_buffer) == 10


def test_buffer_save_load_with_conversion(dataset, make_rollouts, tmp_path):
    """Easy/hard problems are partially converted to normal on load."""
    buffer = Buffer(dataset, BufferConfig(easy_threshold=1.0, hard_threshold=0.0), ["env_a"])
    buffer.update(make_rollouts(dataset, rewards=[1.0, 1.0, 0.5, 0.5, 0.0]))
    buffer.save(tmp_path / "buffer")

    new_buffer = Buffer(dataset, BufferConfig(easy_fraction=0.5), ["env_a"])
    new_buffer.load(tmp_path / "buffer")

    # 1 of 2 easy problems converted to normal
    assert len(new_buffer.easy_problems) == 1
    assert len(_get_normal_ids(new_buffer)) == 3


def test_buffer_multi_env(multi_env_dataset):
    buffer = Buffer(multi_env_dataset, BufferConfig(env_probabilities=[0.8, 0.2]), ["env_a", "env_b"])
    assert len(buffer.problem_buffer["env_a"]) == 3
    assert len(buffer.problem_buffer["env_b"]) == 2

    samples = buffer.sample_problems(100)
    env_a_count = sum(1 for p in samples if p["task"] == "env_a")
    assert 60 <= env_a_count <= 95


def test_buffer_env_probabilities_validation(multi_env_dataset):
    with pytest.raises(AssertionError, match="env_probabilities length"):
        Buffer(multi_env_dataset, BufferConfig(env_probabilities=[0.5, 0.3, 0.2]), ["env_a", "env_b"])

    with pytest.raises(AssertionError, match="must sum to 1.0"):
        Buffer(multi_env_dataset, BufferConfig(env_probabilities=[0.5, 0.3]), ["env_a", "env_b"])
