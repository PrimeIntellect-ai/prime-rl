import random

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
    def _make_rollouts(
        dataset: Dataset, rewards: list[float] | None = None, advantages: list[float] | None = None
    ) -> list[Rollout]:
        rollouts = []
        rewards = rewards or [1.0] * len(dataset)
        advantages = advantages or [1.0] * len(dataset)
        for i, (reward, advantage) in enumerate(zip(rewards, advantages)):
            task = dataset[i]["task"]
            problem_rollouts = [
                Rollout(
                    example_id=i,
                    task=task,
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


def _get_all_normal_ids(buffer: Buffer) -> set[int]:
    all_ids = set()
    for env_problems in buffer.problem_buffer.values():
        all_ids.update(env_problems.keys())
    return all_ids


def test_buffer_init(dataset):
    Buffer(dataset, BufferConfig(), ["env_a"])


def test_buffer_sample_problems(dataset):
    buffer = Buffer(dataset, BufferConfig(), ["env_a"])
    sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problems) == 2
    sampled_ids = {p["example_id"] for p in sampled_problems}
    assert sampled_ids.issubset({0, 1, 2, 3, 4})


def test_buffer_sample_problems_only_normal(dataset, make_rollouts):
    buffer = Buffer(dataset, BufferConfig(easy_threshold=1.0, hard_threshold=0.0), ["env_a"])
    rollouts = make_rollouts(dataset, rewards=[1.0, 1.0, 0.5, 0.5, 0.0])
    buffer.update(rollouts)

    sampled_problems = buffer.sample_problems(2)
    assert len(sampled_problems) == 2
    sampled_ids = [p["example_id"] for p in sampled_problems]
    assert all(pid in [2, 3] for pid in sampled_ids)

    assert 0 in buffer.easy_problems
    assert 1 in buffer.easy_problems
    assert 4 in buffer.hard_problems
    assert 2 in _get_all_normal_ids(buffer)
    assert 3 in _get_all_normal_ids(buffer)


def test_buffer_sample_rollouts(dataset, make_rollouts):
    buffer = Buffer(dataset, BufferConfig(), ["env_a"])
    rollouts = make_rollouts(dataset, rewards=[0.5] * len(dataset))
    buffer.update(rollouts)
    sampled_rollouts = buffer.sample_rollouts(10)
    assert sampled_rollouts == rollouts
    assert len(sampled_rollouts) == 10


def test_buffer_sample_rollouts_more_than_available(dataset, make_rollouts):
    buffer = Buffer(dataset, BufferConfig(), ["env_a"])
    rollouts = make_rollouts(dataset, rewards=[0.5] * len(dataset))
    buffer.update(rollouts)
    sampled_rollouts = buffer.sample_rollouts(20)
    assert len(sampled_rollouts) == 10
    assert len(buffer.rollout_buffer) == 0


def test_buffer_online_difficulty_filtering(dataset, make_rollouts):
    buffer = Buffer(dataset, BufferConfig(easy_threshold=1.0, hard_threshold=0.0), ["env_a"])
    rollouts = make_rollouts(dataset, rewards=[1.0, 0.5, 0.0, 0.5, 0.5])
    buffer.update(rollouts)

    assert len(buffer.rollout_buffer) == 6
    assert 0 in buffer.easy_problems
    assert 1 in _get_all_normal_ids(buffer)
    assert 2 in buffer.hard_problems
    assert 3 in _get_all_normal_ids(buffer)
    assert 4 in _get_all_normal_ids(buffer)


def test_buffer_convert_difficulty_pools(dataset, make_rollouts, tmp_path):
    buffer = Buffer(
        dataset,
        BufferConfig(easy_threshold=1.0, hard_threshold=0.0, easy_to_normal_fraction=0.5, hard_to_normal_fraction=0.5),
        ["env_a"],
    )
    rollouts = make_rollouts(dataset, rewards=[1.0, 1.0, 0.5, 0.5, 0.0])
    buffer.update(rollouts)

    assert 0 in buffer.easy_problems
    assert 1 in buffer.easy_problems
    assert 4 in buffer.hard_problems
    assert 2 in _get_all_normal_ids(buffer)
    assert 3 in _get_all_normal_ids(buffer)

    buffer_path = tmp_path / "buffer"
    buffer.save(buffer_path)

    new_buffer = Buffer(dataset, BufferConfig(easy_to_normal_fraction=0.5, hard_to_normal_fraction=0.5), ["env_a"])
    new_buffer.load(buffer_path)

    easy_converted = sum(1 for pid in [0, 1] if pid in _get_all_normal_ids(new_buffer))
    assert easy_converted == 1
    assert 4 in new_buffer.hard_problems
    assert 2 in _get_all_normal_ids(new_buffer)
    assert 3 in _get_all_normal_ids(new_buffer)
    assert len(_get_all_normal_ids(new_buffer)) == 3

    sampled = new_buffer.sample_problems(3)
    assert len(sampled) == 3


def test_buffer_multi_env_init(multi_env_dataset):
    buffer = Buffer(multi_env_dataset, BufferConfig(), ["env_a", "env_b"])
    assert buffer.env_names == ["env_a", "env_b"]
    assert len(buffer.problem_buffer["env_a"]) == 3
    assert len(buffer.problem_buffer["env_b"]) == 2


def test_buffer_multi_env_uniform_sampling(multi_env_dataset):
    buffer = Buffer(multi_env_dataset, BufferConfig(seed=42), ["env_a", "env_b"])
    samples = buffer.sample_problems(5)
    assert len(samples) == 5
    assert all(p["task"] in ["env_a", "env_b"] for p in samples)


def test_buffer_multi_env_probability_sampling(multi_env_dataset):
    buffer = Buffer(multi_env_dataset, BufferConfig(env_probabilities=[0.8, 0.2], seed=42), ["env_a", "env_b"])
    # Sample enough to check distribution
    samples = buffer.sample_problems(100)
    assert len(samples) == 100
    env_a_count = sum(1 for p in samples if p["task"] == "env_a")
    # Should be roughly 80%, allow some variance
    assert 60 <= env_a_count <= 95


def test_buffer_multi_env_save_load(multi_env_dataset, make_rollouts, tmp_path):
    buffer = Buffer(multi_env_dataset, BufferConfig(easy_threshold=1.0, hard_threshold=0.0), ["env_a", "env_b"])
    rollouts = make_rollouts(multi_env_dataset, rewards=[1.0, 0.5, 0.5, 0.0, 0.5])
    buffer.update(rollouts)

    assert 0 in buffer.easy_problems
    assert 1 in buffer.problem_buffer["env_a"]
    assert 2 in buffer.problem_buffer["env_a"]
    assert 3 in buffer.hard_problems
    assert 4 in buffer.problem_buffer["env_b"]

    buffer_path = tmp_path / "buffer"
    buffer.save(buffer_path)

    new_buffer = Buffer(multi_env_dataset, BufferConfig(), ["env_a", "env_b"])
    new_buffer.load(buffer_path)

    assert 0 in new_buffer.easy_problems
    assert 1 in new_buffer.problem_buffer["env_a"]
    assert 2 in new_buffer.problem_buffer["env_a"]
    assert 3 in new_buffer.hard_problems
    assert 4 in new_buffer.problem_buffer["env_b"]


def test_buffer_multi_env_metrics(multi_env_dataset):
    buffer = Buffer(multi_env_dataset, BufferConfig(), ["env_a", "env_b"])
    buffer.sample_problems(3)
    metrics = buffer.get_metrics()
    # Per-env metrics for sampled rollouts per pool
    # evicted_problems metrics only appear when there are rollouts from update()
    assert "buffer/evicted_problems/easy/env_a" not in metrics
    assert "buffer/evicted_problems/hard/env_a" not in metrics


def test_buffer_env_probabilities_validation(multi_env_dataset):
    with pytest.raises(AssertionError, match="env_probabilities length"):
        Buffer(multi_env_dataset, BufferConfig(env_probabilities=[0.5, 0.3, 0.2]), ["env_a", "env_b"])

    with pytest.raises(AssertionError, match="must sum to 1.0"):
        Buffer(multi_env_dataset, BufferConfig(env_probabilities=[0.5, 0.3]), ["env_a", "env_b"])
