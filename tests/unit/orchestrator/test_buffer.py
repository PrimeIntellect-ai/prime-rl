import random
from unittest.mock import MagicMock

import pytest
import verifiers as vf
from datasets import Dataset

from prime_rl.orchestrator.buffer import Buffer
from prime_rl.orchestrator.config import BufferConfig


@pytest.fixture(autouse=True)
def set_seed():
    random.seed(42)


@pytest.fixture
def mock_openai_client():
    """Return a mocked OpenAI client."""
    return MagicMock()


@pytest.fixture
def dummy_dataset() -> Dataset:
    """Return a dummy dataset with 5 examples."""
    return Dataset.from_dict(
        {
            "question": ["q0", "q1", "q2", "q3", "q4"],
            "answer": ["a0", "a1", "a2", "a3", "a4"],
        }
    )


@pytest.fixture
def dummy_env_group(mock_openai_client, dummy_dataset) -> vf.EnvGroup:
    """Return an EnvGroup with two dummy envs using the same dataset."""
    env_a = vf.SingleTurnEnv(
        client=mock_openai_client,
        model="test-model",
        dataset=dummy_dataset,
        rubric=vf.Rubric(),
    )
    env_b = vf.SingleTurnEnv(
        client=mock_openai_client,
        model="test-model",
        dataset=dummy_dataset,
        rubric=vf.Rubric(),
    )
    return vf.EnvGroup(envs=[env_a, env_b], env_names=["env_a", "env_b"])


@pytest.fixture
def make_rollouts():
    def _make_rollouts(dataset: Dataset, rewards: list[float]) -> list[vf.RolloutOutput]:
        all_rollouts = []
        for i, reward in enumerate(rewards):
            task = dataset[i]["task"]
            example_id = dataset[i]["example_id"]
            prompt = dataset[i]["prompt"]
            rollouts = [
                vf.RolloutOutput(
                    example_id=example_id,
                    task=task,
                    prompt=prompt,
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
            all_rollouts.extend(rollouts)
        return all_rollouts

    return _make_rollouts


def get_normal_ids(buffer: Buffer) -> set[int]:
    return {example_id for env in buffer.example_buffer.values() for example_id in env.keys()}


def test_buffer_init_and_sample(dummy_env_group):
    dataset = dummy_env_group.get_dataset()
    buffer = Buffer(dataset, dummy_env_group.env_names, BufferConfig())
    # Each env has 5 examples, so total is 10
    assert len(buffer.example_buffer["env_a"]) == 5
    assert len(buffer.example_buffer["env_b"]) == 5
    samples = buffer.sample_examples(2)
    assert len(samples) == 2


def test_buffer_problem_pool_assignment(dummy_env_group, make_rollouts):
    """Problems are moved to easy/hard pools based on reward thresholds."""
    dataset = dummy_env_group.get_dataset()
    buffer = Buffer(dataset, dummy_env_group.env_names, BufferConfig(easy_threshold=1.0, hard_threshold=0.0))
    dataset = buffer.dataset
    # Use first 5 examples (all from env_a since they come first in concatenated dataset)
    buffer.update(make_rollouts(dataset.select(range(5)), rewards=[1.0, 1.0, 0.5, 0.5, 0.0]))

    assert len(buffer.easy_examples) == 2
    assert len(buffer.hard_examples) == 1
    # 2 normal from first 5, plus 5 from env_b = 7
    assert len(get_normal_ids(buffer)) == 7


def test_buffer_online_difficulty_filtering(dummy_env_group, make_rollouts):
    """With online_difficulty_filtering=True, only partial reward rollouts are kept."""
    dataset = dummy_env_group.get_dataset()
    buffer = Buffer(
        dataset,
        dummy_env_group.env_names,
        BufferConfig(online_difficulty_filtering=True),
    )
    buffer.update(make_rollouts(dataset.select(range(5)), rewards=[1.0, 0.5, 0.0, 0.5, 0.5]))

    # Only 3 problems with reward 0.5 -> 6 rollouts kept
    assert len(buffer.rollout_buffer) == 6


def test_buffer_no_filtering_by_default(dummy_env_group, make_rollouts):
    """With online_difficulty_filtering=False (default), all rollouts are kept."""
    dataset = dummy_env_group.get_dataset()
    buffer = Buffer(dataset, dummy_env_group.env_names, BufferConfig())
    buffer.update(make_rollouts(dataset.select(range(5)), rewards=[1.0, 0.5, 0.0, 0.5, 0.5]))

    # All 5 problems -> 10 rollouts kept
    assert len(buffer.rollout_buffer) == 10


def test_buffer_save_load_with_conversion(dummy_env_group, make_rollouts, tmp_path):
    """Easy/hard problems are partially converted to normal on load."""
    dataset = dummy_env_group.get_dataset()
    buffer = Buffer(dataset, dummy_env_group.env_names, BufferConfig(easy_threshold=1.0, hard_threshold=0.0))
    buffer.update(make_rollouts(dataset.select(range(5)), rewards=[1.0, 1.0, 0.5, 0.5, 0.0]))
    buffer.save(tmp_path / "buffer")

    new_buffer = Buffer(
        dataset, dummy_env_group.env_names, BufferConfig(easy_fraction=0.5, hash_keys=["prompt", "task"])
    )
    new_buffer.load(tmp_path / "buffer")

    # 1 of 2 easy problems converted to normal
    assert len(new_buffer.easy_examples) == 1
    # 2 were normal + 5 from env_b + 1 converted from easy = 8
    assert len(get_normal_ids(new_buffer)) == 8


def test_buffer_env_ratios(dummy_env_group):
    dataset = dummy_env_group.get_dataset()
    buffer = Buffer(dataset, dummy_env_group.env_names, BufferConfig(env_ratios=[0.8, 0.2]))
    assert len(buffer.example_buffer["env_a"]) == 5
    assert len(buffer.example_buffer["env_b"]) == 5

    samples = buffer.sample_examples(100)
    env_a_count = sum(1 for p in samples if p["task"] == "env_a")
    assert 60 <= env_a_count <= 95


def test_buffer_env_ratios_validation():
    """BufferConfig validates that all env_ratios are positive."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="All env_ratios must be positive"):
        BufferConfig(env_ratios=[0.5, -0.3, 0.2])


def test_buffer_no_cross_env_pool_assignment(mock_openai_client, tmp_path):
    """Pool assignments don't transfer if example_id exists but task/env changed."""
    # Original: create an env_group with only env_a
    original_dataset = Dataset.from_dict({"question": ["q0"], "answer": ["a0"]})
    original_env = vf.SingleTurnEnv(
        client=mock_openai_client,
        model="test-model",
        dataset=original_dataset,
        rubric=vf.Rubric(),
    )
    original_env_group = vf.EnvGroup(envs=[original_env], env_names=["env_a"])
    original_env_dataset = original_env_group.get_dataset()

    buffer = Buffer(original_env_dataset, original_env_group.env_names, BufferConfig(easy_threshold=1.0))
    # Manually move the example to easy pool
    example_id = list(buffer.example_buffer["env_a"].keys())[0]
    example = buffer.example_buffer["env_a"].pop(example_id)
    buffer.easy_examples.append(example)
    buffer.save(tmp_path / "buffer")

    # Resume: create a new env_group with different dataset but similar structure
    new_dataset = Dataset.from_dict({"question": ["different_q"], "answer": ["different_a"]})
    new_env = vf.SingleTurnEnv(
        client=mock_openai_client,
        model="test-model",
        dataset=new_dataset,
        rubric=vf.Rubric(),
    )
    new_env_group = vf.EnvGroup(envs=[new_env], env_names=["env_b"])
    new_env_dataset = new_env_group.get_dataset()

    new_buffer = Buffer(new_env_dataset, new_env_group.env_names, BufferConfig())
    new_buffer.load(tmp_path / "buffer")

    # Should NOT be in easy pool (different content, different hash)
    assert len(new_buffer.easy_examples) == 0
    # Should still be in normal pool for env_b
    assert len(new_buffer.example_buffer["env_b"]) == 1
