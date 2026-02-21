import random
from unittest.mock import MagicMock

import pytest
import verifiers as vf
from datasets import Dataset

from prime_rl.orchestrator.buffer import Buffer
from prime_rl.orchestrator.config import BufferConfig, MaxTokensControllerConfig


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


# ── Max tokens controller tests ──────────────────────────────────────────


def _make_controller_rollouts(
    dataset: Dataset, rewards: list[float], is_truncated: list[bool] | bool = False
) -> list[vf.RolloutOutput]:
    """Helper to create rollouts with controllable truncation flags."""
    if isinstance(is_truncated, bool):
        is_truncated = [is_truncated] * len(rewards)
    all_rollouts = []
    for i, (reward, trunc) in enumerate(zip(rewards, is_truncated)):
        rollout = vf.RolloutOutput(
            example_id=dataset[i]["example_id"],
            task=dataset[i]["task"],
            prompt=dataset[i]["prompt"],
            prompt_ids=[0],
            prompt_mask=[1],
            completion_ids=[1],
            completion_mask=[1],
            completion_logprobs=[0.0],
            is_truncated=trunc,
            reward=reward,
            advantage=1.0,
            metrics={},
        )
        all_rollouts.append(rollout)
    return all_rollouts


@pytest.fixture
def single_env_group(mock_openai_client, dummy_dataset) -> vf.EnvGroup:
    env = vf.SingleTurnEnv(
        client=mock_openai_client,
        model="test-model",
        dataset=dummy_dataset,
        rubric=vf.Rubric(),
    )
    return vf.EnvGroup(envs=[env], env_names=["env_a"])


def test_max_tokens_controller_increase(single_env_group):
    """Increases when reward < target and truncation > 0."""
    dataset = single_env_group.get_dataset()
    ctrl = MaxTokensControllerConfig(
        target_reward=0.8, step_size=32, initial_max_tokens=256, min_max_tokens=64, max_max_tokens=1024
    )
    buffer = Buffer(dataset, single_env_group.env_names, BufferConfig(), max_tokens_controllers={"env_a": ctrl})

    assert buffer.get_max_tokens("env_a") == 256

    # Low reward + truncation -> should increase
    rollouts = _make_controller_rollouts(dataset.select(range(3)), rewards=[0.2, 0.3, 0.1], is_truncated=True)
    buffer.update(rollouts)
    metrics = buffer.get_metrics()

    assert buffer.get_max_tokens("env_a") == 288  # 256 + 32
    assert "max_tokens/env_a" in metrics
    assert "ema_reward/env_a" in metrics


def test_max_tokens_controller_decrease(single_env_group):
    """Decreases when reward > target and truncation rate <= threshold."""
    dataset = single_env_group.get_dataset()
    ctrl = MaxTokensControllerConfig(
        target_reward=0.5,
        step_size=32,
        initial_max_tokens=512,
        min_max_tokens=64,
        max_max_tokens=1024,
        truncation_threshold=0.1,
    )
    buffer = Buffer(dataset, single_env_group.env_names, BufferConfig(), max_tokens_controllers={"env_a": ctrl})

    # High reward + no truncation -> should decrease
    rollouts = _make_controller_rollouts(dataset.select(range(3)), rewards=[0.9, 0.8, 0.9], is_truncated=False)
    buffer.update(rollouts)
    buffer.get_metrics()

    assert buffer.get_max_tokens("env_a") == 480  # 512 - 32


def test_max_tokens_controller_no_increase_without_truncation(single_env_group):
    """Holds steady when reward < target but truncation == 0."""
    dataset = single_env_group.get_dataset()
    ctrl = MaxTokensControllerConfig(
        target_reward=0.8, step_size=32, initial_max_tokens=256, min_max_tokens=64, max_max_tokens=1024
    )
    buffer = Buffer(dataset, single_env_group.env_names, BufferConfig(), max_tokens_controllers={"env_a": ctrl})

    # Low reward but NO truncation -> should not increase
    rollouts = _make_controller_rollouts(dataset.select(range(3)), rewards=[0.2, 0.3, 0.1], is_truncated=False)
    buffer.update(rollouts)
    buffer.get_metrics()

    assert buffer.get_max_tokens("env_a") == 256


def test_max_tokens_controller_no_decrease_high_truncation(single_env_group):
    """Holds steady when reward > target but truncation rate > threshold."""
    dataset = single_env_group.get_dataset()
    ctrl = MaxTokensControllerConfig(
        target_reward=0.5,
        step_size=32,
        initial_max_tokens=512,
        min_max_tokens=64,
        max_max_tokens=1024,
        truncation_threshold=0.1,
    )
    buffer = Buffer(dataset, single_env_group.env_names, BufferConfig(), max_tokens_controllers={"env_a": ctrl})

    # High reward but high truncation -> should not decrease
    rollouts = _make_controller_rollouts(dataset.select(range(3)), rewards=[0.9, 0.8, 0.9], is_truncated=True)
    buffer.update(rollouts)
    buffer.get_metrics()

    assert buffer.get_max_tokens("env_a") == 512


def test_max_tokens_controller_clamping(single_env_group):
    """Respects min/max bounds."""
    dataset = single_env_group.get_dataset()

    # Test min clamping
    ctrl = MaxTokensControllerConfig(
        target_reward=0.5,
        step_size=100,
        initial_max_tokens=100,
        min_max_tokens=64,
        max_max_tokens=1024,
        truncation_threshold=0.1,
    )
    buffer = Buffer(dataset, single_env_group.env_names, BufferConfig(), max_tokens_controllers={"env_a": ctrl})
    rollouts = _make_controller_rollouts(dataset.select(range(3)), rewards=[0.9, 0.8, 0.9], is_truncated=False)
    buffer.update(rollouts)
    buffer.get_metrics()
    assert buffer.get_max_tokens("env_a") == 64  # 100 - 100 = 0 -> clamped to 64

    # Test max clamping
    ctrl2 = MaxTokensControllerConfig(
        target_reward=0.8, step_size=100, initial_max_tokens=980, min_max_tokens=64, max_max_tokens=1024
    )
    buffer2 = Buffer(dataset, single_env_group.env_names, BufferConfig(), max_tokens_controllers={"env_a": ctrl2})
    rollouts2 = _make_controller_rollouts(dataset.select(range(3)), rewards=[0.2, 0.3, 0.1], is_truncated=True)
    buffer2.update(rollouts2)
    buffer2.get_metrics()
    assert buffer2.get_max_tokens("env_a") == 1024  # 980 + 100 = 1080 -> clamped to 1024


def test_max_tokens_controller_ema(single_env_group):
    """EMA smooths the signal correctly over multiple updates."""
    dataset = single_env_group.get_dataset()
    ctrl = MaxTokensControllerConfig(
        target_reward=0.5,
        step_size=32,
        momentum=0.5,
        initial_max_tokens=512,
        min_max_tokens=64,
        max_max_tokens=1024,
        truncation_threshold=0.1,
    )
    buffer = Buffer(dataset, single_env_group.env_names, BufferConfig(), max_tokens_controllers={"env_a": ctrl})

    # Step 1: reward=0.2, truncation=True -> ema seeds to 0.2 (< 0.5 target) -> increase
    rollouts = _make_controller_rollouts(dataset.select(range(1)), rewards=[0.2], is_truncated=True)
    buffer.update(rollouts)
    buffer.get_metrics()
    assert buffer.ema_reward_per_env["env_a"] == pytest.approx(0.2)
    assert buffer.get_max_tokens("env_a") == 544  # 512 + 32

    # Step 2: reward=0.8, truncation=False -> ema = 0.5*0.2 + 0.5*0.8 = 0.5
    # ema == target (not > target), so no decrease
    rollouts = _make_controller_rollouts(dataset.select(range(1)), rewards=[0.8], is_truncated=False)
    buffer.update(rollouts)
    buffer.get_metrics()
    assert buffer.ema_reward_per_env["env_a"] == pytest.approx(0.5)
    assert buffer.get_max_tokens("env_a") == 544  # unchanged

    # Step 3: reward=0.9, truncation=False -> ema = 0.5*0.5 + 0.5*0.9 = 0.7 > 0.5 target -> decrease
    rollouts = _make_controller_rollouts(dataset.select(range(1)), rewards=[0.9], is_truncated=False)
    buffer.update(rollouts)
    buffer.get_metrics()
    assert buffer.ema_reward_per_env["env_a"] == pytest.approx(0.7)
    assert buffer.get_max_tokens("env_a") == 512  # 544 - 32


def test_max_tokens_controller_save_load(single_env_group, tmp_path):
    """State persists across save/load."""
    dataset = single_env_group.get_dataset()
    ctrl = MaxTokensControllerConfig(
        target_reward=0.8, step_size=32, initial_max_tokens=256, min_max_tokens=64, max_max_tokens=1024
    )
    buffer = Buffer(dataset, single_env_group.env_names, BufferConfig(), max_tokens_controllers={"env_a": ctrl})

    # Trigger a step to get some ema state
    rollouts = _make_controller_rollouts(dataset.select(range(3)), rewards=[0.2, 0.3, 0.1], is_truncated=True)
    buffer.update(rollouts)
    buffer.get_metrics()

    saved_max_tokens = buffer.get_max_tokens("env_a")
    saved_ema = buffer.ema_reward_per_env["env_a"]

    buffer.save(tmp_path / "buffer")

    # Load into new buffer
    new_buffer = Buffer(dataset, single_env_group.env_names, BufferConfig(), max_tokens_controllers={"env_a": ctrl})
    new_buffer.load(tmp_path / "buffer")

    assert new_buffer.get_max_tokens("env_a") == saved_max_tokens
    assert new_buffer.ema_reward_per_env["env_a"] == pytest.approx(saved_ema)


def test_max_tokens_controller_per_env(dummy_env_group):
    """Only active for envs with config; others return None from get_max_tokens."""
    dataset = dummy_env_group.get_dataset()
    ctrl = MaxTokensControllerConfig(
        target_reward=0.7, step_size=32, initial_max_tokens=256, min_max_tokens=64, max_max_tokens=1024
    )
    buffer = Buffer(dataset, dummy_env_group.env_names, BufferConfig(), max_tokens_controllers={"env_a": ctrl})

    assert buffer.get_max_tokens("env_a") == 256
    assert buffer.get_max_tokens("env_b") is None


def test_max_tokens_controller_validation():
    """Config validators reject invalid bounds."""
    from pydantic import ValidationError

    # min > max
    with pytest.raises(ValidationError, match="min_max_tokens must be <= max_max_tokens"):
        MaxTokensControllerConfig(target_reward=0.5, min_max_tokens=1024, max_max_tokens=64)

    # initial out of bounds
    with pytest.raises(ValidationError, match="initial_max_tokens must be within"):
        MaxTokensControllerConfig(target_reward=0.5, initial_max_tokens=2048, min_max_tokens=64, max_max_tokens=1024)

    # initial below min
    with pytest.raises(ValidationError, match="initial_max_tokens must be within"):
        MaxTokensControllerConfig(target_reward=0.5, initial_max_tokens=32, min_max_tokens=64, max_max_tokens=1024)


def test_max_tokens_controller_initial_from_sampling(single_env_group):
    """Falls back to initial_max_tokens from sampling config when controller.initial_max_tokens is None."""
    dataset = single_env_group.get_dataset()
    ctrl = MaxTokensControllerConfig(target_reward=0.7, step_size=32, min_max_tokens=64, max_max_tokens=1024)
    buffer = Buffer(
        dataset,
        single_env_group.env_names,
        BufferConfig(),
        max_tokens_controllers={"env_a": ctrl},
        initial_max_tokens=512,
    )
    assert buffer.get_max_tokens("env_a") == 512
