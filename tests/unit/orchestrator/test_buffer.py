import random
from unittest.mock import MagicMock

import pytest
import verifiers as vf
from datasets import Dataset

from prime_rl.configs.orchestrator import BufferConfig, DefaultAdvantageConfig, EnvConfig
from prime_rl.orchestrator.buffer import Buffer
from prime_rl.orchestrator.envs import Envs, TrainEnv
from prime_rl.orchestrator.filters import ZeroAdvantageFilter


def make_env(name: str, vf_env: vf.Environment, **config_kwargs) -> TrainEnv:
    """Create a TrainEnv without calling vf.load_environment."""
    config = EnvConfig(id=name, name=name, **config_kwargs)
    env = TrainEnv.__new__(TrainEnv)
    env.config = config
    env._env = vf_env
    env._env_client = None
    env._env_server_process = None
    env.sampling_args = {}
    return env


def make_envs(env_dict: dict[str, TrainEnv]) -> Envs:
    """Create an Envs container from a dict of Env instances."""
    envs = Envs.__new__(Envs)
    envs._envs = env_dict
    return envs


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
def dummy_envs(mock_openai_client, dummy_dataset) -> Envs:
    """Return an Envs with two dummy envs."""
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
    return make_envs(
        {
            "env_a": make_env("env_a", env_a),
            "env_b": make_env("env_b", env_b),
        }
    )


@pytest.fixture
def make_rollouts():
    def _make_rollouts(
        buffer: Buffer, env_name: str, indices: list[int], rewards: list[float | list[float]]
    ) -> list[vf.RolloutOutput]:
        all_rollouts = []
        eb = buffer.env_buffers[env_name]
        examples = list(eb.examples.values())
        for idx, reward in zip(indices, rewards):
            example = examples[idx]
            rollouts = [
                vf.RolloutOutput(
                    example_id=example["example_id"],
                    task=example["env_name"],
                    prompt=example["prompt"],
                    prompt_ids=[0],
                    prompt_mask=[1],
                    completion_ids=[1],
                    completion_mask=[1],
                    completion_logprobs=[0.0],
                    is_truncated=False,
                    reward=r,
                    advantage=1.0,
                    metrics={},
                )
                for r in (reward if isinstance(reward, list) else [reward, reward])
            ]
            for r in rollouts:
                r["env_name"] = env_name
            all_rollouts.extend(rollouts)
        return all_rollouts

    return _make_rollouts


def get_normal_count(buffer: Buffer) -> int:
    return sum(eb.num_normal for eb in buffer.env_buffers.values())


def test_buffer_init_and_sample(dummy_envs):
    buffer = Buffer(dummy_envs, BufferConfig())
    assert buffer.env_buffers["env_a"].num_normal == 5
    assert buffer.env_buffers["env_b"].num_normal == 5
    samples = buffer.sample_examples(2)
    assert len(samples) == 2


def test_buffer_problem_pool_assignment(dummy_envs, make_rollouts):
    """Problems are moved to easy/hard pools based on reward thresholds."""
    buffer = Buffer(dummy_envs, BufferConfig(easy_threshold=1.0, hard_threshold=0.0))
    buffer.update(make_rollouts(buffer, "env_a", list(range(5)), rewards=[1.0, 1.0, 0.5, 0.5, 0.0]))

    assert len(buffer.env_buffers["env_a"].easy_examples) == 2
    assert len(buffer.env_buffers["env_a"].hard_examples) == 1
    # 2 normal from env_a + 5 from env_b = 7
    assert get_normal_count(buffer) == 7


def test_buffer_online_difficulty_filtering(dummy_envs, make_rollouts):
    """With online_difficulty_filtering=True, only non-zero-std reward groups are kept."""
    buffer = Buffer(
        dummy_envs,
        BufferConfig(easy_threshold=1.0, hard_threshold=0.0, online_difficulty_filtering=True),
    )
    buffer.update(
        make_rollouts(
            buffer,
            "env_a",
            list(range(5)),
            rewards=[
                [1.0, 1.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.5, 0.5],
                [0.0, 0.5],
            ],
        )
    )

    # Only the two groups with reward variance provide GRPO signal.
    assert len(buffer.rollout_buffer) == 4
    assert buffer.get_metrics()["filtered_rollouts/zero_std"] == pytest.approx(0.6)


def test_buffer_can_filter_groups_above_half_solve_rate(dummy_envs, make_rollouts):
    """REINFORCE1/2 trains on groups with solve rate <= 0.5."""
    buffer = Buffer(
        dummy_envs,
        BufferConfig(easy_threshold=0.5000001, online_difficulty_filtering=True),
    )
    buffer.update(make_rollouts(buffer, "env_a", list(range(4)), rewards=[0.0, 0.5, 0.75, 1.0]))

    # All four groups are zero-std and therefore provide no GRPO signal. The
    # 0.75 and 1.0 groups are also marked easy for future sampling.
    assert len(buffer.rollout_buffer) == 0
    assert len(buffer.env_buffers["env_a"].easy_examples) == 2


def test_zero_advantage_filter_flags_uniform_reward_groups_without_pool_eviction(dummy_envs):
    """ZeroAdvantageFilter handles within-group zero advantages after buffer admission."""
    from prime_rl.orchestrator.advantage import compute_advantages
    from prime_rl.orchestrator.filters import apply_filters

    buffer = Buffer(
        dummy_envs,
        BufferConfig(easy_threshold=1.0, online_difficulty_filtering=True),
    )

    def make_group(idx: int, rewards: list[float]) -> list[vf.RolloutOutput]:
        example = list(buffer.env_buffers["env_a"].examples.values())[idx]
        rollouts = []
        for reward in rewards:
            rollout = vf.RolloutOutput(
                example_id=example["example_id"],
                task=example["env_name"],
                prompt=example["prompt"],
                prompt_ids=[0],
                prompt_mask=[1],
                completion_ids=[1],
                completion_mask=[1],
                completion_logprobs=[0.0],
                is_truncated=False,
                reward=reward,
                advantage=0.0,
                metrics={},
                trajectory=[{"tokens": {"completion_ids": [1], "completion_logprobs": [0.0]}}],
                group_id=idx,
            )
            rollout["env_name"] = "env_a"
            rollouts.append(rollout)
        return rollouts

    rollouts = []
    rollouts.extend(make_group(0, [0.0, 0.0]))  # all-zero: filtered by buffer (zero std)
    rollouts.extend(make_group(1, [1.0, 1.0]))  # all-one: filtered by buffer (easy + zero std)
    rollouts.extend(make_group(2, [0.0, 0.5, 1.0]))  # mixed: one rollout has zero advantage
    rollouts.extend(make_group(3, [1.0, 0.0, 1.0]))  # mixed: no zero advantages

    # Buffer-level filtering removes zero-std groups before they can shrink a
    # training batch. Non-uniform groups stay available for later rollout-level filters.
    buffer.update(rollouts)
    assert len(buffer.env_buffers["env_a"].easy_examples) == 1
    assert len(buffer.env_buffers["env_a"].hard_examples) == 0
    # All-zero example stays in normal pool (not evicted to hard)
    assert 0 in buffer.env_buffers["env_a"].examples

    # Only non-uniform groups land in the rollout buffer.
    candidates = buffer.sample_rollouts(n=6)
    assert len(candidates) == 6

    # Compute advantages and apply ZeroAdvantageFilter
    compute_advantages(candidates, 3, DefaultAdvantageConfig())
    apply_filters([ZeroAdvantageFilter(name="zero_advantage", enforce=True)], candidates)

    # The middle rollout in [0.0, 0.5, 1.0] has exactly baseline reward and is
    # flagged by ZeroAdvantageFilter.
    zero_advantage_rollouts = [r for r in candidates if r["example_id"] == 2 and r["reward"] == 0.5]
    assert len(zero_advantage_rollouts) == 1
    assert zero_advantage_rollouts[0]["is_filtered"] is True

    # The other admitted rollouts have nonzero advantage.
    nonzero_advantage_rollouts = [r for r in candidates if r not in zero_advantage_rollouts]
    assert not any(r["is_filtered"] for r in nonzero_advantage_rollouts)

    # Example 0 remains in normal pool: zero-std filtering is a batch-admission
    # decision, not a hard-pool eviction.
    assert 0 in buffer.env_buffers["env_a"].examples
    assert len(buffer.env_buffers["env_a"].hard_examples) == 0


def test_buffer_no_filtering_by_default(dummy_envs, make_rollouts):
    """With online_difficulty_filtering=False (default), all rollouts are kept."""
    buffer = Buffer(dummy_envs, BufferConfig())
    buffer.update(make_rollouts(buffer, "env_a", list(range(5)), rewards=[1.0, 0.5, 0.0, 0.5, 0.5]))

    # All 5 problems -> 10 rollouts kept
    assert len(buffer.rollout_buffer) == 10


def test_buffer_save_load_with_conversion(dummy_envs, make_rollouts, tmp_path):
    """Easy/hard problems are partially converted to normal on load."""
    buffer = Buffer(dummy_envs, BufferConfig(easy_threshold=1.0, hard_threshold=0.0))
    buffer.update(make_rollouts(buffer, "env_a", list(range(5)), rewards=[1.0, 1.0, 0.5, 0.5, 0.0]))
    buffer.save(tmp_path / "buffer")

    new_buffer = Buffer(dummy_envs, BufferConfig(easy_fraction=0.5, hash_keys=["prompt", "env_name"]))
    new_buffer.load(tmp_path / "buffer")

    # 1 of 2 easy problems converted to normal
    assert len(new_buffer.env_buffers["env_a"].easy_examples) == 1
    # 2 were normal + 5 from env_b + 1 converted from easy = 8
    assert get_normal_count(new_buffer) == 8


def test_buffer_env_ratios(mock_openai_client, dummy_dataset):
    env_a = vf.SingleTurnEnv(client=mock_openai_client, model="test-model", dataset=dummy_dataset, rubric=vf.Rubric())
    env_b = vf.SingleTurnEnv(client=mock_openai_client, model="test-model", dataset=dummy_dataset, rubric=vf.Rubric())
    envs = make_envs(
        {
            "env_a": make_env("env_a", env_a, ratio=0.8),
            "env_b": make_env("env_b", env_b, ratio=0.2),
        }
    )

    buffer = Buffer(envs, BufferConfig())
    assert buffer.env_buffers["env_a"].num_normal == 5
    assert buffer.env_buffers["env_b"].num_normal == 5

    samples = buffer.sample_examples(100)
    env_a_count = sum(1 for p in samples if p["env_name"] == "env_a")
    assert 60 <= env_a_count <= 95


def test_buffer_env_ratios_validation():
    """Validates that env ratios must be positive and all-or-nothing."""
    from pydantic import ValidationError

    from prime_rl.configs.orchestrator import TrainConfig, TrainEnvConfig

    with pytest.raises(ValidationError):
        EnvConfig(id="env_a", ratio=-0.3)

    with pytest.raises(ValidationError, match="mix of set and unset"):
        TrainConfig(env=[TrainEnvConfig(id="a", ratio=0.5), TrainEnvConfig(id="b")])


def test_buffer_no_cross_env_pool_assignment(mock_openai_client, tmp_path):
    """Pool assignments don't transfer if example_id exists but env changed."""
    original_dataset = Dataset.from_dict({"question": ["q0"], "answer": ["a0"]})
    original_env = vf.SingleTurnEnv(
        client=mock_openai_client,
        model="test-model",
        dataset=original_dataset,
        rubric=vf.Rubric(),
    )
    original_env_set = make_envs({"env_a": make_env("env_a", original_env)})

    buffer = Buffer(original_env_set, BufferConfig(easy_threshold=1.0))
    eb = buffer.env_buffers["env_a"]
    example_id = list(eb.examples.keys())[0]
    example = eb.examples.pop(example_id)
    eb.easy_examples.append(example)
    buffer.save(tmp_path / "buffer")

    new_dataset = Dataset.from_dict({"question": ["different_q"], "answer": ["different_a"]})
    new_env = vf.SingleTurnEnv(
        client=mock_openai_client,
        model="test-model",
        dataset=new_dataset,
        rubric=vf.Rubric(),
    )
    new_env_set = make_envs({"env_b": make_env("env_b", new_env)})

    new_buffer = Buffer(new_env_set, BufferConfig())
    new_buffer.load(tmp_path / "buffer")

    assert len(new_buffer.env_buffers["env_b"].easy_examples) == 0
    assert new_buffer.env_buffers["env_b"].num_normal == 1
