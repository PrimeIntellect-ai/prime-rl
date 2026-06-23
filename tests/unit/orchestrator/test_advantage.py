import uuid

import pytest

from prime_rl.configs.orchestrator import (
    CustomAdvantageConfig,
    DefaultAdvantageConfig,
    LinearLengthPenaltyConfig,
    OrchestratorConfig,
    RAEAdvantageConfig,
)
from prime_rl.orchestrator.advantage import (
    AdvantageInputs,
    AdvantageOutputs,
    assign_advantages,
    centered_linear_length_penalty,
    default_advantage_fn,
    maxrl_advantage_fn,
    reward_advantage_fn,
    setup_advantage_fn,
)
from prime_rl.orchestrator.envs import Env, TrainEnv
from prime_rl.orchestrator.types import TrainRollout


def _make_rollout(reward: float, completion_len: int = 0) -> dict:
    """Minimal single-turn rollout dict with ``completion_len`` completion tokens."""
    return {
        "reward": reward,
        "trajectory": [{"tokens": {"prompt_ids": [0], "completion_ids": list(range(completion_len))}}],
        "env_name": "test",
        "example_id": 0,
    }


def _make_group(rewards, completion_lengths=None) -> AdvantageInputs:
    """Build single-group AdvantageInputs from 1D arrays of rewards/lengths."""
    rollouts = []
    for i, reward in enumerate(rewards):
        cl = int(completion_lengths[i]) if completion_lengths is not None else 0
        rollouts.append(_make_rollout(float(reward), cl))
    return AdvantageInputs(rollouts=rollouts)


def test_default_advantage_fn_simple_mean():
    inputs = _make_group(rewards=[1.0, 0.5, 0.8], completion_lengths=[10, 12, 8])
    result = default_advantage_fn(inputs)

    assert len(result.advantages) == 3
    assert sum(result.advantages) == pytest.approx(0.0, abs=1e-6)


def test_maxrl_advantage_normalizes_by_mean_reward():
    inputs = _make_group(rewards=[1.0, 0.0, 0.0, 0.0], completion_lengths=[10, 12, 8, 9])
    result = maxrl_advantage_fn(inputs)

    assert result.advantages == pytest.approx([3.0, -1.0, -1.0, -1.0], abs=1e-6)
    assert sum(result.advantages) == pytest.approx(0.0, abs=1e-6)


def test_maxrl_advantage_drops_all_zero_groups():
    inputs = _make_group(rewards=[0.0, 0.0, 0.0], completion_lengths=[10, 12, 8])
    result = maxrl_advantage_fn(inputs)

    assert result.advantages == [0.0, 0.0, 0.0]


def test_reward_advantage_returns_raw_rewards():
    inputs = _make_group(rewards=[1.0, 0.0, 0.5], completion_lengths=[10, 12, 8])
    result = reward_advantage_fn(inputs)

    assert result.advantages == [1.0, 0.0, 0.5]


def test_linear_length_penalty_scales_by_pass_rate():
    """Linear penalty subtracts coef * pass_rate * (completion tokens / max_seq_len), then mean-centers.

    ``pass_rate`` is the group's mean reward, so a half-solved group scales the penalty by 0.5.
    """
    rewards = [1.0, 1.0, 0.0, 0.0]
    inputs = _make_group(rewards=rewards, completion_lengths=[10, 20, 30, 40])
    result = default_advantage_fn(inputs, length_penalty=LinearLengthPenaltyConfig(coef=2.0), max_seq_len=100)

    pass_rate = sum(rewards) / len(rewards)  # 0.5
    penalized = [r - 2.0 * pass_rate * (length / 100) for r, length in zip(rewards, (10, 20, 30, 40))]
    mean = sum(penalized) / len(penalized)
    assert result.advantages == pytest.approx([p - mean for p in penalized], abs=1e-6)

    # Zero-mean per group, and within each reward tier shorter output → higher advantage
    assert sum(result.advantages) == pytest.approx(0.0, abs=1e-6)
    assert result.advantages[0] > result.advantages[1]
    assert result.advantages[2] > result.advantages[3]


def test_centered_linear_length_penalty_caps_truncated_rows():
    result = centered_linear_length_penalty(
        lengths=[10, 50],
        max_seq_len=100,
        coef=2.0,
        scale=0.5,
        weights=[1.0, 1.0],
        truncated=[False, True],
    )

    assert result.penalties == pytest.approx([0.1, 1.0], abs=1e-6)
    assert result.aux == pytest.approx([0.45, -0.45], abs=1e-6)
    assert result.costs == pytest.approx([0.1, 1.0], abs=1e-6)


def test_linear_length_penalty_zero_pass_rate_disables_penalty():
    """A never-solved group (mean reward 0) gets no length pressure — falls back to plain GRPO."""
    inputs = _make_group(rewards=[0.0, 0.0, 0.0], completion_lengths=[10, 50, 200])
    penalized = default_advantage_fn(inputs, length_penalty=LinearLengthPenaltyConfig(coef=5.0), max_seq_len=100)
    plain = default_advantage_fn(inputs)
    assert penalized.advantages == pytest.approx(plain.advantages, abs=1e-6)


def test_linear_length_penalty_gate_by_correctness():
    """Gating scales each rollout's penalty by its reward, so reward-0 rollouts are untouched."""
    rewards = [1.0, 1.0, 0.0, 0.0]
    inputs = _make_group(rewards=rewards, completion_lengths=[10, 20, 30, 40])
    cfg = LinearLengthPenaltyConfig(coef=2.0, gate_by_correctness=True)
    result = default_advantage_fn(inputs, length_penalty=cfg, max_seq_len=100)

    pass_rate = sum(rewards) / len(rewards)  # 0.5
    penalized = [r - r * 2.0 * pass_rate * (length / 100) for r, length in zip(rewards, (10, 20, 30, 40))]
    mean = sum(penalized) / len(penalized)
    assert result.advantages == pytest.approx([p - mean for p in penalized], abs=1e-6)


def test_setup_advantage_fn_threads_max_seq_len():
    """``setup_advantage_fn`` threads ``max_seq_len`` into the linear penalty denominator."""
    rewards = [1.0, 1.0]
    inputs = _make_group(rewards=rewards, completion_lengths=[10, 30])
    fn = setup_advantage_fn(DefaultAdvantageConfig(length_penalty=LinearLengthPenaltyConfig(coef=1.0)), max_seq_len=100)
    result = fn(inputs)

    pass_rate = sum(rewards) / len(rewards)  # 1.0
    penalized = [1.0 - 1.0 * pass_rate * (length / 100) for length in (10, 30)]
    mean = sum(penalized) / len(penalized)
    assert result.advantages == pytest.approx([p - mean for p in penalized], abs=1e-6)


def test_train_env_threads_max_seq_len_into_advantage_fn(monkeypatch):
    """TrainEnv-built advantage funcs use orchestrator seq_len for length penalties."""

    def fake_env_init(self, config):
        self.config = config
        self._env = None
        self._env_client = None
        self._env_server_process = None

    monkeypatch.setattr(Env, "__init__", fake_env_init)
    config = OrchestratorConfig(
        seq_len=100,
        advantage={"type": "default", "length_penalty": {"coef": 1.0}},
    )

    env = TrainEnv(config.train.env[0], max_seq_len=config.seq_len)
    assert env.advantage_fn is not None
    result = env.advantage_fn(_make_group(rewards=[1.0, 1.0], completion_lengths=[10, 30]))

    assert result.advantages == pytest.approx([0.1, -0.1], abs=1e-6)


def test_per_env_linear_advantage_uses_runtime_schema():
    config = OrchestratorConfig(
        seq_len=100,
        train={
            "env": [
                {
                    "advantage": {
                        "type": "default",
                        "length_penalty": {"coef": 1.0},
                    },
                }
            ],
        },
    )

    advantage = config.train.env[0].advantage
    assert isinstance(advantage, DefaultAdvantageConfig)
    assert isinstance(advantage.length_penalty, LinearLengthPenaltyConfig)

    fn = setup_advantage_fn(advantage, max_seq_len=config.seq_len)
    result = fn(_make_group(rewards=[1.0, 1.0], completion_lengths=[10, 30]))

    assert result.advantages == pytest.approx([0.1, -0.1], abs=1e-6)


def test_rae_advantage_uses_same_length_penalty_schema():
    config = OrchestratorConfig(
        seq_len=100,
        advantage={"type": "rae", "length_penalty": {"coef": 1.0}},
    )

    advantage = config.train.env[0].advantage
    assert isinstance(advantage, RAEAdvantageConfig)
    assert isinstance(advantage.length_penalty, LinearLengthPenaltyConfig)
    assert advantage.length_penalty.coef == 1.0


def test_rae_length_penalty_rejects_single_agent_correctness_gate():
    with pytest.raises(ValueError, match="frontier-gated"):
        RAEAdvantageConfig(length_penalty={"gate_by_correctness": True})


def test_per_env_custom_advantage_uses_runtime_schema():
    config = OrchestratorConfig(
        train={
            "env": [
                {
                    "advantage": {
                        "type": "custom",
                        "import_path": "tests.unit.orchestrator.test_advantage._dummy_custom_advantage",
                        "kwargs": {"scale": 3.0},
                    },
                }
            ],
        },
    )

    advantage = config.train.env[0].advantage
    assert isinstance(advantage, CustomAdvantageConfig)

    fn = setup_advantage_fn(advantage)
    result = fn(_make_group(rewards=[1.0, 0.5], completion_lengths=[10, 20]))

    assert result.advantages == pytest.approx([3.0, 1.5], abs=1e-6)


def test_length_weighted_baseline():
    """Length-weighted baseline uses sum(len_i * reward_i) / sum(len_i) instead of the plain mean."""
    rewards = [1.0, 0.0, 0.0]
    inputs = _make_group(rewards=rewards, completion_lengths=[10, 30, 60])
    result = default_advantage_fn(inputs, length_weighted_baseline=True)

    baseline = (10 * 1.0 + 30 * 0.0 + 60 * 0.0) / (10 + 30 + 60)  # 0.1
    assert result.advantages == pytest.approx([r - baseline for r in rewards], abs=1e-6)
    # Token-weighted mean of advantages is zero
    assert sum(length * adv for length, adv in zip((10, 30, 60), result.advantages)) == pytest.approx(0.0, abs=1e-5)


def _train_rollouts(rewards: list[float]) -> list[TrainRollout]:
    """Wrap a list of rewards into ``TrainRollout``\\ s sharing a single
    ``group_id`` — ``assign_advantages`` works on one group at a time
    (the sink groups by ``group_id`` upstream)."""
    gid = uuid.uuid4()
    return [
        TrainRollout(
            raw={"reward": r, "trajectory": []},
            env_name="test",
            example_id=0,
            group_id=gid,
            policy_version=0,
            off_policy_steps=0,
        )
        for r in rewards
    ]


def test_assign_advantages_writes_field():
    rollouts = _train_rollouts([1.0, 0.5, 0.8])
    fn = setup_advantage_fn(DefaultAdvantageConfig())
    assign_advantages(rollouts, fn)
    advs = [r.advantage for r in rollouts]
    assert sum(advs) == pytest.approx(0.0, abs=1e-6)


def test_assign_advantages_writes_length_penalty_annotations():
    rollouts = _train_rollouts([1.0, 1.0])
    rollouts[0].raw["trajectory"] = [{"tokens": {"prompt_ids": [0], "completion_ids": list(range(10))}}]
    rollouts[1].raw["trajectory"] = [{"tokens": {"prompt_ids": [0], "completion_ids": list(range(30))}}]
    fn = setup_advantage_fn(DefaultAdvantageConfig(length_penalty=LinearLengthPenaltyConfig(coef=1.0)), max_seq_len=100)

    assign_advantages(rollouts, fn)

    assert rollouts[0].advantage == pytest.approx(0.1, abs=1e-6)
    assert rollouts[1].advantage == pytest.approx(-0.1, abs=1e-6)
    assert rollouts[0].raw["length_penalty"] == {
        "eligible": True,
        "penalty": pytest.approx(0.1, abs=1e-6),
        "aux": pytest.approx(0.1, abs=1e-6),
        "cost": pytest.approx(0.1, abs=1e-6),
        "sign_flipped": False,
    }
    assert rollouts[1].raw["length_penalty"] == {
        "eligible": True,
        "penalty": pytest.approx(0.3, abs=1e-6),
        "aux": pytest.approx(-0.1, abs=1e-6),
        "cost": pytest.approx(0.3, abs=1e-6),
        "sign_flipped": False,
    }


def test_assign_advantages_without_fn_is_reward():
    """``advantage_fn=None`` falls back to ``advantage = reward``."""
    rollouts = _train_rollouts([1.0, 0.5, 0.8])
    assign_advantages(rollouts, None)
    assert [r.advantage for r in rollouts] == [1.0, 0.5, 0.8]


def test_assign_advantages_singleton_group_is_zero():
    """A group of size 1 has reward == mean, so its advantage is 0."""
    rollouts = _train_rollouts([0.7])
    fn = setup_advantage_fn(DefaultAdvantageConfig())
    assign_advantages(rollouts, fn)
    assert rollouts[0].advantage == pytest.approx(0.0, abs=1e-6)


def test_setup_advantage_fn_with_custom_config():
    config = CustomAdvantageConfig(
        import_path="tests.unit.orchestrator.test_advantage._dummy_custom_advantage",
        kwargs={"scale": 2.0},
    )
    advantage_fn = setup_advantage_fn(config)

    inputs = _make_group(rewards=[1.0, 0.5, 0.8], completion_lengths=[10, 12, 8])

    result = advantage_fn(inputs)
    assert isinstance(result, AdvantageOutputs)
    assert result.advantages == pytest.approx([2.0, 1.0, 1.6], abs=1e-6)


def _dummy_custom_advantage(inputs: AdvantageInputs, scale: float = 1.0) -> AdvantageOutputs:
    """A simple custom advantage for testing."""
    return AdvantageOutputs(advantages=[r["reward"] * scale for r in inputs.rollouts])
