import pytest
import verifiers.v1 as vf

from prime_rl.configs.orchestrator import (
    CustomAdvantageConfig,
    DefaultAdvantageConfig,
    LinearLengthPenaltyConfig,
    OrchestratorConfig,
)
from prime_rl.orchestrator.advantage import (
    AdvantageInputs,
    AdvantageOutputs,
    assign_advantages,
    default_advantage_fn,
    setup_advantage_fn,
)
from prime_rl.orchestrator.envs import Env, TrainEnv
from prime_rl.orchestrator.types import Rollout


def _make_rollout(reward: float, completion_len: int = 0, env_name: str = "test") -> Rollout:
    """Build a ``Rollout`` (message-graph trace): ``reward`` via the reward dict and
    ``completion_len`` model-sampled completion tokens (a single assistant node)."""
    node = vf.MessageNode(
        message=vf.AssistantMessage(content="x"),
        token_ids=list(range(completion_len)),
        mask=[True] * completion_len,
        logprobs=[0.0] * completion_len,
        sampled=True,
    )
    rollout = Rollout[vf.Task](task=vf.Task(idx=0, prompt=""), nodes=[node], rewards={"reward": reward})
    rollout.env_name = env_name
    return rollout


def _make_group(rewards, completion_lengths=None) -> AdvantageInputs:
    """Build single-group AdvantageInputs from 1D arrays of rewards/lengths."""
    rollouts = []
    for i, reward in enumerate(rewards):
        cl = int(completion_lengths[i]) if completion_lengths is not None else 0
        rollouts.append(_make_rollout(float(reward), cl))
    return AdvantageInputs(rollouts=rollouts)


def _train_rollouts(rewards: list[float]) -> list[Rollout]:
    """One group's worth of Rollouts for ``assign_advantages`` (operates on a single group)."""
    return [Rollout[vf.Task](task=vf.Task(idx=0, prompt=""), rewards={"reward": r}) for r in rewards]


def test_default_advantage_fn_simple_mean():
    inputs = _make_group(rewards=[1.0, 0.5, 0.8], completion_lengths=[10, 12, 8])
    result = default_advantage_fn(inputs)

    assert len(result.advantages) == 3
    assert sum(result.advantages) == pytest.approx(0.0, abs=1e-6)


def test_linear_length_penalty_scales_by_pass_rate():
    """Linear penalty subtracts coef * pass_rate * (completion tokens / group max length), then mean-centers.

    ``pass_rate`` is the group's mean reward, so a half-solved group scales the penalty by 0.5.
    The denominator is the group's longest sequence, so the longest rollout takes the full coef.
    """
    rewards = [1.0, 1.0, 0.0, 0.0]
    lengths = [10, 20, 30, 40]
    inputs = _make_group(rewards=rewards, completion_lengths=lengths)
    result = default_advantage_fn(inputs, length_penalty=LinearLengthPenaltyConfig(coef=2.0))

    pass_rate = sum(rewards) / len(rewards)  # 0.5
    denom = max(lengths)  # 40
    penalized = [r - 2.0 * pass_rate * (length / denom) for r, length in zip(rewards, lengths)]
    mean = sum(penalized) / len(penalized)
    assert result.advantages == pytest.approx([p - mean for p in penalized], abs=1e-6)

    # Zero-mean per group, and within each reward tier shorter output → higher advantage
    assert sum(result.advantages) == pytest.approx(0.0, abs=1e-6)
    assert result.advantages[0] > result.advantages[1]
    assert result.advantages[2] > result.advantages[3]


def test_linear_length_penalty_zero_pass_rate_disables_penalty():
    """A never-solved group (mean reward 0) gets no length pressure — falls back to plain GRPO."""
    inputs = _make_group(rewards=[0.0, 0.0, 0.0], completion_lengths=[10, 50, 200])
    penalized = default_advantage_fn(inputs, length_penalty=LinearLengthPenaltyConfig(coef=5.0))
    plain = default_advantage_fn(inputs)
    assert penalized.advantages == pytest.approx(plain.advantages, abs=1e-6)


def test_linear_length_penalty_gate_by_correctness():
    """Gating scales each rollout's penalty by its reward, so reward-0 rollouts are untouched."""
    rewards = [1.0, 1.0, 0.0, 0.0]
    lengths = [10, 20, 30, 40]
    inputs = _make_group(rewards=rewards, completion_lengths=lengths)
    cfg = LinearLengthPenaltyConfig(coef=2.0, gate_by_correctness=True)
    result = default_advantage_fn(inputs, length_penalty=cfg)

    pass_rate = sum(rewards) / len(rewards)  # 0.5
    denom = max(lengths)  # 40
    penalized = [r - r * 2.0 * pass_rate * (length / denom) for r, length in zip(rewards, lengths)]
    mean = sum(penalized) / len(penalized)
    assert result.advantages == pytest.approx([p - mean for p in penalized], abs=1e-6)


def test_setup_advantage_fn_builds_linear_penalty():
    """``setup_advantage_fn`` builds a linear penalty normalized by the group's max length."""
    rewards = [1.0, 1.0]
    lengths = [10, 30]
    inputs = _make_group(rewards=rewards, completion_lengths=lengths)
    fn = setup_advantage_fn(DefaultAdvantageConfig(length_penalty=LinearLengthPenaltyConfig(coef=1.0)))
    result = fn(inputs)

    pass_rate = sum(rewards) / len(rewards)  # 1.0
    denom = max(lengths)  # 30
    penalized = [1.0 - 1.0 * pass_rate * (length / denom) for length in lengths]
    mean = sum(penalized) / len(penalized)
    assert result.advantages == pytest.approx([p - mean for p in penalized], abs=1e-6)


def test_train_env_builds_advantage_fn(monkeypatch):
    """TrainEnv-built advantage funcs normalize the length penalty by the group's max length."""

    def fake_env_init(self, config):
        self.config = config

    monkeypatch.setattr(Env, "__init__", fake_env_init)
    config = OrchestratorConfig(
        train={
            "env": [
                {
                    "taskset": {"id": "reverse-text-v1"},
                    "advantage": {"type": "default", "length_penalty": {"coef": 1.0}},
                }
            ]
        },
    )

    env = TrainEnv(config.train.env[0])
    assert env.advantage_fn is not None
    result = env.advantage_fn(_make_group(rewards=[1.0, 1.0], completion_lengths=[10, 30]))

    # pass_rate 1.0, denom max(10,30)=30: penalized [1-1/3, 0] -> centered [1/3, -1/3]
    assert result.advantages == pytest.approx([1 / 3, -1 / 3], abs=1e-6)


def test_per_env_linear_advantage_uses_runtime_schema():
    config = OrchestratorConfig(
        train={
            "env": [
                {
                    "taskset": {"id": "reverse-text-v1"},
                    "advantage": {"type": "default", "length_penalty": {"coef": 1.0}},
                }
            ]
        },
    )

    advantage = config.train.env[0].advantage
    assert isinstance(advantage, DefaultAdvantageConfig)
    assert isinstance(advantage.length_penalty, LinearLengthPenaltyConfig)

    fn = setup_advantage_fn(advantage)
    result = fn(_make_group(rewards=[1.0, 1.0], completion_lengths=[10, 30]))

    assert result.advantages == pytest.approx([1 / 3, -1 / 3], abs=1e-6)


def test_per_env_custom_advantage_uses_runtime_schema():
    config = OrchestratorConfig(
        train={
            "env": [
                {
                    "taskset": {"id": "reverse-text-v1"},
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


def test_assign_advantages_writes_field():
    rollouts = _train_rollouts([1.0, 0.5, 0.8])
    fn = setup_advantage_fn(DefaultAdvantageConfig())
    assign_advantages(rollouts, fn)
    advs = [r.advantage for r in rollouts]
    assert sum(advs) == pytest.approx(0.0, abs=1e-6)


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
    return AdvantageOutputs(advantages=[r.reward * scale for r in inputs.rollouts])
