import math
import uuid

import pytest
import verifiers.v1 as vf

from prime_rl.configs.orchestrator import (
    CustomAdvantageConfig,
    DefaultAdvantageConfig,
    TokensLengthPenaltyConfig,
    TurnsLengthPenaltyConfig,
)
from prime_rl.orchestrator.advantage import (
    AdvantageInputs,
    AdvantageOutputs,
    assign_advantages,
    default_advantage_fn,
    setup_advantage_fn,
)
from prime_rl.orchestrator.types import Rollout


def _make_rollout(
    reward: float,
    completion_len: int = 0,
    num_turns: int = 1,
    env_name: str = "test",
    tool_response_len: int = 0,
) -> Rollout:
    """Build a ``Rollout`` (message-graph trace) for advantage testing: ``reward`` via the
    reward dict, ``completion_len`` sampled tokens split across ``num_turns`` assistant nodes,
    and an optional tool-response token count surfaced as a metric."""
    per_turn, rem = divmod(completion_len, max(num_turns, 1))
    nodes = [
        vf.MessageNode(
            message=vf.AssistantMessage(content="x"),
            token_ids=list(range(n := per_turn + (rem if i == 0 else 0))),
            mask=[True] * n,
            logprobs=[0.0] * n,
            sampled=True,
        )
        for i in range(num_turns)
    ]
    metrics = {"rlm_total_tool_response_tokens": tool_response_len} if tool_response_len else {}
    rollout = Rollout[vf.Task](task=vf.Task(idx=0, prompt=""), nodes=nodes, rewards={"reward": reward}, metrics=metrics)
    rollout.env_name = env_name
    return rollout


def _make_group(rewards, completion_lengths=None, num_turns=None) -> AdvantageInputs:
    """Build single-group AdvantageInputs from 1D arrays of rewards/lengths/turns."""
    rollouts = []
    for i, reward in enumerate(rewards):
        cl = int(completion_lengths[i]) if completion_lengths is not None else 0
        nt = int(num_turns[i]) if num_turns is not None else 1
        rollouts.append(_make_rollout(float(reward), cl, nt))
    return AdvantageInputs(rollouts=rollouts)


# Helper aliases for readability — completion-only and tool-only token shaping.
_TOKENS_COMPLETION = TokensLengthPenaltyConfig(completion_weight=1.0, tool_response_weight=0.0)
_TOKENS_TOOL_ONLY = TokensLengthPenaltyConfig(completion_weight=0.0, tool_response_weight=1.0)


def test_default_advantage_fn_simple_mean():
    inputs = _make_group(rewards=[1.0, 0.5, 0.8], completion_lengths=[10, 12, 8])
    result = default_advantage_fn(inputs)

    assert len(result.advantages) == 3
    assert sum(result.advantages) == pytest.approx(0.0, abs=1e-6)


def test_efficiency_mixed_group():
    """Mixed group: reward shaping preserves zero-mean, shorter correct gets higher advantage."""
    inputs = _make_group(rewards=[1.0, 1.0, 0.0, 1.0], completion_lengths=[10, 30, 20, 20])
    result = default_advantage_fn(inputs, length_penalty=_TOKENS_COMPLETION)

    # mean_correct_len = (10+30+20)/3 = 20
    # bonus = clamp(1 - [10,30,20,20]/20, 0, 1) = [0.5, 0, 0, 0]
    # shaped_rewards = R * (1 + bonus * correct_mask) = [1.5, 1, 0, 1]
    # baseline = mean(shaped_rewards) = 0.875
    # A = shaped_rewards - baseline = [0.625, 0.125, -0.875, 0.125]
    assert result.advantages == pytest.approx([0.625, 0.125, -0.875, 0.125], abs=1e-6)

    # Zero-mean per group
    assert sum(result.advantages) == pytest.approx(0.0, abs=1e-6)

    # All correct rollouts have positive advantage
    for rollout, adv in zip(inputs.rollouts, result.advantages):
        if rollout.reward >= 1.0:
            assert adv > 0


def test_efficiency_all_correct_group():
    """All-correct group: zero-mean, shorter gets higher advantage."""
    inputs = _make_group(rewards=[1.0, 1.0, 1.0], completion_lengths=[10, 20, 40])
    result = default_advantage_fn(inputs, length_penalty=_TOKENS_COMPLETION)

    # mean_len = 70/3 ≈ 23.33
    # bonus = clamp(1 - [10, 20, 40] / (70/3), 0, 1) = [4/7, 1/7, 0]
    # shaped_rewards = [1+4/7, 1+1/7, 1] = [11/7, 8/7, 1]
    shaped = [11.0 / 7, 8.0 / 7, 1.0]
    mean_shaped = sum(shaped) / len(shaped)
    expected = [s - mean_shaped for s in shaped]
    assert result.advantages == pytest.approx(expected, abs=1e-6)

    # Zero-mean
    assert sum(result.advantages) == pytest.approx(0.0, abs=1e-6)

    # Shortest has highest advantage
    assert result.advantages[0] > result.advantages[1] > result.advantages[2]


def test_efficiency_all_zero_rewards():
    """When all rewards are 0, no length shaping — falls back to standard GRPO."""
    inputs = _make_group(rewards=[0.0, 0.0, 0.0], completion_lengths=[10, 20, 15])
    result_with = default_advantage_fn(inputs, length_penalty=_TOKENS_COMPLETION)
    result_without = default_advantage_fn(inputs)

    assert result_with.advantages == pytest.approx(result_without.advantages, abs=1e-6)


def test_efficiency_single_correct():
    """Single correct rollout: bonus=0 (at its own mean), same as standard GRPO."""
    inputs = _make_group(rewards=[1.0, 0.0, 0.0, 0.0], completion_lengths=[100, 50, 200, 150])
    result = default_advantage_fn(inputs, length_penalty=_TOKENS_COMPLETION)

    assert result.advantages == pytest.approx([0.75, -0.25, -0.25, -0.25], abs=1e-6)


def test_efficiency_shorter_correct_higher_advantage():
    """Among correct rollouts in a mixed group, shorter always gets higher advantage."""
    inputs = _make_group(rewards=[1.0, 1.0, 1.0, 0.0, 0.0], completion_lengths=[50, 100, 200, 80, 120])
    result = default_advantage_fn(inputs, length_penalty=_TOKENS_COMPLETION)

    advs = result.advantages
    assert advs[0] > advs[1] > advs[2]
    assert all(a > 0 for a in advs[:3])
    assert all(a < 0 for a in advs[3:])


def test_efficiency_zero_mean_per_group():
    """Reward shaping preserves zero-mean advantages within each group."""
    mixed = default_advantage_fn(
        _make_group(rewards=[1.0, 1.0, 0.0, 1.0], completion_lengths=[10, 30, 20, 20]),
        length_penalty=_TOKENS_COMPLETION,
    )
    all_correct = default_advantage_fn(
        _make_group(rewards=[1.0, 1.0, 1.0, 1.0], completion_lengths=[10, 20, 40, 80]),
        length_penalty=_TOKENS_COMPLETION,
    )

    assert sum(mixed.advantages) == pytest.approx(0.0, abs=1e-6)
    assert sum(all_correct.advantages) == pytest.approx(0.0, abs=1e-6)


def test_efficiency_amplification_bounded():
    """Even with extreme length outliers, reward amplification is capped at 2x."""
    inputs = _make_group(rewards=[1.0, 1.0, 0.0], completion_lengths=[1, 10000, 5000])
    result = default_advantage_fn(inputs, length_penalty=_TOKENS_COMPLETION)

    # Shortest correct gets bonus ≈ 1, so shaped_reward ≈ 2
    # Standard reward = 1, so amplification ≈ 2x
    # shaped_rewards ≈ [2, 1, 0], baseline ≈ 1, max advantage ≈ 1
    assert result.advantages[0] < 1.0 + 1e-3


def test_efficiency_tokens_with_tool_response_weight():
    """`tool_response_weight` shifts shaping onto tool-response tokens read from rollout metrics."""
    rollouts = [
        _make_rollout(1.0, completion_len=10, tool_response_len=200),
        _make_rollout(1.0, completion_len=10, tool_response_len=0),
        _make_rollout(1.0, completion_len=10, tool_response_len=100),
    ]
    inputs = AdvantageInputs(rollouts=rollouts)

    # completion tokens identical (10 each) → completion-only shaping is a no-op
    result_completion_only = default_advantage_fn(inputs, length_penalty=_TOKENS_COMPLETION)
    assert result_completion_only.advantages == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)

    # tool-response only: costs are [200, 0, 100], mean=100, bonus is one-sided
    # so only the below-mean rollout (idx 1) gets amplified; the at/above-mean tie.
    result_tool_only = default_advantage_fn(inputs, length_penalty=_TOKENS_TOOL_ONLY)
    advs = result_tool_only.advantages
    assert advs[1] > advs[0]
    assert advs[1] > advs[2]
    assert advs[0] == pytest.approx(advs[2], abs=1e-6)
    assert sum(advs) == pytest.approx(0.0, abs=1e-6)


def test_efficiency_fractional_weight_with_int_rewards():
    """Fractional weights must not truncate when rollout rewards are emitted as ints."""
    lens = [7, 11, 13]
    int_rewards = [1, 1, 0]
    rollouts_int = [_make_rollout(r, completion_len=n) for r, n in zip(int_rewards, lens)]
    rollouts_float = [_make_rollout(float(r), completion_len=n) for r, n in zip(int_rewards, lens)]

    fractional = TokensLengthPenaltyConfig(completion_weight=0.3, tool_response_weight=0.0)
    int_result = default_advantage_fn(AdvantageInputs(rollouts=rollouts_int), length_penalty=fractional)
    float_result = default_advantage_fn(AdvantageInputs(rollouts=rollouts_float), length_penalty=fractional)
    assert int_result.advantages == pytest.approx(float_result.advantages, abs=1e-6)


def test_efficiency_zero_costs_falls_back_to_plain_grpo():
    """When all effective costs are zero, shaping is a no-op (no NaNs from div-by-zero)."""
    # tool-only weights but no harness metric → all costs == 0
    inputs = _make_group(rewards=[1.0, 1.0, 0.0], completion_lengths=[10, 10, 10])
    result = default_advantage_fn(inputs, length_penalty=_TOKENS_TOOL_ONLY)
    expected = default_advantage_fn(inputs)  # plain GRPO
    assert not any(math.isnan(a) for a in result.advantages)
    assert result.advantages == pytest.approx(expected.advantages, abs=1e-6)


def test_efficiency_tokens_default_weights_match_completion_when_no_metric():
    """Default TokensLengthPenaltyConfig (1,1) reduces to completion-only when rollouts lack the metric."""
    inputs = _make_group(rewards=[1.0, 1.0, 0.0, 1.0], completion_lengths=[10, 30, 20, 20])
    result_default = default_advantage_fn(inputs, length_penalty=TokensLengthPenaltyConfig())
    result_completion = default_advantage_fn(inputs, length_penalty=_TOKENS_COMPLETION)
    assert result_default.advantages == pytest.approx(result_completion.advantages, abs=1e-6)


def test_efficiency_turns_penalty():
    """`TurnsLengthPenaltyConfig` shapes by trajectory turn count rather than token count."""
    inputs = _make_group(
        rewards=[1.0, 1.0, 0.0, 1.0],
        # token counts identical, but turns differ — turns penalty should still differentiate
        completion_lengths=[100, 100, 100, 100],
        num_turns=[1, 3, 2, 2],
    )
    result = default_advantage_fn(inputs, length_penalty=TurnsLengthPenaltyConfig())

    # mean_correct_turns = (1+3+2)/3 = 2
    # bonus = clamp(1 - [1,3,2,2]/2, 0, 1) = [0.5, 0, 0, 0]
    assert result.advantages == pytest.approx([0.625, 0.125, -0.875, 0.125], abs=1e-6)


def _train_rollouts(rewards: list[float]) -> list[Rollout]:
    """Build ``Rollout``\\ s sharing a single ``group_id`` — ``assign_advantages`` works on one
    group at a time (the sink groups by ``group_id`` upstream)."""
    gid = uuid.uuid4()
    rollouts = []
    for r in rewards:
        rollout = Rollout[vf.Task](task=vf.Task(idx=0, prompt=""), rewards={"reward": r})
        rollout.env_name = "test"
        rollout.group_id = gid
        rollouts.append(rollout)
    return rollouts


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
