import math

import pytest
import verifiers.v1 as vf

from prime_rl.configs.algorithm import (
    CustomAlgorithmConfig,
    TokensLengthPenaltyConfig,
    TurnsLengthPenaltyConfig,
)
from prime_rl.orchestrator.algo import CustomAlgorithm
from prime_rl.orchestrator.algo.advantage import (
    apply_advantage_fn,
    default_advantage_fn,
    max_rl_advantage_fn,
)
from prime_rl.orchestrator.types import Rollout, RolloutView
from prime_rl.transport.types import TrainingSample


def _task(idx: int = 0) -> vf.Task:
    return vf.Task(idx=idx, prompt="")


def _make_rollout(
    reward: float,
    completion_len: int = 0,
    tool_response_len: int = 0,
    num_turns: int = 1,
    env_name: str = "test",
    example_id: int = 0,
) -> Rollout:
    """Create a minimal trace rollout for advantage testing.

    `completion_len` tokens are split across sampled assistant nodes. Tool
    response tokens are non-sampled tool nodes, matching the trace-level cost
    read by `RolloutView`.
    """
    per_turn, rem = divmod(completion_len, max(num_turns, 1))
    nodes: list[vf.MessageNode] = []
    parent: int | None = None
    next_token = 0
    for i in range(num_turns):
        n = per_turn + (rem if i == 0 else 0)
        nodes.append(
            vf.MessageNode(
                parent=parent,
                message=vf.AssistantMessage(content="x"),
                sampled=True,
                token_ids=list(range(next_token, next_token + n)),
                mask=[True] * n,
                logprobs=[0.0] * n,
            )
        )
        parent = len(nodes) - 1
        next_token += n
    if tool_response_len:
        nodes.append(
            vf.MessageNode(
                parent=parent,
                message=vf.ToolMessage(tool_call_id="call", content="tool"),
                token_ids=list(range(next_token, next_token + tool_response_len)),
                mask=[False] * tool_response_len,
                logprobs=[],
            )
        )
    rollout = Rollout[vf.Task](task=_task(example_id), nodes=nodes, rewards={"reward": reward})
    rollout.env_name = env_name
    return rollout


def _train_rollout(reward: float = 0.0, completion_ids: tuple[int, ...] = (2,)) -> Rollout:
    """One rollout carrying a single training sample with the given completion
    tokens. The trace itself can be empty; this helper exercises the transport
    stream written through `RolloutView.assign_advantages`.
    """
    rollout = Rollout[vf.Task](task=_task(), rewards={"reward": reward})
    rollout.env_name = "test"
    rollout.samples = [
        TrainingSample(
            prompt_ids=[1],
            prompt_mask=[False],
            completion_ids=list(completion_ids),
            completion_mask=[True] * len(completion_ids),
            completion_logprobs=[-0.1] * len(completion_ids),
            completion_temperatures=[],
            env_name="test",
        )
    ]
    return rollout


def _views(rollouts: list[Rollout]) -> list[RolloutView]:
    """Wrap raw rollout dicts into ``RolloutView``\\ s over single-token
    samples — the advantage fns see exactly what ``score_group`` sees."""
    return [RolloutView(rollout) for rollout in rollouts]


def _make_group(rewards, completion_lengths=None, num_turns=None) -> list[RolloutView]:
    """Build one group of ``RolloutView``\\ s from 1D arrays of rewards/lengths/turns."""
    raw_rollouts = []
    for i, reward in enumerate(rewards):
        cl = int(completion_lengths[i]) if completion_lengths is not None else 0
        nt = int(num_turns[i]) if num_turns is not None else 1
        raw_rollouts.append(_make_rollout(float(reward), completion_len=cl, num_turns=nt))
    return _views(raw_rollouts)


# Helper aliases for readability — completion-only and tool-only token shaping.
_TOKENS_COMPLETION = TokensLengthPenaltyConfig(completion_weight=1.0, tool_response_weight=0.0)
_TOKENS_TOOL_ONLY = TokensLengthPenaltyConfig(completion_weight=0.0, tool_response_weight=1.0)


def test_default_advantage_fn_simple_mean():
    result = default_advantage_fn(_make_group(rewards=[1.0, 0.5, 0.8], completion_lengths=[10, 12, 8]))

    assert len(result) == 3
    assert sum(result) == pytest.approx(0.0, abs=1e-6)


def test_max_rl_advantage_fn_mean_normalized():
    # mean 0.25: the success gets (1 - 0.25)/0.25 = 3, failures (0 - 0.25)/0.25 = -1
    result = max_rl_advantage_fn(_make_group(rewards=[1.0, 0.0, 0.0, 0.0]))
    assert result == pytest.approx([3.0, -1.0, -1.0, -1.0])

    # no-success groups carry no signal (the paper's K=0 convention) ...
    assert max_rl_advantage_fn(_make_group(rewards=[0.0, 0.0])) == [0.0, 0.0]
    # ... and all-success groups center to zero like GRPO
    assert max_rl_advantage_fn(_make_group(rewards=[1.0, 1.0])) == pytest.approx([0.0, 0.0])


def test_efficiency_mixed_group():
    """Mixed group: reward shaping preserves zero-mean, shorter correct gets higher advantage."""
    group = _make_group(rewards=[1.0, 1.0, 0.0, 1.0], completion_lengths=[10, 30, 20, 20])
    result = default_advantage_fn(group, length_penalty=_TOKENS_COMPLETION)

    # mean_correct_len = (10+30+20)/3 = 20
    # bonus = clamp(1 - [10,30,20,20]/20, 0, 1) = [0.5, 0, 0, 0]
    # shaped_rewards = R * (1 + bonus * correct_mask) = [1.5, 1, 0, 1]
    # baseline = mean(shaped_rewards) = 0.875
    # A = shaped_rewards - baseline = [0.625, 0.125, -0.875, 0.125]
    assert result == pytest.approx([0.625, 0.125, -0.875, 0.125], abs=1e-6)

    # Zero-mean per group
    assert sum(result) == pytest.approx(0.0, abs=1e-6)

    # All correct rollouts have positive advantage
    for view, adv in zip(group, result):
        if view.reward >= 1.0:
            assert adv > 0


def test_efficiency_all_correct_group():
    """All-correct group: zero-mean, shorter gets higher advantage."""
    result = default_advantage_fn(
        _make_group(rewards=[1.0, 1.0, 1.0], completion_lengths=[10, 20, 40]),
        length_penalty=_TOKENS_COMPLETION,
    )

    # mean_len = 70/3 ≈ 23.33
    # bonus = clamp(1 - [10, 20, 40] / (70/3), 0, 1) = [4/7, 1/7, 0]
    # shaped_rewards = [1+4/7, 1+1/7, 1] = [11/7, 8/7, 1]
    shaped = [11.0 / 7, 8.0 / 7, 1.0]
    mean_shaped = sum(shaped) / len(shaped)
    expected = [s - mean_shaped for s in shaped]
    assert result == pytest.approx(expected, abs=1e-6)

    # Zero-mean
    assert sum(result) == pytest.approx(0.0, abs=1e-6)

    # Shortest has highest advantage
    assert result[0] > result[1] > result[2]


def test_efficiency_all_zero_rewards():
    """When all rewards are 0, no length shaping — falls back to standard GRPO."""
    group = _make_group(rewards=[0.0, 0.0, 0.0], completion_lengths=[10, 20, 15])
    result_with = default_advantage_fn(group, length_penalty=_TOKENS_COMPLETION)
    result_without = default_advantage_fn(group)

    assert result_with == pytest.approx(result_without, abs=1e-6)


def test_efficiency_single_correct():
    """Single correct rollout: bonus=0 (at its own mean), same as standard GRPO."""
    result = default_advantage_fn(
        _make_group(rewards=[1.0, 0.0, 0.0, 0.0], completion_lengths=[100, 50, 200, 150]),
        length_penalty=_TOKENS_COMPLETION,
    )

    assert result == pytest.approx([0.75, -0.25, -0.25, -0.25], abs=1e-6)


def test_efficiency_shorter_correct_higher_advantage():
    """Among correct rollouts in a mixed group, shorter always gets higher advantage."""
    result = default_advantage_fn(
        _make_group(rewards=[1.0, 1.0, 1.0, 0.0, 0.0], completion_lengths=[50, 100, 200, 80, 120]),
        length_penalty=_TOKENS_COMPLETION,
    )

    assert result[0] > result[1] > result[2]
    assert all(a > 0 for a in result[:3])
    assert all(a < 0 for a in result[3:])


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

    assert sum(mixed) == pytest.approx(0.0, abs=1e-6)
    assert sum(all_correct) == pytest.approx(0.0, abs=1e-6)


def test_efficiency_amplification_bounded():
    """Even with extreme length outliers, reward amplification is capped at 2x."""
    result = default_advantage_fn(
        _make_group(rewards=[1.0, 1.0, 0.0], completion_lengths=[1, 10000, 5000]),
        length_penalty=_TOKENS_COMPLETION,
    )

    # Shortest correct gets bonus ≈ 1, so shaped_reward ≈ 2; baseline ≈ 1, max advantage ≈ 1
    assert result[0] < 1.0 + 1e-3


def test_efficiency_tokens_with_tool_response_weight():
    """`tool_response_weight` shifts shaping onto tool-response tokens read from rollout metrics."""
    rollouts = [
        _make_rollout(1.0, completion_len=10, tool_response_len=200),
        _make_rollout(1.0, completion_len=10, tool_response_len=0),
        _make_rollout(1.0, completion_len=10, tool_response_len=100),
    ]
    group = _views(rollouts)

    # completion tokens identical (10 each) → completion-only shaping is a no-op
    result_completion_only = default_advantage_fn(group, length_penalty=_TOKENS_COMPLETION)
    assert result_completion_only == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)

    # tool-response only: costs are [200, 0, 100], mean=100, bonus is one-sided
    # so only the below-mean rollout (idx 1) gets amplified; the at/above-mean tie.
    advs = default_advantage_fn(group, length_penalty=_TOKENS_TOOL_ONLY)
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
    int_result = default_advantage_fn(_views(rollouts_int), length_penalty=fractional)
    float_result = default_advantage_fn(_views(rollouts_float), length_penalty=fractional)
    assert int_result == pytest.approx(float_result, abs=1e-6)


def test_efficiency_zero_costs_falls_back_to_plain_grpo():
    """When all effective costs are zero, shaping is a no-op (no NaNs from div-by-zero)."""
    # tool-only weights but no harness metric → all costs == 0
    rollouts = [
        _make_rollout(1.0, completion_len=10),
        _make_rollout(1.0, completion_len=10),
        _make_rollout(0.0, completion_len=10),
    ]
    group = _views(rollouts)
    result = default_advantage_fn(group, length_penalty=_TOKENS_TOOL_ONLY)
    expected = default_advantage_fn(group)  # plain GRPO
    assert not any(math.isnan(a) for a in result)
    assert result == pytest.approx(expected, abs=1e-6)


def test_efficiency_tokens_default_weights_match_completion_when_no_metric():
    """Default TokensLengthPenaltyConfig (1,1) reduces to completion-only when rollouts lack the metric."""
    group = _make_group(rewards=[1.0, 1.0, 0.0, 1.0], completion_lengths=[10, 30, 20, 20])
    result_default = default_advantage_fn(group, length_penalty=TokensLengthPenaltyConfig())
    result_completion = default_advantage_fn(group, length_penalty=_TOKENS_COMPLETION)
    assert result_default == pytest.approx(result_completion, abs=1e-6)


def test_efficiency_turns_penalty():
    """`TurnsLengthPenaltyConfig` shapes by trajectory turn count rather than token count."""
    result = default_advantage_fn(
        _make_group(
            rewards=[1.0, 1.0, 0.0, 1.0],
            # token counts identical, but turns differ — turns penalty should still differentiate
            completion_lengths=[100, 100, 100, 100],
            num_turns=[1, 3, 2, 2],
        ),
        length_penalty=TurnsLengthPenaltyConfig(),
    )

    # mean_correct_turns = (1+3+2)/3 = 2
    # bonus = clamp(1 - [1,3,2,2]/2, 0, 1) = [0.5, 0, 0, 0]
    assert result == pytest.approx([0.625, 0.125, -0.875, 0.125], abs=1e-6)


def test_rollout_view_assign_advantages_broadcasts_scalar():
    """A scalar broadcasts uniformly over the rollout's completion tokens."""
    rollout = _train_rollout(completion_ids=(2, 3))
    RolloutView(rollout).assign_advantages(0.7)
    assert rollout.advantages == [0.7, 0.7]


def test_rollout_view_assign_advantages_rejects_misaligned():
    rollout = _train_rollout(completion_ids=(2, 3))
    with pytest.raises(ValueError, match="align"):
        RolloutView(rollout).assign_advantages([0.5])


def test_apply_advantage_fn_broadcasts_group_norm():
    rollouts = [_train_rollout(reward=r, completion_ids=(2, 3)) for r in (1.0, 0.5, 0.8)]
    apply_advantage_fn([RolloutView(r) for r in rollouts], default_advantage_fn)
    streams = [r.advantages for r in rollouts]
    # group credit broadcasts uniformly over each rollout's completion tokens
    assert all(len(s) == 2 and s[0] == s[1] for s in streams)
    assert sum(s[0] for s in streams) == pytest.approx(0.0, abs=1e-6)


def test_apply_advantage_fn_singleton_group_is_zero():
    """A group of size 1 has reward == mean, so its advantage is 0."""
    rollouts = [_train_rollout(reward=0.7, completion_ids=(2, 3))]
    apply_advantage_fn([RolloutView(r) for r in rollouts], default_advantage_fn)
    assert rollouts[0].advantages == pytest.approx([0.0, 0.0], abs=1e-6)


def test_custom_advantage_algorithm():
    config = CustomAlgorithmConfig(
        import_path="tests.unit.orchestrator.test_advantage._dummy_custom_advantage",
        kwargs={"scale": 2.0},
    )
    algorithm = CustomAlgorithm(config, policy_pool=None, renderer=None)

    group = _make_group(rewards=[1.0, 0.5, 0.8], completion_lengths=[10, 12, 8])

    result = algorithm.advantage_fn(group)
    assert result == pytest.approx([2.0, 1.0, 1.6], abs=1e-6)


def _dummy_custom_advantage(group: list[RolloutView], scale: float = 1.0) -> list[float]:
    """A simple custom advantage for testing — one scalar per rollout."""
    return [view.reward * scale for view in group]
