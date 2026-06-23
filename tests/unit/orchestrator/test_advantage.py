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
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.orchestrator.types import Rollout, RolloutView

# Token-cost penalty over completion tokens only (the harness tool-response
# metric is absent on these rollouts, so the tool weight contributes nothing).
_TOKENS_COMPLETION = TokensLengthPenaltyConfig(completion_weight=1.0, tool_response_weight=0.0)
# Token-cost penalty over tool-response tokens only.
_TOKENS_TOOL_ONLY = TokensLengthPenaltyConfig(completion_weight=0.0, tool_response_weight=1.0)


def _build_rollout(
    reward: float,
    *,
    sampled_lengths: list[int],
    obs_lengths: list[int] | None = None,
    env_name: str = "test",
    metrics: dict | None = None,
) -> Rollout:
    """Build a ``Rollout`` (a ``vf.Trace``) as an alternating message graph.

    ``sampled_lengths`` gives the token count of each model turn (a sampled
    ``AssistantMessage`` node); ``obs_lengths`` (one shorter, if given) gives the
    token count of the non-sampled observation node injected *after* each turn
    (tool output / user feedback). ``samples`` is built via the real
    ``trace_to_samples`` so the rollout matches what ``score_group`` sees.
    """
    obs_lengths = obs_lengths or []
    nodes: list[vf.MessageNode] = []
    parent: int | None = None
    next_token = 0

    def _take(n: int) -> list[int]:
        nonlocal next_token
        ids = list(range(next_token, next_token + n))
        next_token += n
        return ids

    # Leading user prompt (never trainable).
    prompt_ids = _take(1)
    nodes.append(
        vf.MessageNode(
            message=vf.UserMessage(content="q"),
            token_ids=prompt_ids,
            mask=[False] * len(prompt_ids),
            logprobs=[0.0] * len(prompt_ids),
            sampled=False,
            parent=parent,
        )
    )
    parent = len(nodes) - 1

    for i, n_sampled in enumerate(sampled_lengths):
        ids = _take(n_sampled)
        nodes.append(
            vf.MessageNode(
                message=vf.AssistantMessage(content="a"),
                token_ids=ids,
                mask=[True] * n_sampled,
                logprobs=[-0.1] * n_sampled,
                sampled=True,
                parent=parent,
            )
        )
        parent = len(nodes) - 1
        if i < len(obs_lengths):
            obs_ids = _take(obs_lengths[i])
            nodes.append(
                vf.MessageNode(
                    message=vf.ToolMessage(content="t", tool_call_id="x"),
                    token_ids=obs_ids,
                    mask=[False] * obs_lengths[i],
                    logprobs=[0.0] * obs_lengths[i],
                    sampled=False,
                    parent=parent,
                )
            )
            parent = len(nodes) - 1

    rollout = Rollout[vf.Task](
        task=vf.Task(idx=0, prompt=None),
        nodes=nodes,
        rewards={"reward": reward},
        metrics=metrics or {},
    )
    rollout.env_name = env_name
    rollout.samples = trace_to_samples(rollout, env_name=env_name)
    return rollout


def _make_rollout(
    reward: float,
    completion_len: int = 1,
    num_turns: int = 1,
    env_name: str = "test",
    metrics: dict | None = None,
) -> Rollout:
    """Build a ``Rollout`` carrying ``completion_len`` model-sampled tokens split
    across ``num_turns`` sampled turns. Always carries at least one trainable
    token so credit broadcasts somewhere."""
    num_turns = max(num_turns, 1)
    per_turn, rem = divmod(max(completion_len, 1), num_turns)
    sampled_lengths = [per_turn + (rem if i == 0 else 0) for i in range(num_turns)]
    sampled_lengths = [max(n, 1) for n in sampled_lengths]
    return _build_rollout(reward, sampled_lengths=sampled_lengths, env_name=env_name, metrics=metrics)


def _views(rollouts: list[Rollout]) -> list[RolloutView]:
    """Wrap rollouts into ``RolloutView``\\ s — the advantage fns see exactly what
    ``score_group`` sees."""
    return [RolloutView(r) for r in rollouts]


def _make_group(rewards, completion_lengths=None, num_turns=None) -> list[RolloutView]:
    """Build one group of ``RolloutView``\\ s from 1D arrays of rewards/lengths/turns."""
    rollouts = []
    for i, reward in enumerate(rewards):
        cl = int(completion_lengths[i]) if completion_lengths is not None else 1
        nt = int(num_turns[i]) if num_turns is not None else 1
        rollouts.append(_make_rollout(float(reward), cl, nt))
    return _views(rollouts)


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


def test_efficiency_zero_mean_preserved():
    """Reward shaping preserves zero-mean for both mixed and all-correct groups."""
    mixed = default_advantage_fn(
        _make_group(rewards=[1.0, 1.0, 0.0, 1.0], completion_lengths=[10, 30, 20, 20]),
        length_penalty=_TOKENS_COMPLETION,
    )
    all_correct = default_advantage_fn(
        _make_group(rewards=[1.0, 1.0, 1.0], completion_lengths=[10, 20, 40]),
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
        _build_rollout(1.0, sampled_lengths=[10], metrics={"rlm_total_tool_response_tokens": 200}),
        _build_rollout(1.0, sampled_lengths=[10], metrics={"rlm_total_tool_response_tokens": 0}),
        _build_rollout(1.0, sampled_lengths=[10], metrics={"rlm_total_tool_response_tokens": 100}),
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
    lengths = [7, 11, 13]
    rewards_int = [1, 1, 0]
    rollouts_int = [_build_rollout(r, sampled_lengths=[length]) for r, length in zip(rewards_int, lengths)]
    rollouts_float = [_build_rollout(float(r), sampled_lengths=[length]) for r, length in zip(rewards_int, lengths)]

    fractional = TokensLengthPenaltyConfig(completion_weight=0.3, tool_response_weight=0.0)
    int_result = default_advantage_fn(_views(rollouts_int), length_penalty=fractional)
    float_result = default_advantage_fn(_views(rollouts_float), length_penalty=fractional)
    assert int_result == pytest.approx(float_result, abs=1e-6)


def test_efficiency_zero_costs_falls_back_to_plain_grpo():
    """When all effective costs are zero, shaping is a no-op (no NaNs from div-by-zero)."""
    # tool-only weights but no harness metric → all costs == 0
    group = _make_group(rewards=[1.0, 1.0, 0.0], completion_lengths=[10, 10, 10])
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
    """A scalar broadcasts uniformly over the rollout's trainable (mask-True) tokens."""
    rollout = _build_rollout(0.0, sampled_lengths=[2])
    # one user prompt token (masked) + 2 sampled tokens (trainable)
    RolloutView(rollout).assign_advantages(0.7)
    assert rollout.advantages == [0.0, 0.7, 0.7]


def test_rollout_view_assign_advantages_zeros_non_trainable():
    """Non-trainable (mask=False) positions stay 0.0 under scalar broadcast."""
    # prompt(1, masked) + sampled(1) + obs(1, masked): mask is [F, T, F]
    rollout = _build_rollout(0.0, sampled_lengths=[1], obs_lengths=[1])
    RolloutView(rollout).assign_advantages(0.7)
    assert rollout.advantages == [0.0, 0.7, 0.0]


def test_rollout_view_assign_advantages_rejects_misaligned():
    rollout = _build_rollout(0.0, sampled_lengths=[2])
    # full length is 3 (prompt + 2 sampled); a 1-element list must be rejected
    with pytest.raises(ValueError, match="align"):
        RolloutView(rollout).assign_advantages([0.5])


def test_apply_advantage_fn_broadcasts_group_norm():
    rollouts = [_build_rollout(r, sampled_lengths=[2]) for r in (1.0, 0.5, 0.8)]
    apply_advantage_fn([RolloutView(r) for r in rollouts], default_advantage_fn)
    # each rollout: [prompt(masked) -> 0.0, sampled, sampled] all sharing the group scalar
    streams = [r.advantages for r in rollouts]
    for s in streams:
        assert len(s) == 3
        assert s[0] == 0.0
        assert s[1] == s[2]
    assert sum(s[1] for s in streams) == pytest.approx(0.0, abs=1e-6)


def test_apply_advantage_fn_singleton_group_is_zero():
    """A group of size 1 has reward == mean, so its advantage is 0."""
    rollout = _build_rollout(0.7, sampled_lengths=[2])
    apply_advantage_fn([RolloutView(rollout)], default_advantage_fn)
    assert rollout.advantages == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)


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
