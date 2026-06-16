import asyncio
import math
import uuid

import pytest

from prime_rl.configs.algorithm import (
    AlgorithmConfig,
    LinearLengthPenaltyConfig,
    TokensLengthPenaltyConfig,
    TurnsLengthPenaltyConfig,
)
from prime_rl.orchestrator.algo import CustomAlgorithm, build_algorithm
from prime_rl.orchestrator.algo.advantage import (
    apply_advantage_fn,
    efficiency_shaping_advantage,
    grpo_advantage,
    length_penalty_advantage,
    max_rl_advantage_fn,
)
from prime_rl.orchestrator.types import RolloutView, TrainRollout
from prime_rl.transport.types import TrainingSample


def _make_rollout(
    reward: float,
    completion_len: int = 0,
    num_turns: int = 1,
    env_name: str = "test",
    example_id: int = 0,
) -> dict:
    """Create a minimal rollout dict for advantage testing.

    `completion_len` tokens are split across `num_turns` trajectory steps —
    they feed the length penalty's cost computation (read from `raw`), not the
    sample's advantage length.
    """
    per_turn, rem = divmod(completion_len, max(num_turns, 1))
    trajectory = [
        {"tokens": {"prompt_ids": [0], "completion_ids": list(range(per_turn + (rem if i == 0 else 0)))}}
        for i in range(num_turns)
    ]
    return {
        "reward": reward,
        "trajectory": trajectory,
        "env_name": env_name,
        "example_id": example_id,
    }


def _train_rollout(raw: dict, completion_ids: tuple[int, ...] = (2,)) -> TrainRollout:
    """One ``TrainRollout`` carrying a single training sample with the given
    completion tokens (default length 1, so a group-norm scalar is its
    rollout's whole advantage)."""
    return TrainRollout(
        raw=raw,
        env_name="test",
        example_id=0,
        group_id=uuid.uuid4(),
        policy_version=0,
        off_policy_steps=0,
        samples=[
            TrainingSample(
                prompt_ids=[1],
                prompt_mask=[False],
                completion_ids=list(completion_ids),
                completion_mask=[True] * len(completion_ids),
                completion_logprobs=[-0.1] * len(completion_ids),
                completion_temperatures=[],
                env_name="test",
            )
        ],
    )


def _views(raw_rollouts: list[dict]) -> list[RolloutView]:
    """Wrap raw rollout dicts into ``RolloutView``\\ s over single-token
    samples — the advantage fns see exactly what ``score_group`` sees."""
    return [RolloutView(_train_rollout(raw)) for raw in raw_rollouts]


def _make_group(rewards, completion_lengths=None, num_turns=None) -> list[RolloutView]:
    """Build one group of ``RolloutView``\\ s from 1D arrays of rewards/lengths/turns."""
    raw_rollouts = []
    for i, reward in enumerate(rewards):
        cl = int(completion_lengths[i]) if completion_lengths is not None else 0
        nt = int(num_turns[i]) if num_turns is not None else 1
        raw_rollouts.append(_make_rollout(float(reward), cl, nt))
    return _views(raw_rollouts)


# Helper aliases for readability — completion-only and tool-only token shaping.
_TOKENS_COMPLETION = TokensLengthPenaltyConfig(completion_weight=1.0, tool_response_weight=0.0)
_TOKENS_TOOL_ONLY = TokensLengthPenaltyConfig(completion_weight=0.0, tool_response_weight=1.0)


def test_grpo_advantage_simple_mean():
    result = grpo_advantage(_make_group(rewards=[1.0, 0.5, 0.8], completion_lengths=[10, 12, 8]))

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
    result = efficiency_shaping_advantage(group, _TOKENS_COMPLETION)

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
    result = efficiency_shaping_advantage(
        _make_group(rewards=[1.0, 1.0, 1.0], completion_lengths=[10, 20, 40]),
        _TOKENS_COMPLETION,
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
    result_with = efficiency_shaping_advantage(group, _TOKENS_COMPLETION)
    result_without = grpo_advantage(group)

    assert result_with == pytest.approx(result_without, abs=1e-6)


def test_efficiency_single_correct():
    """Single correct rollout: bonus=0 (at its own mean), same as standard GRPO."""
    result = efficiency_shaping_advantage(
        _make_group(rewards=[1.0, 0.0, 0.0, 0.0], completion_lengths=[100, 50, 200, 150]),
        _TOKENS_COMPLETION,
    )

    assert result == pytest.approx([0.75, -0.25, -0.25, -0.25], abs=1e-6)


def test_efficiency_shorter_correct_higher_advantage():
    """Among correct rollouts in a mixed group, shorter always gets higher advantage."""
    result = efficiency_shaping_advantage(
        _make_group(rewards=[1.0, 1.0, 1.0, 0.0, 0.0], completion_lengths=[50, 100, 200, 80, 120]),
        _TOKENS_COMPLETION,
    )

    assert result[0] > result[1] > result[2]
    assert all(a > 0 for a in result[:3])
    assert all(a < 0 for a in result[3:])


def test_efficiency_zero_mean_per_group():
    """Reward shaping preserves zero-mean advantages within each group."""
    mixed = efficiency_shaping_advantage(
        _make_group(rewards=[1.0, 1.0, 0.0, 1.0], completion_lengths=[10, 30, 20, 20]),
        _TOKENS_COMPLETION,
    )
    all_correct = efficiency_shaping_advantage(
        _make_group(rewards=[1.0, 1.0, 1.0, 1.0], completion_lengths=[10, 20, 40, 80]),
        _TOKENS_COMPLETION,
    )

    assert sum(mixed) == pytest.approx(0.0, abs=1e-6)
    assert sum(all_correct) == pytest.approx(0.0, abs=1e-6)


def test_efficiency_amplification_bounded():
    """Even with extreme length outliers, reward amplification is capped at 2x."""
    result = efficiency_shaping_advantage(
        _make_group(rewards=[1.0, 1.0, 0.0], completion_lengths=[1, 10000, 5000]),
        _TOKENS_COMPLETION,
    )

    # Shortest correct gets bonus ≈ 1, so shaped_reward ≈ 2; baseline ≈ 1, max advantage ≈ 1
    assert result[0] < 1.0 + 1e-3


def test_efficiency_tokens_with_tool_response_weight():
    """`tool_response_weight` shifts shaping onto tool-response tokens read from rollout metrics."""
    rollouts = [
        {
            "reward": 1.0,
            "trajectory": [{"tokens": {"prompt_ids": [0], "completion_ids": list(range(10))}}],
            "metrics": {"rlm_total_tool_response_tokens": 200},
        },
        {
            "reward": 1.0,
            "trajectory": [{"tokens": {"prompt_ids": [0], "completion_ids": list(range(10))}}],
            "metrics": {"rlm_total_tool_response_tokens": 0},
        },
        {
            "reward": 1.0,
            "trajectory": [{"tokens": {"prompt_ids": [0], "completion_ids": list(range(10))}}],
            "metrics": {"rlm_total_tool_response_tokens": 100},
        },
    ]
    group = _views(rollouts)

    # completion tokens identical (10 each) → completion-only shaping is a no-op
    result_completion_only = efficiency_shaping_advantage(group, _TOKENS_COMPLETION)
    assert result_completion_only == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)

    # tool-response only: costs are [200, 0, 100], mean=100, bonus is one-sided
    # so only the below-mean rollout (idx 1) gets amplified; the at/above-mean tie.
    advs = efficiency_shaping_advantage(group, _TOKENS_TOOL_ONLY)
    assert advs[1] > advs[0]
    assert advs[1] > advs[2]
    assert advs[0] == pytest.approx(advs[2], abs=1e-6)
    assert sum(advs) == pytest.approx(0.0, abs=1e-6)


def test_efficiency_fractional_weight_with_int_rewards():
    """Fractional weights must not truncate when rollout rewards are emitted as ints."""
    rollouts_int = [
        {"reward": 1, "trajectory": [{"tokens": {"prompt_ids": [0], "completion_ids": list(range(7))}}]},
        {"reward": 1, "trajectory": [{"tokens": {"prompt_ids": [0], "completion_ids": list(range(11))}}]},
        {"reward": 0, "trajectory": [{"tokens": {"prompt_ids": [0], "completion_ids": list(range(13))}}]},
    ]
    rollouts_float = [{**r, "reward": float(r["reward"])} for r in rollouts_int]

    fractional = TokensLengthPenaltyConfig(completion_weight=0.3, tool_response_weight=0.0)
    int_result = efficiency_shaping_advantage(_views(rollouts_int), fractional)
    float_result = efficiency_shaping_advantage(_views(rollouts_float), fractional)
    assert int_result == pytest.approx(float_result, abs=1e-6)


def test_efficiency_zero_costs_falls_back_to_plain_grpo():
    """When all effective costs are zero, shaping is a no-op (no NaNs from div-by-zero)."""
    # tool-only weights but no harness metric → all costs == 0
    rollouts = [
        {"reward": 1.0, "trajectory": [{"tokens": {"prompt_ids": [0], "completion_ids": list(range(10))}}]},
        {"reward": 1.0, "trajectory": [{"tokens": {"prompt_ids": [0], "completion_ids": list(range(10))}}]},
        {"reward": 0.0, "trajectory": [{"tokens": {"prompt_ids": [0], "completion_ids": list(range(10))}}]},
    ]
    group = _views(rollouts)
    result = efficiency_shaping_advantage(group, _TOKENS_TOOL_ONLY)
    expected = grpo_advantage(group)  # plain GRPO
    assert not any(math.isnan(a) for a in result)
    assert result == pytest.approx(expected, abs=1e-6)


def test_efficiency_tokens_default_weights_match_completion_when_no_metric():
    """Default TokensLengthPenaltyConfig (1,1) reduces to completion-only when rollouts lack the metric."""
    group = _make_group(rewards=[1.0, 1.0, 0.0, 1.0], completion_lengths=[10, 30, 20, 20])
    result_default = efficiency_shaping_advantage(group, TokensLengthPenaltyConfig())
    result_completion = efficiency_shaping_advantage(group, _TOKENS_COMPLETION)
    assert result_default == pytest.approx(result_completion, abs=1e-6)


def test_efficiency_turns_penalty():
    """`TurnsLengthPenaltyConfig` shapes by trajectory turn count rather than token count."""
    result = efficiency_shaping_advantage(
        _make_group(
            rewards=[1.0, 1.0, 0.0, 1.0],
            # token counts identical, but turns differ — turns penalty should still differentiate
            completion_lengths=[100, 100, 100, 100],
            num_turns=[1, 3, 2, 2],
        ),
        TurnsLengthPenaltyConfig(),
    )

    # mean_correct_turns = (1+3+2)/3 = 2
    # bonus = clamp(1 - [1,3,2,2]/2, 0, 1) = [0.5, 0, 0, 0]
    assert result == pytest.approx([0.625, 0.125, -0.875, 0.125], abs=1e-6)


def test_linear_penalty_is_grpo_plus_centered_penalty():
    """The linear penalty is a separate additive advantage: grpo_advantage + length_penalty_advantage
    is identical to folding the penalty into the reward before centering (centering is linear)."""
    group = _make_group(rewards=[1.0, 1.0, 0.0, 0.0], completion_lengths=[100, 200, 100, 200])
    cfg = LinearLengthPenaltyConfig(coef=0.25)
    summed = [a + p for a, p in zip(grpo_advantage(group), length_penalty_advantage(group, cfg, 1000), strict=True)]

    # folded reference: (reward - penalty) centered by the plain mean
    rewards = [1.0, 1.0, 0.0, 0.0]
    pass_rate = sum(rewards) / len(rewards)  # 0.5
    penalty = [0.25 * pass_rate * (length / 1000) for length in (100, 200, 100, 200)]
    folded_raw = [r - p for r, p in zip(rewards, penalty)]
    mean = sum(folded_raw) / len(folded_raw)
    folded = [x - mean for x in folded_raw]

    assert summed == pytest.approx(folded, abs=1e-6)
    assert summed == pytest.approx([0.50625, 0.49375, -0.49375, -0.50625], abs=1e-6)
    assert sum(summed) == pytest.approx(0.0, abs=1e-6)
    # shorter beats longer within each outcome
    assert summed[0] > summed[1]
    assert summed[2] > summed[3]


def test_length_penalty_advantage_zero_pass_rate_is_zero():
    """A never-solved group (mean reward 0) has zero penalty everywhere — it adds nothing to GRPO."""
    group = _make_group(rewards=[0.0, 0.0, 0.0], completion_lengths=[10, 20, 30])
    penalty = length_penalty_advantage(group, LinearLengthPenaltyConfig(coef=0.25), max_seq_len=100)
    assert penalty == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)


def test_length_penalty_advantage_uniform_lengths_is_zero():
    """Equal lengths → uniform penalty → the centered term is zero, so it is a no-op on GRPO."""
    group = _make_group(rewards=[1.0, 1.0, 0.0, 0.0], completion_lengths=[100, 100, 100, 100])
    penalty = length_penalty_advantage(group, LinearLengthPenaltyConfig(coef=0.25), max_seq_len=100)
    assert penalty == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=1e-6)


def test_length_penalty_advantage_gate_by_correctness():
    """gate_by_correctness penalizes only correct rollouts; with equal lengths the centered
    penalty term is non-zero (correct rollouts pushed down, incorrect ones up)."""
    group = _make_group(rewards=[1.0, 1.0, 0.0, 0.0], completion_lengths=[100, 100, 100, 100])
    penalty = length_penalty_advantage(
        group, LinearLengthPenaltyConfig(coef=0.25, gate_by_correctness=True), max_seq_len=100
    )
    # penalty_i = 0.25 * 0.5 * 1 * reward = [0.125, 0.125, 0, 0]; centered = mean - penalty
    assert penalty == pytest.approx([-0.0625, -0.0625, 0.0625, 0.0625], abs=1e-6)


def test_length_penalty_advantage_requires_max_seq_len():
    """The linear penalty's denominator is orchestrator.seq_len — missing it is an error, not a guess."""
    group = _make_group(rewards=[1.0, 0.0], completion_lengths=[100, 100])
    with pytest.raises(ValueError, match="max_seq_len"):
        length_penalty_advantage(group, LinearLengthPenaltyConfig(), max_seq_len=None)


def test_grpo_length_weighted_baseline():
    """The length-weighted baseline centers by per-token expected reward:
    sum(len_i * reward_i) / sum(len_i) instead of the plain mean."""
    group = _make_group(rewards=[1.0, 0.0], completion_lengths=[100, 300])
    result = grpo_advantage(group, length_weighted_baseline=True)

    # baseline = (100*1 + 300*0) / 400 = 0.25
    assert result == pytest.approx([0.75, -0.25], abs=1e-6)
    # advantages are length-weighted-zero (not mean-zero)
    assert (100 * result[0] + 300 * result[1]) == pytest.approx(0.0, abs=1e-6)


def test_grpo_algorithm_sums_linear_penalty_end_to_end():
    """build_algorithm injects max_seq_len; GRPOAlgorithm.score_group writes
    grpo_advantage + length_penalty_advantage onto each rollout."""
    config = AlgorithmConfig.model_validate(
        {"advantage": {"type": "grpo", "length_penalty": {"type": "linear", "coef": 0.25}}}
    )
    algorithm = build_algorithm(config, policy_pool=None, renderer=None, max_seq_len=1000)

    rollouts = [
        _train_rollout(_make_rollout(reward, completion_len=length))
        for reward, length in [(1.0, 100), (1.0, 200), (0.0, 100), (0.0, 200)]
    ]
    asyncio.run(algorithm.score_group([RolloutView(r) for r in rollouts]))

    # single-token samples → each rollout's stream is its scalar advantage
    advantages = [r.advantages[0] for r in rollouts]
    assert advantages == pytest.approx([0.50625, 0.49375, -0.49375, -0.50625], abs=1e-6)


def test_rollout_view_assign_advantages_broadcasts_scalar():
    """A scalar broadcasts uniformly over the rollout's completion tokens."""
    rollout = _train_rollout({"reward": 0.0, "trajectory": []}, completion_ids=(2, 3))
    RolloutView(rollout).assign_advantages(0.7)
    assert rollout.advantages == [0.7, 0.7]


def test_rollout_view_assign_advantages_rejects_misaligned():
    rollout = _train_rollout({"reward": 0.0, "trajectory": []}, completion_ids=(2, 3))
    with pytest.raises(ValueError, match="align"):
        RolloutView(rollout).assign_advantages([0.5])


def test_apply_advantage_fn_broadcasts_group_norm():
    rollouts = [_train_rollout({"reward": r, "trajectory": []}, completion_ids=(2, 3)) for r in (1.0, 0.5, 0.8)]
    apply_advantage_fn([RolloutView(r) for r in rollouts], grpo_advantage)
    streams = [r.advantages for r in rollouts]
    # group credit broadcasts uniformly over each rollout's completion tokens
    assert all(len(s) == 2 and s[0] == s[1] for s in streams)
    assert sum(s[0] for s in streams) == pytest.approx(0.0, abs=1e-6)


def test_apply_advantage_fn_singleton_group_is_zero():
    """A group of size 1 has reward == mean, so its advantage is 0."""
    rollouts = [_train_rollout({"reward": 0.7, "trajectory": []}, completion_ids=(2, 3))]
    apply_advantage_fn([RolloutView(r) for r in rollouts], grpo_advantage)
    assert rollouts[0].advantages == pytest.approx([0.0, 0.0], abs=1e-6)


def test_custom_advantage_algorithm():
    config = AlgorithmConfig.model_validate(
        {
            "advantage": {
                "type": "custom",
                "import_path": "tests.unit.orchestrator.test_advantage._dummy_custom_advantage",
                "kwargs": {"scale": 2.0},
            }
        }
    )
    algorithm = CustomAlgorithm(config.advantage, policy_pool=None, renderer=None)

    group = _make_group(rewards=[1.0, 0.5, 0.8], completion_lengths=[10, 12, 8])

    result = algorithm.advantage_fn(group)
    assert result == pytest.approx([2.0, 1.0, 1.6], abs=1e-6)


def _dummy_custom_advantage(group: list[RolloutView], scale: float = 1.0) -> list[float]:
    """A simple custom advantage for testing — one scalar per rollout."""
    return [view.reward * scale for view in group]
