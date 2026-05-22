import torch

from prime_rl.configs.orchestrator import (
    CustomAdvantageConfig,
    DefaultAdvantageConfig,
    TokensLengthPenaltyConfig,
    TurnsLengthPenaltyConfig,
)
from prime_rl.orchestrator.advantage import (
    AdvantageInputs,
    AdvantageOutputs,
    compute_advantages,
    default_advantage_fn,
    setup_advantage_fn,
)


def _make_rollout(
    reward: float,
    completion_len: int = 0,
    num_turns: int = 1,
    env_name: str = "test",
    example_id: int = 0,
) -> dict:
    """Create a minimal rollout dict for advantage testing.

    `completion_len` tokens are split across `num_turns` trajectory steps.
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


def _make_inputs(rewards, completion_lengths=None, num_turns=None) -> AdvantageInputs:
    """Build AdvantageInputs from 2D arrays of rewards/lengths/turns."""
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32)
    num_problems, rollouts_per_example = rewards_t.shape
    rollouts = []
    for i in range(num_problems):
        group = []
        for j in range(rollouts_per_example):
            cl = int(completion_lengths[i][j]) if completion_lengths is not None else 0
            nt = int(num_turns[i][j]) if num_turns is not None else 1
            group.append(_make_rollout(float(rewards_t[i, j]), cl, nt, example_id=i))
        rollouts.append(group)
    return AdvantageInputs(rollouts=rollouts)


# Helper aliases for readability — completion-only and tool-only token shaping.
_TOKENS_COMPLETION = TokensLengthPenaltyConfig(completion_weight=1.0, tool_response_weight=0.0)
_TOKENS_TOOL_ONLY = TokensLengthPenaltyConfig(completion_weight=0.0, tool_response_weight=1.0)


def test_default_advantage_fn_simple_mean():
    inputs = _make_inputs(
        rewards=[[1.0, 0.5, 0.8], [0.2, 0.9, 0.1]],
        completion_lengths=[[10, 12, 8], [15, 11, 9]],
    )
    result = default_advantage_fn(inputs)

    assert result.advantages.shape == (2, 3)
    # Check that mean is subtracted per row
    assert torch.allclose(result.advantages.mean(dim=1), torch.zeros(2), atol=1e-6)


def test_efficiency_mixed_group():
    """Mixed group: reward shaping preserves zero-mean, shorter correct gets higher advantage."""
    inputs = _make_inputs(
        rewards=[[1.0, 1.0, 0.0, 1.0]],
        completion_lengths=[[10, 30, 20, 20]],
    )
    result = default_advantage_fn(inputs, length_penalty=_TOKENS_COMPLETION)

    # mean_correct_len = (10+30+20)/3 = 20
    # bonus = clamp(1 - [10,30,20,20]/20, 0, 1) = [0.5, 0, 0, 0]
    # shaped_rewards = R * (1 + bonus * correct_mask) = [1.5, 1, 0, 1]
    # baseline = mean(shaped_rewards) = 0.875
    # A = shaped_rewards - baseline = [0.625, 0.125, -0.875, 0.125]
    expected = torch.tensor([[0.625, 0.125, -0.875, 0.125]])
    assert torch.allclose(result.advantages, expected, atol=1e-6)

    # Zero-mean per group
    assert torch.allclose(result.advantages.mean(dim=1), torch.zeros(1), atol=1e-6)

    # All correct rollouts have positive advantage
    rewards = torch.tensor([r["reward"] for r in inputs.rollouts[0]])
    correct_mask = rewards >= 1.0
    assert (result.advantages[0][correct_mask] > 0).all()


def test_efficiency_all_correct_group():
    """All-correct group: zero-mean, shorter gets higher advantage."""
    inputs = _make_inputs(
        rewards=[[1.0, 1.0, 1.0]],
        completion_lengths=[[10, 20, 40]],
    )
    result = default_advantage_fn(inputs, length_penalty=_TOKENS_COMPLETION)

    # mean_len = 70/3 ≈ 23.33
    # bonus = clamp(1 - [10, 20, 40] / (70/3), 0, 1) = [4/7, 1/7, 0]
    # shaped_rewards = [1+4/7, 1+1/7, 1] = [11/7, 8/7, 1]
    # baseline = mean = (11/7 + 8/7 + 1) / 3 = (11+8+7)/(7*3) = 26/21
    # A = shaped - baseline
    shaped = torch.tensor([[11.0 / 7, 8.0 / 7, 1.0]])
    baseline = shaped.mean(dim=1, keepdim=True)
    expected = shaped - baseline
    assert torch.allclose(result.advantages, expected, atol=1e-6)

    # Zero-mean
    assert torch.allclose(result.advantages.mean(dim=1), torch.zeros(1), atol=1e-6)

    # Shortest has highest advantage
    assert result.advantages[0, 0] > result.advantages[0, 1] > result.advantages[0, 2]


def test_efficiency_all_zero_rewards():
    """When all rewards are 0, no length shaping — falls back to standard GRPO."""
    inputs = _make_inputs(
        rewards=[[0.0, 0.0, 0.0]],
        completion_lengths=[[10, 20, 15]],
    )
    result_with = default_advantage_fn(inputs, length_penalty=_TOKENS_COMPLETION)
    result_without = default_advantage_fn(inputs)

    assert torch.allclose(result_with.advantages, result_without.advantages, atol=1e-6)


def test_efficiency_single_correct():
    """Single correct rollout: bonus=0 (at its own mean), same as standard GRPO."""
    inputs = _make_inputs(
        rewards=[[1.0, 0.0, 0.0, 0.0]],
        completion_lengths=[[100, 50, 200, 150]],
    )
    result = default_advantage_fn(inputs, length_penalty=_TOKENS_COMPLETION)

    expected = torch.tensor([[0.75, -0.25, -0.25, -0.25]])
    assert torch.allclose(result.advantages, expected, atol=1e-6)


def test_efficiency_shorter_correct_higher_advantage():
    """Among correct rollouts in a mixed group, shorter always gets higher advantage."""
    inputs = _make_inputs(
        rewards=[[1.0, 1.0, 1.0, 0.0, 0.0]],
        completion_lengths=[[50, 100, 200, 80, 120]],
    )
    result = default_advantage_fn(inputs, length_penalty=_TOKENS_COMPLETION)

    advs = result.advantages[0]
    assert advs[0] > advs[1] > advs[2]
    assert (advs[:3] > 0).all()
    assert (advs[3:] < 0).all()


def test_efficiency_zero_mean_per_group():
    """Reward shaping preserves zero-mean advantages per group."""
    inputs = _make_inputs(
        rewards=[
            [1.0, 1.0, 0.0, 1.0],  # mixed
            [1.0, 1.0, 1.0, 1.0],  # all correct
        ],
        completion_lengths=[
            [10, 30, 20, 20],
            [10, 20, 40, 80],
        ],
    )
    result = default_advantage_fn(inputs, length_penalty=_TOKENS_COMPLETION)

    assert torch.allclose(result.advantages.mean(dim=1), torch.zeros(2), atol=1e-6)


def test_efficiency_amplification_bounded():
    """Even with extreme length outliers, reward amplification is capped at 2x."""
    inputs = _make_inputs(
        rewards=[[1.0, 1.0, 0.0]],
        completion_lengths=[[1, 10000, 5000]],
    )
    result = default_advantage_fn(inputs, length_penalty=_TOKENS_COMPLETION)

    # Shortest correct gets bonus ≈ 1, so shaped_reward ≈ 2
    # Standard reward = 1, so amplification ≈ 2x
    # shaped_rewards ≈ [2, 1, 0], baseline ≈ 1, max advantage ≈ 1
    assert result.advantages[0, 0] < 1.0 + 1e-3


def test_efficiency_multiple_problems():
    """Handles multiple problems independently."""
    inputs = _make_inputs(
        rewards=[
            [1.0, 1.0, 0.0],  # mixed
            [1.0, 1.0, 1.0],  # all correct
        ],
        completion_lengths=[
            [10, 20, 15],
            [10, 20, 40],
        ],
    )
    result = default_advantage_fn(inputs, length_penalty=_TOKENS_COMPLETION)

    # Row 0: mixed group — shorter correct > longer correct
    assert result.advantages[0, 0] > result.advantages[0, 1]
    assert (result.advantages[0, :2] > 0).all()
    assert result.advantages[0, 2] < 0

    # Row 1: all-correct group — shorter gets higher advantage
    assert result.advantages[1, 0] > result.advantages[1, 1] > result.advantages[1, 2]

    # Both rows have zero-mean
    assert torch.allclose(result.advantages.mean(dim=1), torch.zeros(2), atol=1e-6)


def test_efficiency_tokens_with_tool_response_weight():
    """`tool_response_weight` shifts shaping onto tool-response tokens read from rollout metrics."""
    rollouts = [
        [
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
    ]
    inputs = AdvantageInputs(rollouts=rollouts)

    # completion tokens identical (10 each) → completion-only shaping is a no-op
    result_completion_only = default_advantage_fn(inputs, length_penalty=_TOKENS_COMPLETION)
    assert torch.allclose(result_completion_only.advantages, torch.zeros(1, 3), atol=1e-6)

    # tool-response only: costs are [200, 0, 100], mean=100, bonus is one-sided
    # so only the below-mean rollout (idx 1) gets amplified; the at/above-mean tie.
    result_tool_only = default_advantage_fn(inputs, length_penalty=_TOKENS_TOOL_ONLY)
    advs = result_tool_only.advantages[0]
    assert advs[1] > advs[0]
    assert advs[1] > advs[2]
    assert torch.allclose(advs[0], advs[2], atol=1e-6)
    assert torch.allclose(result_tool_only.advantages.mean(dim=1), torch.zeros(1), atol=1e-6)


def test_efficiency_fractional_weight_with_int_rewards():
    """Fractional weights must not truncate when rollout rewards are emitted as ints."""
    rollouts_int = [
        [
            {"reward": 1, "trajectory": [{"tokens": {"prompt_ids": [0], "completion_ids": list(range(7))}}]},
            {"reward": 1, "trajectory": [{"tokens": {"prompt_ids": [0], "completion_ids": list(range(11))}}]},
            {"reward": 0, "trajectory": [{"tokens": {"prompt_ids": [0], "completion_ids": list(range(13))}}]},
        ]
    ]
    rollouts_float = [[{**r, "reward": float(r["reward"])} for r in g] for g in rollouts_int]

    fractional = TokensLengthPenaltyConfig(completion_weight=0.3, tool_response_weight=0.0)
    int_result = default_advantage_fn(AdvantageInputs(rollouts=rollouts_int), length_penalty=fractional)
    float_result = default_advantage_fn(AdvantageInputs(rollouts=rollouts_float), length_penalty=fractional)
    assert torch.allclose(int_result.advantages, float_result.advantages, atol=1e-6)


def test_efficiency_zero_costs_falls_back_to_plain_grpo():
    """When all effective costs are zero, shaping is a no-op (no NaNs from div-by-zero)."""
    # tool-only weights but no harness metric → all costs == 0
    rollouts = [
        [
            {"reward": 1.0, "trajectory": [{"tokens": {"prompt_ids": [0], "completion_ids": list(range(10))}}]},
            {"reward": 1.0, "trajectory": [{"tokens": {"prompt_ids": [0], "completion_ids": list(range(10))}}]},
            {"reward": 0.0, "trajectory": [{"tokens": {"prompt_ids": [0], "completion_ids": list(range(10))}}]},
        ]
    ]
    inputs = AdvantageInputs(rollouts=rollouts)
    result = default_advantage_fn(inputs, length_penalty=_TOKENS_TOOL_ONLY)
    expected = default_advantage_fn(inputs)  # plain GRPO
    assert not torch.isnan(result.advantages).any()
    assert torch.allclose(result.advantages, expected.advantages, atol=1e-6)


def test_efficiency_tokens_default_weights_match_completion_when_no_metric():
    """Default TokensLengthPenaltyConfig (1,1) reduces to completion-only when rollouts lack the metric."""
    inputs = _make_inputs(
        rewards=[[1.0, 1.0, 0.0, 1.0]],
        completion_lengths=[[10, 30, 20, 20]],
    )
    result_default = default_advantage_fn(inputs, length_penalty=TokensLengthPenaltyConfig())
    result_completion = default_advantage_fn(inputs, length_penalty=_TOKENS_COMPLETION)
    assert torch.allclose(result_default.advantages, result_completion.advantages, atol=1e-6)


def test_efficiency_turns_penalty():
    """`TurnsLengthPenaltyConfig` shapes by trajectory turn count rather than token count."""
    inputs = _make_inputs(
        rewards=[[1.0, 1.0, 0.0, 1.0]],
        # token counts identical, but turns differ — turns penalty should still differentiate
        completion_lengths=[[100, 100, 100, 100]],
        num_turns=[[1, 3, 2, 2]],
    )
    result = default_advantage_fn(inputs, length_penalty=TurnsLengthPenaltyConfig())

    # mean_correct_turns = (1+3+2)/3 = 2
    # bonus = clamp(1 - [1,3,2,2]/2, 0, 1) = [0.5, 0, 0, 0]
    expected = torch.tensor([[0.625, 0.125, -0.875, 0.125]])
    assert torch.allclose(result.advantages, expected, atol=1e-6)


def test_compute_advantages_with_config():
    rewards = [1.0, 0.5, 0.8, 0.2, 0.9, 0.1]
    lengths = [10, 12, 8, 15, 11, 9]
    rollouts = [_make_rollout(r, l, example_id=i // 3) for i, (r, l) in enumerate(zip(rewards, lengths))]

    compute_advantages(rollouts, advantage_config=DefaultAdvantageConfig())

    advantages = [r["advantage"] for r in rollouts]
    assert len(advantages) == 6
    assert abs(sum(advantages[:3])) < 1e-5
    assert abs(sum(advantages[3:])) < 1e-5


def test_compute_advantages_no_cross_group_leakage():
    """Per-problem grouping: each problem must be centered against its own mean, in-order.

    Two problems with very different reward scales — cross-group leakage would pull the
    small-scale group's advantages toward the large-scale group's mean (and vice versa).
    Distinct positional values also catch ordering bugs in the group→flat round-trip.
    """
    rewards = [10.0, 20.0, 30.0, 0.0, 0.1, 0.2]
    rollouts = [_make_rollout(r, example_id=i // 3) for i, r in enumerate(rewards)]

    compute_advantages(rollouts, advantage_config=DefaultAdvantageConfig())

    advantages = [r["advantage"] for r in rollouts]
    expected = [-10.0, 0.0, 10.0, -0.1, 0.0, 0.1]
    for got, want in zip(advantages, expected):
        assert abs(got - want) < 1e-5, (advantages, expected)


def test_compute_advantages_without_config():
    rewards = [1.0, 0.5, 0.8]
    lengths = [10, 12, 8]
    rollouts = [_make_rollout(r, l) for r, l in zip(rewards, lengths)]

    compute_advantages(rollouts, advantage_config=None)

    advantages = [r["advantage"] for r in rollouts]
    assert advantages == rewards


def test_compute_advantages_partial_groups():
    """Partial groups (size < rollouts_per_example) are advantaged against their own mean.

    Bucketing by size lets two groups of different sizes round-trip cleanly: each
    group's advantages must sum to zero and not leak into the other.
    """
    # Group A (example_id=0): 4 rollouts. Group B (example_id=1): 2 rollouts.
    rollouts = [
        _make_rollout(1.0, example_id=0),
        _make_rollout(0.0, example_id=0),
        _make_rollout(1.0, example_id=0),
        _make_rollout(0.0, example_id=0),
        _make_rollout(0.3, example_id=1),
        _make_rollout(0.7, example_id=1),
    ]

    compute_advantages(rollouts, advantage_config=DefaultAdvantageConfig())

    advantages = [r["advantage"] for r in rollouts]
    # Group A: mean=0.5, advantages=[0.5, -0.5, 0.5, -0.5]
    expected_a = [0.5, -0.5, 0.5, -0.5]
    # Group B: mean=0.5, advantages=[-0.2, 0.2]
    expected_b = [-0.2, 0.2]
    for got, want in zip(advantages[:4], expected_a):
        assert abs(got - want) < 1e-5, (advantages[:4], expected_a)
    for got, want in zip(advantages[4:], expected_b):
        assert abs(got - want) < 1e-5, (advantages[4:], expected_b)


def test_compute_advantages_singleton_group_gets_zero_advantage():
    """A group of size 1 has reward == mean, so its advantage is 0 (filterable downstream)."""
    rollouts = [
        _make_rollout(0.5, example_id=0),
        _make_rollout(0.8, example_id=0),
        _make_rollout(0.3, example_id=1),  # singleton group
    ]

    compute_advantages(rollouts, advantage_config=DefaultAdvantageConfig())

    advantages = [r["advantage"] for r in rollouts]
    # Group 0: mean=0.65, advantages=[-0.15, 0.15]
    assert abs(advantages[0] + 0.15) < 1e-5
    assert abs(advantages[1] - 0.15) < 1e-5
    # Group 1 (singleton): advantage=0
    assert abs(advantages[2]) < 1e-5


def test_compute_advantages_disambiguates_example_id_across_envs():
    """example_id=0 in env A and example_id=0 in env B must not be grouped together."""
    rollouts = [
        _make_rollout(1.0, env_name="env_a", example_id=0),
        _make_rollout(0.0, env_name="env_a", example_id=0),
        _make_rollout(100.0, env_name="env_b", example_id=0),
        _make_rollout(200.0, env_name="env_b", example_id=0),
    ]

    compute_advantages(rollouts, advantage_config=DefaultAdvantageConfig())

    advantages = [r["advantage"] for r in rollouts]
    # env_a group: mean=0.5, advantages=[0.5, -0.5]
    assert abs(advantages[0] - 0.5) < 1e-5
    assert abs(advantages[1] + 0.5) < 1e-5
    # env_b group: mean=150, advantages=[-50, 50]
    assert abs(advantages[2] + 50.0) < 1e-5
    assert abs(advantages[3] - 50.0) < 1e-5


def test_setup_advantage_fn_with_custom_config():
    config = CustomAdvantageConfig(
        import_path="tests.unit.orchestrator.test_advantage._dummy_custom_advantage",
        kwargs={"scale": 2.0},
    )
    advantage_fn = setup_advantage_fn(config)

    inputs = _make_inputs(
        rewards=[[1.0, 0.5, 0.8]],
        completion_lengths=[[10, 12, 8]],
    )

    result = advantage_fn(inputs)
    assert isinstance(result, AdvantageOutputs)
    assert torch.allclose(result.advantages, torch.tensor([[2.0, 1.0, 1.6]]))


def _dummy_custom_advantage(inputs: AdvantageInputs, scale: float = 1.0) -> AdvantageOutputs:
    """A simple custom advantage for testing."""
    rewards = torch.tensor([[r["reward"] for r in group] for group in inputs.rollouts])
    return AdvantageOutputs(advantages=rewards * scale)
