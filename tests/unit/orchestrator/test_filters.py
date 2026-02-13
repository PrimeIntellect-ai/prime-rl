import math

from prime_rl.orchestrator.filters import (
    GibberishFilter,
    RepetitionFilter,
    apply_filters,
)


def _make_rollout(completion_ids, completion_logprobs, reward=1.0, multi_step=False):
    """Create a minimal rollout dict matching the verifiers RolloutOutput structure."""
    if multi_step:
        # Split into two steps
        mid = len(completion_ids) // 2
        trajectory = [
            {
                "tokens": {
                    "completion_ids": completion_ids[:mid],
                    "completion_logprobs": completion_logprobs[:mid],
                    "completion_mask": [1] * mid,
                }
            },
            {
                "tokens": {
                    "completion_ids": completion_ids[mid:],
                    "completion_logprobs": completion_logprobs[mid:],
                    "completion_mask": [1] * (len(completion_ids) - mid),
                }
            },
        ]
    else:
        trajectory = [
            {
                "tokens": {
                    "completion_ids": completion_ids,
                    "completion_logprobs": completion_logprobs,
                    "completion_mask": [1] * len(completion_ids),
                }
            }
        ]
    return {
        "trajectory": trajectory,
        "reward": reward,
        "stop_condition": None,
        "metrics": {},
    }


# --- GibberishFilter tests ---


def test_gibberish_detects_rare_low_prob_token():
    vocab_size = 128_000
    logprob_threshold = -math.log(vocab_size) - 2.0
    f = GibberishFilter(token_id_threshold=100_000, logprob_threshold=logprob_threshold)

    # Token 120000 > 100000, logprob well below threshold
    result = f.check(
        _make_rollout(
            completion_ids=[50, 120_000, 80],
            completion_logprobs=[-1.0, logprob_threshold - 1.0, -0.5],
        )
    )
    assert result.detected is True
    assert result.detection_index == 1


def test_gibberish_ignores_normal_tokens():
    vocab_size = 128_000
    logprob_threshold = -math.log(vocab_size) - 2.0
    f = GibberishFilter(token_id_threshold=100_000, logprob_threshold=logprob_threshold)

    result = f.check(
        _make_rollout(
            completion_ids=[10, 200, 5000],
            completion_logprobs=[-1.0, -2.0, -3.0],
        )
    )
    assert result.detected is False
    assert result.detection_index is None


def test_gibberish_ignores_high_prob_rare_token():
    vocab_size = 128_000
    logprob_threshold = -math.log(vocab_size) - 2.0
    f = GibberishFilter(token_id_threshold=100_000, logprob_threshold=logprob_threshold)

    # Token ID is rare but logprob is above threshold (model is confident)
    result = f.check(
        _make_rollout(
            completion_ids=[120_000],
            completion_logprobs=[-0.5],
        )
    )
    assert result.detected is False


def test_gibberish_works_across_trajectory_steps():
    vocab_size = 128_000
    logprob_threshold = -math.log(vocab_size) - 2.0
    f = GibberishFilter(token_id_threshold=100_000, logprob_threshold=logprob_threshold)

    # Gibberish token is in the second step
    result = f.check(
        _make_rollout(
            completion_ids=[50, 60, 120_000, 80],
            completion_logprobs=[-1.0, -0.5, logprob_threshold - 1.0, -0.5],
            multi_step=True,
        )
    )
    assert result.detected is True
    assert result.detection_index == 2


def test_gibberish_from_config():
    class FakeConfig:
        token_id_threshold = 100_000
        logprob_offset = 2.0

    f = GibberishFilter.from_config(FakeConfig(), vocab_size=128_000)
    expected_threshold = -math.log(128_000) - 2.0
    assert abs(f.logprob_threshold - expected_threshold) < 1e-10
    assert f.token_id_threshold == 100_000


# --- RepetitionFilter tests ---


def test_repetition_triggers_after_window():
    f = RepetitionFilter(window=5, logprob_threshold=math.log(0.99))

    # 5 consecutive high-prob tokens should trigger
    result = f.check(
        _make_rollout(
            completion_ids=list(range(5)),
            completion_logprobs=[-0.001] * 5,  # all > log(0.99) ≈ -0.01
        )
    )
    assert result.detected is True
    assert result.detection_index == 4


def test_repetition_no_trigger_below_window():
    f = RepetitionFilter(window=5, logprob_threshold=math.log(0.99))

    result = f.check(
        _make_rollout(
            completion_ids=list(range(4)),
            completion_logprobs=[-0.001] * 4,
        )
    )
    assert result.detected is False


def test_repetition_resets_on_low_prob():
    f = RepetitionFilter(window=5, logprob_threshold=math.log(0.99))

    # 3 high-prob, 1 low-prob (resets), 3 high-prob — never reaches window of 5
    logprobs = [-0.001] * 3 + [-2.0] + [-0.001] * 3
    result = f.check(
        _make_rollout(
            completion_ids=list(range(7)),
            completion_logprobs=logprobs,
        )
    )
    assert result.detected is False


def test_repetition_varied_probs_no_trigger():
    f = RepetitionFilter(window=3, logprob_threshold=math.log(0.99))

    # Mix of high and low probability tokens
    result = f.check(
        _make_rollout(
            completion_ids=list(range(6)),
            completion_logprobs=[-0.001, -3.0, -0.001, -3.0, -0.001, -3.0],
        )
    )
    assert result.detected is False


def test_repetition_from_config():
    class FakeConfig:
        window = 3_000
        prob_threshold = 0.99

    f = RepetitionFilter.from_config(FakeConfig())
    assert f.window == 3_000
    assert abs(f.logprob_threshold - math.log(0.99)) < 1e-10


# --- apply_filters tests ---


def test_apply_filters_zeros_reward_and_mask():
    vocab_size = 128_000
    logprob_threshold = -math.log(vocab_size) - 2.0
    gf = GibberishFilter(token_id_threshold=100_000, logprob_threshold=logprob_threshold)

    rollout = _make_rollout(
        completion_ids=[120_000],
        completion_logprobs=[logprob_threshold - 1.0],
        reward=1.0,
    )

    metrics = apply_filters([gf], [rollout])

    assert rollout["reward"] == 0.0
    assert all(m == 0 for m in rollout["trajectory"][0]["tokens"]["completion_mask"])
    assert rollout["stop_condition"] == "gibberish"
    assert rollout["metrics"]["filter/gibberish"] == 1.0
    assert metrics["filter/gibberish_count"] == 1.0
    assert metrics["filter/gibberish_rate"] == 1.0
    assert metrics["filter/total_filtered_rate"] == 1.0


def test_apply_filters_preserves_clean_rollouts():
    vocab_size = 128_000
    logprob_threshold = -math.log(vocab_size) - 2.0
    gf = GibberishFilter(token_id_threshold=100_000, logprob_threshold=logprob_threshold)

    rollout = _make_rollout(
        completion_ids=[50, 60, 70],
        completion_logprobs=[-1.0, -2.0, -1.5],
        reward=1.0,
    )

    metrics = apply_filters([gf], [rollout])

    assert rollout["reward"] == 1.0
    assert all(m == 1 for m in rollout["trajectory"][0]["tokens"]["completion_mask"])
    assert rollout["stop_condition"] is None
    assert metrics["filter/gibberish_count"] == 0.0
    assert metrics["filter/total_filtered_rate"] == 0.0


def test_apply_filters_first_filter_wins():
    vocab_size = 128_000
    logprob_threshold = -math.log(vocab_size) - 2.0
    gf = GibberishFilter(token_id_threshold=100_000, logprob_threshold=logprob_threshold)
    rf = RepetitionFilter(window=2, logprob_threshold=math.log(0.99))

    # This rollout triggers BOTH filters (rare low-prob token + high-prob streak)
    rollout = _make_rollout(
        completion_ids=[120_000, 1, 2],
        completion_logprobs=[logprob_threshold - 1.0, -0.001, -0.001],
        reward=1.0,
    )

    # Gibberish is listed first, so it should win
    metrics = apply_filters([gf, rf], [rollout])

    assert rollout["stop_condition"] == "gibberish"
    assert metrics["filter/gibberish_count"] == 1.0
    assert metrics["filter/repetition_count"] == 0.0
    assert metrics["filter/total_filtered_rate"] == 1.0


def test_apply_filters_empty_list():
    rollout = _make_rollout(
        completion_ids=[1, 2, 3],
        completion_logprobs=[-1.0, -1.0, -1.0],
    )
    metrics = apply_filters([], [rollout])
    assert metrics == {}
    assert rollout["reward"] == 1.0


def test_apply_filters_mixed_batch():
    vocab_size = 128_000
    logprob_threshold = -math.log(vocab_size) - 2.0
    gf = GibberishFilter(token_id_threshold=100_000, logprob_threshold=logprob_threshold)

    clean = _make_rollout(completion_ids=[50], completion_logprobs=[-1.0], reward=1.0)
    dirty = _make_rollout(completion_ids=[120_000], completion_logprobs=[logprob_threshold - 1.0], reward=1.0)

    metrics = apply_filters([gf], [clean, dirty])

    assert clean["reward"] == 1.0
    assert dirty["reward"] == 0.0
    assert metrics["filter/gibberish_count"] == 1.0
    assert metrics["filter/gibberish_rate"] == 0.5
    assert metrics["filter/total_filtered_rate"] == 0.5
