import math

from prime_rl.orchestrator.config import GibberishFilterConfig, RepetitionFilterConfig
from prime_rl.orchestrator.filters import (
    GibberishFilter,
    RepetitionFilter,
    apply_filters,
    setup_filter,
    setup_filters,
)


def _make_rollout(completion_ids, completion_logprobs, reward=1.0, multi_step=False):
    """Create a minimal rollout dict matching the verifiers RolloutOutput structure."""
    if multi_step:
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


def _make_gibberish_filter(vocab_size=128_000, token_id_threshold=100_000, logprob_offset=2.0, enforce=False):
    logprob_threshold = -math.log(vocab_size) - logprob_offset
    return GibberishFilter(
        name="gibberish", token_id_threshold=token_id_threshold, logprob_threshold=logprob_threshold, enforce=enforce
    )


def _make_repetition_filter(window=5, prob_threshold=0.99, enforce=False):
    return RepetitionFilter(
        name="repetition", window=window, logprob_threshold=math.log(prob_threshold), enforce=enforce
    )


# --- GibberishFilter tests ---


def test_gibberish_detects_rare_low_prob_token():
    gibberish_filter = _make_gibberish_filter()

    result = gibberish_filter.check(
        _make_rollout(
            completion_ids=[50, 120_000, 80],
            completion_logprobs=[-1.0, gibberish_filter.logprob_threshold - 1.0, -0.5],
        )
    )
    assert result.detected is True
    assert result.detection_index == 1


def test_gibberish_ignores_normal_tokens():
    gibberish_filter = _make_gibberish_filter()

    result = gibberish_filter.check(
        _make_rollout(
            completion_ids=[10, 200, 5000],
            completion_logprobs=[-1.0, -2.0, -3.0],
        )
    )
    assert result.detected is False
    assert result.detection_index is None


def test_gibberish_ignores_high_prob_rare_token():
    gibberish_filter = _make_gibberish_filter()

    result = gibberish_filter.check(
        _make_rollout(
            completion_ids=[120_000],
            completion_logprobs=[-0.5],
        )
    )
    assert result.detected is False


def test_gibberish_works_across_trajectory_steps():
    gibberish_filter = _make_gibberish_filter()

    result = gibberish_filter.check(
        _make_rollout(
            completion_ids=[50, 60, 120_000, 80],
            completion_logprobs=[-1.0, -0.5, gibberish_filter.logprob_threshold - 1.0, -0.5],
            multi_step=True,
        )
    )
    assert result.detected is True
    assert result.detection_index == 2


# --- RepetitionFilter tests ---


def test_repetition_triggers_after_window():
    repetition_filter = _make_repetition_filter(window=5)

    result = repetition_filter.check(
        _make_rollout(
            completion_ids=list(range(5)),
            completion_logprobs=[-0.001] * 5,
        )
    )
    assert result.detected is True
    assert result.detection_index == 4


def test_repetition_no_trigger_below_window():
    repetition_filter = _make_repetition_filter(window=5)

    result = repetition_filter.check(
        _make_rollout(
            completion_ids=list(range(4)),
            completion_logprobs=[-0.001] * 4,
        )
    )
    assert result.detected is False


def test_repetition_resets_on_low_prob():
    repetition_filter = _make_repetition_filter(window=5)

    logprobs = [-0.001] * 3 + [-2.0] + [-0.001] * 3
    result = repetition_filter.check(
        _make_rollout(
            completion_ids=list(range(7)),
            completion_logprobs=logprobs,
        )
    )
    assert result.detected is False


def test_repetition_varied_probs_no_trigger():
    repetition_filter = _make_repetition_filter(window=3)

    result = repetition_filter.check(
        _make_rollout(
            completion_ids=list(range(6)),
            completion_logprobs=[-0.001, -3.0, -0.001, -3.0, -0.001, -3.0],
        )
    )
    assert result.detected is False


# --- setup_filter / setup_filters tests ---


def test_setup_filter_gibberish():
    config = GibberishFilterConfig(token_id_threshold=100_000, logprob_offset=2.0)
    gibberish_filter = setup_filter(config, vocab_size=128_000)
    assert isinstance(gibberish_filter, GibberishFilter)
    assert gibberish_filter.name == "gibberish"
    assert gibberish_filter.token_id_threshold == 100_000
    assert abs(gibberish_filter.logprob_threshold - (-math.log(128_000) - 2.0)) < 1e-10
    assert gibberish_filter.enforce is False


def test_setup_filter_gibberish_uses_config_vocab_size():
    config = GibberishFilterConfig(vocab_size=64_000)
    gibberish_filter = setup_filter(config, vocab_size=128_000)
    assert abs(gibberish_filter.logprob_threshold - (-math.log(64_000) - 2.0)) < 1e-10


def test_setup_filter_gibberish_enforce():
    config = GibberishFilterConfig(enforce=True)
    gibberish_filter = setup_filter(config, vocab_size=128_000)
    assert gibberish_filter.enforce is True


def test_setup_filter_repetition():
    config = RepetitionFilterConfig(window=3_000, prob_threshold=0.99)
    repetition_filter = setup_filter(config, vocab_size=128_000)
    assert isinstance(repetition_filter, RepetitionFilter)
    assert repetition_filter.name == "repetition"
    assert repetition_filter.window == 3_000
    assert abs(repetition_filter.logprob_threshold - math.log(0.99)) < 1e-10
    assert repetition_filter.enforce is False


def test_setup_filter_repetition_enforce():
    config = RepetitionFilterConfig(enforce=True)
    repetition_filter = setup_filter(config, vocab_size=128_000)
    assert repetition_filter.enforce is True


def test_setup_filters_multiple():
    configs = [
        GibberishFilterConfig(),
        RepetitionFilterConfig(),
    ]
    filters = setup_filters(configs, vocab_size=128_000)
    assert len(filters) == 2
    assert filters[0].name == "gibberish"
    assert filters[1].name == "repetition"


# --- apply_filters tests (enforce=True) ---


def test_apply_filters_enforced_zeros_reward_and_mask():
    gibberish_filter = _make_gibberish_filter(enforce=True)

    rollout = _make_rollout(
        completion_ids=[120_000],
        completion_logprobs=[gibberish_filter.logprob_threshold - 1.0],
        reward=1.0,
    )

    metrics = apply_filters([gibberish_filter], [rollout])

    assert rollout["reward"] == 0.0
    assert all(m == 0 for m in rollout["trajectory"][0]["tokens"]["completion_mask"])
    assert rollout["stop_condition"] == "gibberish"
    assert rollout["metrics"]["filter/gibberish"] == 1.0
    assert metrics["filter/gibberish_count"] == 1.0
    assert metrics["filter/gibberish_rate"] == 1.0
    assert metrics["filter/total_detected_rate"] == 1.0
    assert metrics["filter/total_enforced_rate"] == 1.0


def test_apply_filters_preserves_clean_rollouts():
    gibberish_filter = _make_gibberish_filter(enforce=True)

    rollout = _make_rollout(
        completion_ids=[50, 60, 70],
        completion_logprobs=[-1.0, -2.0, -1.5],
        reward=1.0,
    )

    metrics = apply_filters([gibberish_filter], [rollout])

    assert rollout["reward"] == 1.0
    assert all(m == 1 for m in rollout["trajectory"][0]["tokens"]["completion_mask"])
    assert rollout["stop_condition"] is None
    assert metrics["filter/gibberish_count"] == 0.0
    assert metrics["filter/total_detected_rate"] == 0.0
    assert metrics["filter/total_enforced_rate"] == 0.0


def test_apply_filters_first_filter_wins():
    gibberish_filter = _make_gibberish_filter(enforce=True)
    repetition_filter = _make_repetition_filter(window=2, enforce=True)

    rollout = _make_rollout(
        completion_ids=[120_000, 1, 2],
        completion_logprobs=[gibberish_filter.logprob_threshold - 1.0, -0.001, -0.001],
        reward=1.0,
    )

    metrics = apply_filters([gibberish_filter, repetition_filter], [rollout])

    assert rollout["stop_condition"] == "gibberish"
    assert metrics["filter/gibberish_count"] == 1.0
    assert metrics["filter/repetition_count"] == 0.0
    assert metrics["filter/total_detected_rate"] == 1.0
    assert metrics["filter/total_enforced_rate"] == 1.0


def test_apply_filters_empty_list():
    rollout = _make_rollout(
        completion_ids=[1, 2, 3],
        completion_logprobs=[-1.0, -1.0, -1.0],
    )
    metrics = apply_filters([], [rollout])
    assert metrics == {}
    assert rollout["reward"] == 1.0


def test_apply_filters_mixed_batch():
    gibberish_filter = _make_gibberish_filter(enforce=True)

    clean = _make_rollout(completion_ids=[50], completion_logprobs=[-1.0], reward=1.0)
    dirty = _make_rollout(
        completion_ids=[120_000], completion_logprobs=[gibberish_filter.logprob_threshold - 1.0], reward=1.0
    )

    metrics = apply_filters([gibberish_filter], [clean, dirty])

    assert clean["reward"] == 1.0
    assert dirty["reward"] == 0.0
    assert metrics["filter/gibberish_count"] == 1.0
    assert metrics["filter/gibberish_rate"] == 0.5
    assert metrics["filter/total_detected_rate"] == 0.5
    assert metrics["filter/total_enforced_rate"] == 0.5


# --- apply_filters tests (monitor-only, enforce=False) ---


def test_apply_filters_monitor_only_tracks_detection():
    gibberish_filter = _make_gibberish_filter(enforce=False)

    rollout = _make_rollout(
        completion_ids=[120_000],
        completion_logprobs=[gibberish_filter.logprob_threshold - 1.0],
        reward=1.0,
    )

    metrics = apply_filters([gibberish_filter], [rollout])

    assert rollout["reward"] == 1.0
    assert all(m == 1 for m in rollout["trajectory"][0]["tokens"]["completion_mask"])
    assert rollout["stop_condition"] is None
    assert rollout["metrics"]["filter/gibberish"] == 1.0
    assert metrics["filter/gibberish_count"] == 1.0
    assert metrics["filter/gibberish_rate"] == 1.0
    assert metrics["filter/total_detected_rate"] == 1.0
    assert metrics["filter/total_enforced_rate"] == 0.0


def test_apply_filters_monitor_only_mixed_batch():
    gibberish_filter = _make_gibberish_filter(enforce=False)

    clean = _make_rollout(completion_ids=[50], completion_logprobs=[-1.0], reward=1.0)
    dirty = _make_rollout(
        completion_ids=[120_000], completion_logprobs=[gibberish_filter.logprob_threshold - 1.0], reward=1.0
    )

    metrics = apply_filters([gibberish_filter], [clean, dirty])

    assert clean["reward"] == 1.0
    assert dirty["reward"] == 1.0
    assert metrics["filter/gibberish_rate"] == 0.5
    assert metrics["filter/total_detected_rate"] == 0.5
    assert metrics["filter/total_enforced_rate"] == 0.0
