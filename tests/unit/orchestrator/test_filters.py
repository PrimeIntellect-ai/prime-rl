import math
import uuid

import pytest
from pydantic import ValidationError

from prime_rl.configs.orchestrator import (
    DefaultAdvantageConfig,
    GibberishFilterConfig,
    RepetitionFilterConfig,
    ZeroAdvantageFilterConfig,
)
from prime_rl.orchestrator.advantage import assign_advantages, setup_advantage_fn
from prime_rl.orchestrator.filters import (
    GibberishFilter,
    RepetitionFilter,
    ZeroAdvantageFilter,
    apply_filters,
    setup_filter,
    setup_filters,
    split_filters,
)
from prime_rl.orchestrator.types import TrainRollout


def _make_rollout(
    completion_ids: list[int],
    completion_logprobs: list[float],
    *,
    reward: float = 1.0,
    multi_step: bool = False,
) -> TrainRollout:
    """Build a ``TrainRollout`` with a minimal ``vf.RolloutOutput``-shaped
    raw payload — enough for the filters to inspect ``trajectory`` /
    ``stop_condition`` / etc."""
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
    raw = {
        "trajectory": trajectory,
        "reward": reward,
        "stop_condition": None,
        "metrics": {},
    }
    return TrainRollout(
        raw=raw,
        env_name="test",
        example_id=0,
        group_id=uuid.uuid4(),
        policy_version=0,
        off_policy_steps=0,
    )


def _make_gibberish_filter(
    vocab_size=128_000, token_id_threshold=100_000, logprob_offset=2.0, action="monitor", penalty_reward=-1.0
):
    logprob_threshold = -math.log(vocab_size) - logprob_offset
    return GibberishFilter(
        name="gibberish",
        token_id_threshold=token_id_threshold,
        logprob_threshold=logprob_threshold,
        action=action,
        penalty_reward=penalty_reward,
    )


def _make_repetition_filter(window=5, prob_threshold=0.99, action="monitor", penalty_reward=-1.0):
    return RepetitionFilter(
        name="repetition",
        window=window,
        logprob_threshold=math.log(prob_threshold),
        action=action,
        penalty_reward=penalty_reward,
    )


def _make_dirty_rollout(gibberish_filter, *, reward: float = 1.0) -> TrainRollout:
    """A rollout that triggers the given gibberish filter."""
    return _make_rollout(
        completion_ids=[120_000],
        completion_logprobs=[gibberish_filter.logprob_threshold - 1.0],
        reward=reward,
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


# --- config resolution / validation tests ---


def test_config_default_actions():
    assert GibberishFilterConfig().resolved_action == "monitor"
    assert RepetitionFilterConfig().resolved_action == "monitor"
    assert ZeroAdvantageFilterConfig().resolved_action == "drop"


def test_config_legacy_enforce_true_resolves_to_drop():
    assert GibberishFilterConfig(enforce=True).resolved_action == "drop"
    assert RepetitionFilterConfig(enforce=True).resolved_action == "drop"
    assert ZeroAdvantageFilterConfig(enforce=True).resolved_action == "drop"


def test_config_legacy_enforce_false_resolves_to_monitor():
    assert GibberishFilterConfig(enforce=False).resolved_action == "monitor"
    assert RepetitionFilterConfig(enforce=False).resolved_action == "monitor"
    assert ZeroAdvantageFilterConfig(enforce=False).resolved_action == "monitor"


def test_config_penalize_parses_with_penalty_reward():
    config = GibberishFilterConfig(action="penalize", penalty_reward=-0.5)
    assert config.resolved_action == "penalize"
    assert config.penalty_reward == -0.5


def test_config_conflicting_action_and_enforce_raises():
    with pytest.raises(ValidationError):
        GibberishFilterConfig(action="penalize", enforce=True)
    with pytest.raises(ValidationError):
        RepetitionFilterConfig(action="monitor", enforce=True)
    with pytest.raises(ValidationError):
        ZeroAdvantageFilterConfig(action="drop", enforce=False)


def test_config_consistent_action_and_enforce_ok():
    assert GibberishFilterConfig(action="drop", enforce=True).resolved_action == "drop"
    assert RepetitionFilterConfig(action="monitor", enforce=False).resolved_action == "monitor"


# --- setup_filter / setup_filters tests ---


def test_setup_filter_gibberish():
    config = GibberishFilterConfig(token_id_threshold=100_000, logprob_offset=2.0)
    gibberish_filter = setup_filter(config, vocab_size=128_000)
    assert isinstance(gibberish_filter, GibberishFilter)
    assert gibberish_filter.name == "gibberish"
    assert gibberish_filter.token_id_threshold == 100_000
    assert abs(gibberish_filter.logprob_threshold - (-math.log(128_000) - 2.0)) < 1e-10
    assert gibberish_filter.action == "monitor"
    assert gibberish_filter.phase == "pre_advantage"


def test_setup_filter_gibberish_legacy_enforce():
    config = GibberishFilterConfig(enforce=True)
    gibberish_filter = setup_filter(config, vocab_size=128_000)
    assert gibberish_filter.action == "drop"


def test_setup_filter_gibberish_penalize():
    config = GibberishFilterConfig(action="penalize", penalty_reward=-0.5)
    gibberish_filter = setup_filter(config, vocab_size=128_000)
    assert gibberish_filter.action == "penalize"
    assert gibberish_filter.penalty_reward == -0.5


def test_setup_filter_repetition():
    config = RepetitionFilterConfig(window=3_000, prob_threshold=0.99)
    repetition_filter = setup_filter(config, vocab_size=128_000)
    assert isinstance(repetition_filter, RepetitionFilter)
    assert repetition_filter.name == "repetition"
    assert repetition_filter.window == 3_000
    assert abs(repetition_filter.logprob_threshold - math.log(0.99)) < 1e-10
    assert repetition_filter.action == "monitor"
    assert repetition_filter.phase == "pre_advantage"


def test_setup_filter_repetition_legacy_enforce():
    config = RepetitionFilterConfig(enforce=True)
    repetition_filter = setup_filter(config, vocab_size=128_000)
    assert repetition_filter.action == "drop"


def test_setup_filter_zero_advantage_defaults_to_drop():
    config = ZeroAdvantageFilterConfig()
    zero_advantage_filter = setup_filter(config, vocab_size=128_000)
    assert isinstance(zero_advantage_filter, ZeroAdvantageFilter)
    assert zero_advantage_filter.action == "drop"
    assert zero_advantage_filter.phase == "post_advantage"


def test_setup_filters_multiple():
    configs = [
        GibberishFilterConfig(),
        RepetitionFilterConfig(),
    ]
    filters = setup_filters(configs, vocab_size=128_000, kind="post-batch")
    assert len(filters) == 2
    assert filters[0].name == "gibberish"
    assert filters[1].name == "repetition"


def test_split_filters_by_phase():
    filters = [
        _make_gibberish_filter(),
        _make_repetition_filter(),
        ZeroAdvantageFilter(name="zero_advantage"),
    ]
    pre, post = split_filters(filters)
    assert [f.name for f in pre] == ["gibberish", "repetition"]
    assert [f.name for f in post] == ["zero_advantage"]


# --- apply_filters tests (action="drop") ---


def test_apply_filters_drop_flags_rollout():
    gibberish_filter = _make_gibberish_filter(action="drop")

    rollout = _make_rollout(
        completion_ids=[120_000],
        completion_logprobs=[gibberish_filter.logprob_threshold - 1.0],
        reward=1.0,
    )

    apply_filters([gibberish_filter], [rollout])

    assert rollout.reward == 1.0
    assert rollout.raw["trajectory"][0]["tokens"]["completion_ids"] == [120_000]
    assert rollout.raw["trajectory"][0]["tokens"]["completion_mask"] == [1]
    assert rollout.raw["stop_condition"] is None
    assert rollout.filter_results == {"gibberish": True}
    assert rollout.is_filtered is True


def test_apply_filters_preserves_clean_rollouts():
    gibberish_filter = _make_gibberish_filter(action="drop")

    rollout = _make_rollout(
        completion_ids=[50, 60, 70],
        completion_logprobs=[-1.0, -2.0, -1.5],
        reward=1.0,
    )

    apply_filters([gibberish_filter], [rollout])

    assert rollout.reward == 1.0
    assert rollout.raw["trajectory"][0]["tokens"]["completion_ids"] == [50, 60, 70]
    assert all(m == 1 for m in rollout.raw["trajectory"][0]["tokens"]["completion_mask"])
    assert rollout.raw["stop_condition"] is None
    assert rollout.filter_results == {"gibberish": False}
    assert rollout.is_filtered is False


def test_apply_filters_first_filter_wins():
    gibberish_filter = _make_gibberish_filter(action="drop")
    repetition_filter = _make_repetition_filter(window=2, action="drop")

    rollout = _make_rollout(
        completion_ids=[120_000, 1, 2],
        completion_logprobs=[gibberish_filter.logprob_threshold - 1.0, -0.001, -0.001],
        reward=1.0,
    )

    apply_filters([gibberish_filter, repetition_filter], [rollout])

    assert rollout.raw["stop_condition"] is None
    assert rollout.filter_results == {"gibberish": True, "repetition": False}
    assert rollout.is_filtered is True


def test_apply_filters_empty_list():
    rollout = _make_rollout(
        completion_ids=[1, 2, 3],
        completion_logprobs=[-1.0, -1.0, -1.0],
    )
    apply_filters([], [rollout])
    assert rollout.filter_results == {}
    assert rollout.is_filtered is False
    assert rollout.reward == 1.0


def test_apply_filters_mixed_batch():
    gibberish_filter = _make_gibberish_filter(action="drop")

    clean = _make_rollout(completion_ids=[50], completion_logprobs=[-1.0], reward=1.0)
    dirty = _make_dirty_rollout(gibberish_filter)

    apply_filters([gibberish_filter], [clean, dirty])

    assert clean.reward == 1.0
    assert dirty.reward == 1.0
    assert clean.is_filtered is False
    assert dirty.is_filtered is True


def test_apply_filters_drop_preserves_rollout_tokens():
    gibberish_filter = _make_gibberish_filter(action="drop")

    rollout = _make_rollout(
        completion_ids=[10, 120_000, 30],
        completion_logprobs=[-1.0, gibberish_filter.logprob_threshold - 1.0, -0.5],
        reward=1.0,
    )

    apply_filters([gibberish_filter], [rollout])

    assert rollout.raw["trajectory"][0]["tokens"]["completion_ids"] == [10, 120_000, 30]
    assert rollout.raw["trajectory"][0]["tokens"]["completion_logprobs"] == [
        -1.0,
        gibberish_filter.logprob_threshold - 1.0,
        -0.5,
    ]
    assert rollout.raw["trajectory"][0]["tokens"]["completion_mask"] == [1, 1, 1]
    assert rollout.is_filtered is True


def test_apply_filters_preserves_existing_stop_condition():
    gibberish_filter = _make_gibberish_filter(action="drop")

    rollout = _make_dirty_rollout(gibberish_filter)
    rollout.raw["stop_condition"] = "generation_truncated"

    apply_filters([gibberish_filter], [rollout])

    assert rollout.raw["stop_condition"] == "generation_truncated"
    assert rollout.is_filtered is True


# --- apply_filters tests (action="monitor") ---


def test_apply_filters_monitor_only_tracks_detection():
    gibberish_filter = _make_gibberish_filter(action="monitor")

    rollout = _make_dirty_rollout(gibberish_filter)

    apply_filters([gibberish_filter], [rollout])

    assert rollout.reward == 1.0
    assert all(m == 1 for m in rollout.raw["trajectory"][0]["tokens"]["completion_mask"])
    assert rollout.raw["stop_condition"] is None
    assert rollout.filter_results == {"gibberish": True}
    assert rollout.is_filtered is False
    assert rollout.raw_reward is None
    assert rollout.reward_penalties == {}


def test_apply_filters_monitor_only_mixed_batch():
    gibberish_filter = _make_gibberish_filter(action="monitor")

    clean = _make_rollout(completion_ids=[50], completion_logprobs=[-1.0], reward=1.0)
    dirty = _make_dirty_rollout(gibberish_filter)

    apply_filters([gibberish_filter], [clean, dirty])

    assert clean.reward == 1.0
    assert dirty.reward == 1.0
    assert clean.is_filtered is False
    assert dirty.is_filtered is False


# --- apply_filters tests (action="penalize") ---


def test_penalize_gibberish_caps_reward():
    gibberish_filter = _make_gibberish_filter(action="penalize")

    rollout = _make_dirty_rollout(gibberish_filter, reward=1.0)

    apply_filters([gibberish_filter], [rollout])

    assert rollout.reward == -1.0
    assert rollout.raw_reward == 1.0
    assert rollout.filter_results == {"gibberish": True}
    assert rollout.reward_penalties == {
        "gibberish": {"raw_reward": 1.0, "penalized_reward": -1.0, "detection_index": 0}
    }


def test_penalize_repetition_caps_reward():
    repetition_filter = _make_repetition_filter(window=3, action="penalize")

    rollout = _make_rollout(
        completion_ids=list(range(3)),
        completion_logprobs=[-0.001] * 3,
        reward=0.0,
    )

    apply_filters([repetition_filter], [rollout])

    assert rollout.reward == -1.0
    assert rollout.raw_reward == 0.0
    assert rollout.reward_penalties["repetition"]["detection_index"] == 2


def test_penalize_does_not_filter_rollout():
    gibberish_filter = _make_gibberish_filter(action="penalize")

    rollout = _make_dirty_rollout(gibberish_filter)

    apply_filters([gibberish_filter], [rollout])

    assert rollout.is_filtered is False
    # Trajectory tokens stay untouched — rollout remains trainable
    assert rollout.raw["trajectory"][0]["tokens"]["completion_mask"] == [1]


def test_penalize_respects_custom_penalty_reward():
    gibberish_filter = _make_gibberish_filter(action="penalize", penalty_reward=-0.5)

    rollout = _make_dirty_rollout(gibberish_filter, reward=1.0)

    apply_filters([gibberish_filter], [rollout])

    assert rollout.reward == -0.5


def test_penalize_does_not_improve_already_negative_reward():
    gibberish_filter = _make_gibberish_filter(action="penalize", penalty_reward=-1.0)

    rollout = _make_dirty_rollout(gibberish_filter, reward=-2.0)

    apply_filters([gibberish_filter], [rollout])

    assert rollout.reward == -2.0
    assert rollout.raw_reward == -2.0


def test_penalize_skips_clean_rollouts():
    gibberish_filter = _make_gibberish_filter(action="penalize")

    rollout = _make_rollout(completion_ids=[50], completion_logprobs=[-1.0], reward=1.0)

    apply_filters([gibberish_filter], [rollout])

    assert rollout.reward == 1.0
    assert rollout.raw_reward is None
    assert rollout.reward_penalties == {}


def test_penalize_first_match_wins_single_penalty():
    gibberish_filter = _make_gibberish_filter(action="penalize")
    repetition_filter = _make_repetition_filter(window=2, action="penalize")

    # Triggers both gibberish (token 0) and repetition (tokens 1-2)
    rollout = _make_rollout(
        completion_ids=[120_000, 1, 2],
        completion_logprobs=[gibberish_filter.logprob_threshold - 1.0, -0.001, -0.001],
        reward=1.0,
    )

    apply_filters([gibberish_filter, repetition_filter], [rollout])

    assert rollout.reward == -1.0
    assert rollout.filter_results == {"gibberish": True, "repetition": False}
    assert list(rollout.reward_penalties) == ["gibberish"]


def test_repeated_apply_filters_preserves_prior_metadata():
    gibberish_filter = _make_gibberish_filter(action="penalize")
    zero_advantage_filter = ZeroAdvantageFilter(name="zero_advantage", action="drop")

    rollout = _make_dirty_rollout(gibberish_filter, reward=1.0)

    # Phase 1: pre-advantage penalty
    apply_filters([gibberish_filter], [rollout])
    rollout.advantage = 0.0
    # Phase 2: post-advantage drop must not wipe phase-1 results
    apply_filters([zero_advantage_filter], [rollout])

    assert rollout.filter_results == {"gibberish": True, "zero_advantage": True}
    assert rollout.is_filtered is True
    assert rollout.reward == -1.0
    assert rollout.raw_reward == 1.0
    assert "gibberish" in rollout.reward_penalties


# --- ordering tests: penalty visible to advantage computation ---


def test_penalized_reward_changes_advantage():
    gibberish_filter = _make_gibberish_filter(action="penalize")
    advantage_fn = setup_advantage_fn(DefaultAdvantageConfig())

    clean = _make_rollout(completion_ids=[50], completion_logprobs=[-1.0], reward=1.0)
    dirty = _make_dirty_rollout(gibberish_filter, reward=1.0)

    apply_filters([gibberish_filter], [clean, dirty])
    assign_advantages([clean, dirty], advantage_fn)

    # Group rewards are (1.0, -1.0): mean 0 → clean +1, dirty -1
    assert clean.advantage == pytest.approx(1.0)
    assert dirty.advantage == pytest.approx(-1.0)
    assert dirty.advantage < clean.advantage


def test_equally_penalized_group_collapses_to_zero_advantage():
    gibberish_filter = _make_gibberish_filter(action="penalize")
    zero_advantage_filter = ZeroAdvantageFilter(name="zero_advantage", action="drop")
    advantage_fn = setup_advantage_fn(DefaultAdvantageConfig())

    rollouts = [_make_dirty_rollout(gibberish_filter, reward=1.0) for _ in range(2)]

    apply_filters([gibberish_filter], rollouts)
    assign_advantages(rollouts, advantage_fn)
    apply_filters([zero_advantage_filter], rollouts)

    for rollout in rollouts:
        assert rollout.reward == -1.0
        assert rollout.advantage == pytest.approx(0.0)
        assert rollout.filter_results == {"gibberish": True, "zero_advantage": True}
        assert rollout.is_filtered is True
