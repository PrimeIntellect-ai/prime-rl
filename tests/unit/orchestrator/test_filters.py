import math
import uuid

import verifiers.v1 as vf

from prime_rl.configs.orchestrator import GibberishFilterConfig, RepetitionFilterConfig
from prime_rl.orchestrator.filters import (
    GibberishFilter,
    RepetitionFilter,
    apply_filters,
    setup_filter,
    setup_filters,
)
from prime_rl.orchestrator.types import Rollout


def _assistant_node(token_ids: list[int], logprobs: list[float]) -> vf.MessageNode:
    """An assistant node whose tokens are all model-sampled (the filters read each node's
    masked-True tokens + logprobs)."""
    return vf.MessageNode(
        message=vf.AssistantMessage(content="x"),
        token_ids=token_ids,
        mask=[True] * len(token_ids),
        logprobs=logprobs,
    )


def _scaffold_assistant_node(
    completion_ids: list[int], completion_logprobs: list[float], *, scaffold: int = 2
) -> vf.MessageNode:
    """A realistic v1 assistant node: a leading generation-prompt scaffold (mask=False, not
    model-sampled) then the sampled completion. ``logprobs`` cover only the completion suffix
    (vLLM returns logprobs for generated tokens only) — the exact layout where per-node
    ``zip(token_ids, logprobs, mask)`` mispairs and the branch streams normalize."""
    return vf.MessageNode(
        message=vf.AssistantMessage(content="x"),
        token_ids=[1] * scaffold + completion_ids,
        mask=[False] * scaffold + [True] * len(completion_ids),
        logprobs=completion_logprobs,
    )


def _make_rollout(
    completion_ids: list[int],
    completion_logprobs: list[float],
    *,
    reward: float = 1.0,
    multi_step: bool = False,
) -> Rollout:
    """Build a ``Rollout`` (a message-graph trace) carrying the completion tokens — enough for
    the filters to inspect each node's sampled tokens / logprobs."""
    if multi_step:
        mid = len(completion_ids) // 2
        nodes = [
            _assistant_node(completion_ids[:mid], completion_logprobs[:mid]),
            _assistant_node(completion_ids[mid:], completion_logprobs[mid:]),
        ]
    else:
        nodes = [_assistant_node(completion_ids, completion_logprobs)]
    rollout = Rollout[vf.TaskData](
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="")), nodes=nodes, rewards={"reward": reward}
    )
    rollout.env_name = "test"
    rollout.group_id = uuid.uuid4()
    return rollout


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


def test_gibberish_ignores_normal_tokens():
    gibberish_filter = _make_gibberish_filter()

    result = gibberish_filter.check(
        _make_rollout(
            completion_ids=[10, 200, 5000],
            completion_logprobs=[-1.0, -2.0, -3.0],
        )
    )
    assert result.detected is False


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


def test_gibberish_aligns_logprobs_under_generation_prompt_scaffold():
    """Regression: the assistant node carries a generation-prompt scaffold (mask=False) and
    suffix-only logprobs, and the gibberish token is the LAST completion token. The old
    per-node ``zip(token_ids, logprobs, mask)`` truncated at len(logprobs) and never examined
    it; reading the aligned branch streams detects it."""
    gibberish_filter = _make_gibberish_filter()

    rollout = Rollout[vf.TaskData](
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="")),
        nodes=[_scaffold_assistant_node([50, 80, 120_000], [-1.0, -0.5, gibberish_filter.logprob_threshold - 1.0])],
        rewards={"reward": 1.0},
    )

    result = gibberish_filter.check(rollout)
    assert result.detected is True


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
    filters = setup_filters(configs, vocab_size=128_000, kind="post-batch")
    assert len(filters) == 2
    assert filters[0].name == "gibberish"
    assert filters[1].name == "repetition"


# --- apply_filters tests (enforce=True) ---


def test_apply_filters_enforced_flags_rollout():
    gibberish_filter = _make_gibberish_filter(enforce=True)

    rollout = _make_rollout(
        completion_ids=[120_000],
        completion_logprobs=[gibberish_filter.logprob_threshold - 1.0],
        reward=1.0,
    )

    apply_filters([gibberish_filter], [rollout])

    assert rollout.reward == 1.0
    assert rollout.nodes[0].token_ids == [120_000]
    assert rollout.nodes[0].mask == [True]
    assert rollout.stop_condition is None
    assert rollout.filter_results == {"gibberish": True}
    assert rollout.is_filtered is True


def test_apply_filters_preserves_clean_rollouts():
    gibberish_filter = _make_gibberish_filter(enforce=True)

    rollout = _make_rollout(
        completion_ids=[50, 60, 70],
        completion_logprobs=[-1.0, -2.0, -1.5],
        reward=1.0,
    )

    apply_filters([gibberish_filter], [rollout])

    assert rollout.reward == 1.0
    assert rollout.nodes[0].token_ids == [50, 60, 70]
    assert all(rollout.nodes[0].mask)
    assert rollout.stop_condition is None
    assert rollout.filter_results == {"gibberish": False}
    assert rollout.is_filtered is False


def test_apply_filters_first_filter_wins():
    gibberish_filter = _make_gibberish_filter(enforce=True)
    repetition_filter = _make_repetition_filter(window=2, enforce=True)

    rollout = _make_rollout(
        completion_ids=[120_000, 1, 2],
        completion_logprobs=[gibberish_filter.logprob_threshold - 1.0, -0.001, -0.001],
        reward=1.0,
    )

    apply_filters([gibberish_filter, repetition_filter], [rollout])

    assert rollout.stop_condition is None
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
    gibberish_filter = _make_gibberish_filter(enforce=True)

    clean = _make_rollout(completion_ids=[50], completion_logprobs=[-1.0], reward=1.0)
    dirty = _make_rollout(
        completion_ids=[120_000], completion_logprobs=[gibberish_filter.logprob_threshold - 1.0], reward=1.0
    )

    apply_filters([gibberish_filter], [clean, dirty])

    assert clean.reward == 1.0
    assert dirty.reward == 1.0
    assert clean.is_filtered is False
    assert dirty.is_filtered is True


def test_apply_filters_enforced_preserves_rollout_tokens():
    gibberish_filter = _make_gibberish_filter(enforce=True)

    rollout = _make_rollout(
        completion_ids=[10, 120_000, 30],
        completion_logprobs=[-1.0, gibberish_filter.logprob_threshold - 1.0, -0.5],
        reward=1.0,
    )

    apply_filters([gibberish_filter], [rollout])

    assert rollout.nodes[0].token_ids == [10, 120_000, 30]
    assert rollout.nodes[0].logprobs == [
        -1.0,
        gibberish_filter.logprob_threshold - 1.0,
        -0.5,
    ]
    assert rollout.nodes[0].mask == [True, True, True]
    assert rollout.is_filtered is True


def test_apply_filters_preserves_existing_stop_condition():
    gibberish_filter = _make_gibberish_filter(enforce=True)

    rollout = _make_rollout(
        completion_ids=[120_000],
        completion_logprobs=[gibberish_filter.logprob_threshold - 1.0],
        reward=1.0,
    )
    rollout.stop_condition = "generation_truncated"

    apply_filters([gibberish_filter], [rollout])

    assert rollout.stop_condition == "generation_truncated"
    assert rollout.is_filtered is True


# --- apply_filters tests (monitor-only, enforce=False) ---


def test_apply_filters_monitor_only_tracks_detection():
    gibberish_filter = _make_gibberish_filter(enforce=False)

    rollout = _make_rollout(
        completion_ids=[120_000],
        completion_logprobs=[gibberish_filter.logprob_threshold - 1.0],
        reward=1.0,
    )

    apply_filters([gibberish_filter], [rollout])

    assert rollout.reward == 1.0
    assert all(rollout.nodes[0].mask)
    assert rollout.stop_condition is None
    assert rollout.filter_results == {"gibberish": True}
    assert rollout.is_filtered is False


def test_apply_filters_monitor_only_mixed_batch():
    gibberish_filter = _make_gibberish_filter(enforce=False)

    clean = _make_rollout(completion_ids=[50], completion_logprobs=[-1.0], reward=1.0)
    dirty = _make_rollout(
        completion_ids=[120_000], completion_logprobs=[gibberish_filter.logprob_threshold - 1.0], reward=1.0
    )

    apply_filters([gibberish_filter], [clean, dirty])

    assert clean.reward == 1.0
    assert dirty.reward == 1.0
    assert clean.is_filtered is False
    assert dirty.is_filtered is False
