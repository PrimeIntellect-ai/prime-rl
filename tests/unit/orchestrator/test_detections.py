import math
import uuid

import verifiers.v1 as vf

from prime_rl.configs.orchestrator import GibberishDetectionConfig, RepetitionDetectionConfig
from prime_rl.orchestrator.detections import (
    GibberishDetection,
    RepetitionDetection,
    run_detections,
    setup_detection,
    setup_detections,
)
from prime_rl.orchestrator.types import Rollout


def _assistant_node(token_ids: list[int], logprobs: list[float]) -> vf.MessageNode:
    """An assistant node whose tokens are all model-sampled (the detections read each node's
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
    the detections to inspect each node's sampled tokens / logprobs."""
    if multi_step:
        mid = len(completion_ids) // 2
        nodes = [
            _assistant_node(completion_ids[:mid], completion_logprobs[:mid]),
            _assistant_node(completion_ids[mid:], completion_logprobs[mid:]),
        ]
    else:
        nodes = [_assistant_node(completion_ids, completion_logprobs)]
    rollout = Rollout[vf.TaskData](
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="")),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=nodes,
        rewards={"reward": vf.Reward(score=reward)},
    )
    rollout.env_name = "test"
    rollout.group_id = uuid.uuid4()
    return rollout


def _make_gibberish_detection(vocab_size=128_000, token_id_threshold=100_000, logprob_offset=2.0, enforce=False):
    logprob_threshold = -math.log(vocab_size) - logprob_offset
    return GibberishDetection(
        name="gibberish", token_id_threshold=token_id_threshold, logprob_threshold=logprob_threshold, enforce=enforce
    )


def _make_repetition_detection(window=5, prob_threshold=0.99, enforce=False):
    return RepetitionDetection(
        name="repetition", window=window, logprob_threshold=math.log(prob_threshold), enforce=enforce
    )


# --- GibberishDetection tests ---


def test_gibberish_detects_rare_low_prob_token():
    gibberish_detection = _make_gibberish_detection()

    result = gibberish_detection.check(
        _make_rollout(
            completion_ids=[50, 120_000, 80],
            completion_logprobs=[-1.0, gibberish_detection.logprob_threshold - 1.0, -0.5],
        )
    )
    assert result.detected is True


def test_gibberish_ignores_normal_tokens():
    gibberish_detection = _make_gibberish_detection()

    result = gibberish_detection.check(
        _make_rollout(
            completion_ids=[10, 200, 5000],
            completion_logprobs=[-1.0, -2.0, -3.0],
        )
    )
    assert result.detected is False


def test_gibberish_ignores_high_prob_rare_token():
    gibberish_detection = _make_gibberish_detection()

    result = gibberish_detection.check(
        _make_rollout(
            completion_ids=[120_000],
            completion_logprobs=[-0.5],
        )
    )
    assert result.detected is False


def test_gibberish_works_across_trajectory_steps():
    gibberish_detection = _make_gibberish_detection()

    result = gibberish_detection.check(
        _make_rollout(
            completion_ids=[50, 60, 120_000, 80],
            completion_logprobs=[-1.0, -0.5, gibberish_detection.logprob_threshold - 1.0, -0.5],
            multi_step=True,
        )
    )
    assert result.detected is True


def test_gibberish_aligns_logprobs_under_generation_prompt_scaffold():
    """Regression: the assistant node carries a generation-prompt scaffold (mask=False) and
    suffix-only logprobs, and the gibberish token is the LAST completion token. The old
    per-node ``zip(token_ids, logprobs, mask)`` truncated at len(logprobs) and never examined
    it; reading the aligned branch streams detects it."""
    gibberish_detection = _make_gibberish_detection()

    rollout = Rollout[vf.TaskData](
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="")),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=[_scaffold_assistant_node([50, 80, 120_000], [-1.0, -0.5, gibberish_detection.logprob_threshold - 1.0])],
        rewards={"reward": vf.Reward(score=1.0)},
    )

    result = gibberish_detection.check(rollout)
    assert result.detected is True


# --- RepetitionDetection tests ---


def test_repetition_triggers_after_window():
    repetition_detection = _make_repetition_detection(window=5)

    result = repetition_detection.check(
        _make_rollout(
            completion_ids=list(range(5)),
            completion_logprobs=[-0.001] * 5,
        )
    )
    assert result.detected is True


def test_repetition_no_trigger_below_window():
    repetition_detection = _make_repetition_detection(window=5)

    result = repetition_detection.check(
        _make_rollout(
            completion_ids=list(range(4)),
            completion_logprobs=[-0.001] * 4,
        )
    )
    assert result.detected is False


def test_repetition_resets_on_low_prob():
    repetition_detection = _make_repetition_detection(window=5)

    logprobs = [-0.001] * 3 + [-2.0] + [-0.001] * 3
    result = repetition_detection.check(
        _make_rollout(
            completion_ids=list(range(7)),
            completion_logprobs=logprobs,
        )
    )
    assert result.detected is False


def test_repetition_varied_probs_no_trigger():
    repetition_detection = _make_repetition_detection(window=3)

    result = repetition_detection.check(
        _make_rollout(
            completion_ids=list(range(6)),
            completion_logprobs=[-0.001, -3.0, -0.001, -3.0, -0.001, -3.0],
        )
    )
    assert result.detected is False


# --- setup_detection / setup_detections tests ---


def test_setup_detection_gibberish():
    config = GibberishDetectionConfig(token_id_threshold=100_000, logprob_offset=2.0)
    gibberish_detection = setup_detection(config, vocab_size=128_000)
    assert isinstance(gibberish_detection, GibberishDetection)
    assert gibberish_detection.name == "gibberish"
    assert gibberish_detection.token_id_threshold == 100_000
    assert abs(gibberish_detection.logprob_threshold - (-math.log(128_000) - 2.0)) < 1e-10
    assert gibberish_detection.enforce is False


def test_setup_detection_gibberish_enforce():
    config = GibberishDetectionConfig(enforce=True)
    gibberish_detection = setup_detection(config, vocab_size=128_000)
    assert gibberish_detection.enforce is True


def test_setup_detection_repetition():
    config = RepetitionDetectionConfig(window=3_000, prob_threshold=0.99)
    repetition_detection = setup_detection(config, vocab_size=128_000)
    assert isinstance(repetition_detection, RepetitionDetection)
    assert repetition_detection.name == "repetition"
    assert repetition_detection.window == 3_000
    assert abs(repetition_detection.logprob_threshold - math.log(0.99)) < 1e-10
    assert repetition_detection.enforce is False


def test_setup_detection_repetition_enforce():
    config = RepetitionDetectionConfig(enforce=True)
    repetition_detection = setup_detection(config, vocab_size=128_000)
    assert repetition_detection.enforce is True


def test_setup_detections_multiple():
    configs = [
        GibberishDetectionConfig(),
        RepetitionDetectionConfig(),
    ]
    detections = setup_detections(configs, vocab_size=128_000)
    assert len(detections) == 2
    assert detections[0].name == "gibberish"
    assert detections[1].name == "repetition"


# --- run_detections tests (enforce=True) ---


def test_run_detections_enforced_flags_rollout():
    gibberish_detection = _make_gibberish_detection(enforce=True)

    rollout = _make_rollout(
        completion_ids=[120_000],
        completion_logprobs=[gibberish_detection.logprob_threshold - 1.0],
        reward=1.0,
    )

    run_detections([gibberish_detection], rollout)

    assert rollout.reward == 1.0
    assert rollout.nodes[0].token_ids == [120_000]
    assert rollout.nodes[0].mask == [True]
    assert rollout.stop_condition is None
    assert rollout.detections == {"gibberish": True}
    assert rollout.is_excluded is True


def test_run_detections_preserves_clean_rollouts():
    gibberish_detection = _make_gibberish_detection(enforce=True)

    rollout = _make_rollout(
        completion_ids=[50, 60, 70],
        completion_logprobs=[-1.0, -2.0, -1.5],
        reward=1.0,
    )

    run_detections([gibberish_detection], rollout)

    assert rollout.reward == 1.0
    assert rollout.nodes[0].token_ids == [50, 60, 70]
    assert all(rollout.nodes[0].mask)
    assert rollout.stop_condition is None
    assert rollout.detections == {"gibberish": False}
    assert rollout.is_excluded is False


def test_run_detections_first_detection_wins():
    gibberish_detection = _make_gibberish_detection(enforce=True)
    repetition_detection = _make_repetition_detection(window=2, enforce=True)

    rollout = _make_rollout(
        completion_ids=[120_000, 1, 2],
        completion_logprobs=[gibberish_detection.logprob_threshold - 1.0, -0.001, -0.001],
        reward=1.0,
    )

    run_detections([gibberish_detection, repetition_detection], rollout)

    assert rollout.stop_condition is None
    assert rollout.detections == {"gibberish": True, "repetition": False}
    assert rollout.is_excluded is True


def test_run_detections_empty_list():
    rollout = _make_rollout(
        completion_ids=[1, 2, 3],
        completion_logprobs=[-1.0, -1.0, -1.0],
    )
    run_detections([], rollout)
    assert rollout.detections == {}
    assert rollout.is_excluded is False
    assert rollout.reward == 1.0


def test_run_detections_mixed_batch():
    gibberish_detection = _make_gibberish_detection(enforce=True)

    clean = _make_rollout(completion_ids=[50], completion_logprobs=[-1.0], reward=1.0)
    dirty = _make_rollout(
        completion_ids=[120_000], completion_logprobs=[gibberish_detection.logprob_threshold - 1.0], reward=1.0
    )

    for r in (clean, dirty):
        run_detections([gibberish_detection], r)

    assert clean.reward == 1.0
    assert dirty.reward == 1.0
    assert clean.is_excluded is False
    assert dirty.is_excluded is True


def test_run_detections_enforced_preserves_rollout_tokens():
    gibberish_detection = _make_gibberish_detection(enforce=True)

    rollout = _make_rollout(
        completion_ids=[10, 120_000, 30],
        completion_logprobs=[-1.0, gibberish_detection.logprob_threshold - 1.0, -0.5],
        reward=1.0,
    )

    run_detections([gibberish_detection], rollout)

    assert rollout.nodes[0].token_ids == [10, 120_000, 30]
    assert rollout.nodes[0].logprobs == [
        -1.0,
        gibberish_detection.logprob_threshold - 1.0,
        -0.5,
    ]
    assert rollout.nodes[0].mask == [True, True, True]
    assert rollout.is_excluded is True


def test_run_detections_preserves_existing_stop_condition():
    gibberish_detection = _make_gibberish_detection(enforce=True)

    rollout = _make_rollout(
        completion_ids=[120_000],
        completion_logprobs=[gibberish_detection.logprob_threshold - 1.0],
        reward=1.0,
    )
    rollout.stop_condition = "generation_truncated"

    run_detections([gibberish_detection], rollout)

    assert rollout.stop_condition == "generation_truncated"
    assert rollout.is_excluded is True


# --- run_detections tests (monitor-only, enforce=False) ---


def test_run_detections_monitor_only_tracks_detection():
    gibberish_detection = _make_gibberish_detection(enforce=False)

    rollout = _make_rollout(
        completion_ids=[120_000],
        completion_logprobs=[gibberish_detection.logprob_threshold - 1.0],
        reward=1.0,
    )

    run_detections([gibberish_detection], rollout)

    assert rollout.reward == 1.0
    assert all(rollout.nodes[0].mask)
    assert rollout.stop_condition is None
    assert rollout.detections == {"gibberish": True}
    assert rollout.is_excluded is False


def test_run_detections_monitor_only_mixed_batch():
    gibberish_detection = _make_gibberish_detection(enforce=False)

    clean = _make_rollout(completion_ids=[50], completion_logprobs=[-1.0], reward=1.0)
    dirty = _make_rollout(
        completion_ids=[120_000], completion_logprobs=[gibberish_detection.logprob_threshold - 1.0], reward=1.0
    )

    for r in (clean, dirty):
        run_detections([gibberish_detection], r)

    assert clean.reward == 1.0
    assert dirty.reward == 1.0
    assert clean.is_excluded is False
    assert dirty.is_excluded is False
