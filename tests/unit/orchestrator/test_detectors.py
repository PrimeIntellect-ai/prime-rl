import math

import verifiers.v1 as vf

from prime_rl.configs.orchestrator import DetectorsConfig, GibberishDetectorConfig, RepetitionDetectorConfig
from prime_rl.orchestrator.detectors import (
    GibberishDetector,
    RepetitionDetector,
    detect,
    drop_reasons,
    setup_detectors,
)
from prime_rl.orchestrator.types import TrainRollout


def _assistant_node(token_ids: list[int], logprobs: list[float]) -> vf.MessageNode:
    """An assistant node whose tokens are all model-sampled (the detectors read each node's
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
) -> TrainRollout:
    """Build a ``TrainRollout`` (a message-graph trace) carrying the completion tokens — enough for
    the detectors to inspect each node's sampled tokens / logprobs."""
    if multi_step:
        mid = len(completion_ids) // 2
        nodes = [
            _assistant_node(completion_ids[:mid], completion_logprobs[:mid]),
            _assistant_node(completion_ids[mid:], completion_logprobs[mid:]),
        ]
    else:
        nodes = [_assistant_node(completion_ids, completion_logprobs)]
    rollout = TrainRollout[vf.TaskData](
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="")),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=nodes,
        rewards={"reward": vf.Reward(score=reward)},
    )
    rollout.env_name = "test"
    return rollout


def _make_gibberish_detector(vocab_size=128_000, token_id_threshold=100_000, logprob_offset=2.0):
    return GibberishDetector(
        name="gibberish",
        token_id_threshold=token_id_threshold,
        logprob_threshold=-math.log(vocab_size) - logprob_offset,
    )


def _make_repetition_detector(window=5, prob_threshold=0.99):
    return RepetitionDetector(name="repetition", window=window, logprob_threshold=math.log(prob_threshold))


# --- GibberishDetector ---


def test_gibberish_detects_rare_low_prob_token():
    gibberish = _make_gibberish_detector()

    detected = gibberish.detect(
        _make_rollout(
            completion_ids=[50, 120_000, 80],
            completion_logprobs=[-1.0, gibberish.logprob_threshold - 1.0, -0.5],
        )
    )
    assert detected is True


def test_gibberish_ignores_normal_tokens():
    gibberish = _make_gibberish_detector()

    detected = gibberish.detect(
        _make_rollout(
            completion_ids=[10, 200, 5000],
            completion_logprobs=[-1.0, -2.0, -3.0],
        )
    )
    assert detected is False


def test_gibberish_ignores_high_prob_rare_token():
    gibberish = _make_gibberish_detector()

    detected = gibberish.detect(
        _make_rollout(
            completion_ids=[120_000],
            completion_logprobs=[-0.5],
        )
    )
    assert detected is False


def test_gibberish_works_across_trajectory_steps():
    gibberish = _make_gibberish_detector()

    detected = gibberish.detect(
        _make_rollout(
            completion_ids=[50, 60, 120_000, 80],
            completion_logprobs=[-1.0, -0.5, gibberish.logprob_threshold - 1.0, -0.5],
            multi_step=True,
        )
    )
    assert detected is True


def test_gibberish_aligns_logprobs_under_generation_prompt_scaffold():
    """Regression: the assistant node carries a generation-prompt scaffold (mask=False) and
    suffix-only logprobs, and the gibberish token is the LAST completion token. The old
    per-node ``zip(token_ids, logprobs, mask)`` truncated at len(logprobs) and never examined
    it; reading the aligned branch streams detects it."""
    gibberish = _make_gibberish_detector()

    rollout = TrainRollout[vf.TaskData](
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="")),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=[_scaffold_assistant_node([50, 80, 120_000], [-1.0, -0.5, gibberish.logprob_threshold - 1.0])],
        rewards={"reward": vf.Reward(score=1.0)},
    )

    detected = gibberish.detect(rollout)
    assert detected is True


# --- RepetitionDetector ---


def test_repetition_triggers_after_window():
    repetition = _make_repetition_detector(window=5)

    detected = repetition.detect(
        _make_rollout(
            completion_ids=list(range(5)),
            completion_logprobs=[-0.001] * 5,
        )
    )
    assert detected is True


def test_repetition_no_trigger_below_window():
    repetition = _make_repetition_detector(window=5)

    detected = repetition.detect(
        _make_rollout(
            completion_ids=list(range(4)),
            completion_logprobs=[-0.001] * 4,
        )
    )
    assert detected is False


def test_repetition_resets_on_low_prob():
    repetition = _make_repetition_detector(window=5)

    logprobs = [-0.001] * 3 + [-2.0] + [-0.001] * 3
    detected = repetition.detect(
        _make_rollout(
            completion_ids=list(range(7)),
            completion_logprobs=logprobs,
        )
    )
    assert detected is False


def test_repetition_varied_probs_no_trigger():
    repetition = _make_repetition_detector(window=3)

    detected = repetition.detect(
        _make_rollout(
            completion_ids=list(range(6)),
            completion_logprobs=[-0.001, -3.0, -0.001, -3.0, -0.001, -3.0],
        )
    )
    assert detected is False


# --- setup + drop policy ---


def test_setup_detectors_builds_both_and_derives_thresholds():
    detectors = setup_detectors(DetectorsConfig(), vocab_size=128_000)
    assert [d.name for d in detectors] == ["gibberish", "repetition"]
    gibberish, repetition = detectors
    assert gibberish.token_id_threshold == 100_000
    assert gibberish.logprob_threshold == -math.log(128_000) - 2.0
    assert repetition.logprob_threshold == math.log(0.99)


def test_setup_detectors_can_turn_one_off():
    config = DetectorsConfig(gibberish=None, repetition=RepetitionDetectorConfig(window=7))
    assert [d.name for d in setup_detectors(config, vocab_size=128_000)] == ["repetition"]


def test_setup_detectors_thresholds_follow_config():
    config = DetectorsConfig(gibberish=GibberishDetectorConfig(token_id_threshold=50_000, logprob_offset=1.0))
    gibberish = setup_detectors(config, vocab_size=1_000)[0]
    assert (gibberish.token_id_threshold, gibberish.logprob_threshold) == (50_000, -math.log(1_000) - 1.0)


def test_detect_measures_every_detector():
    """Every detector measures every trace — one hit does not shadow the others, which is what
    makes a detection rate a metric rather than a by-product of the drop order."""
    gibberish = _make_gibberish_detector()
    rollout = _make_rollout(
        completion_ids=[120_000] * 5,
        completion_logprobs=[gibberish.logprob_threshold - 1.0] * 5,
    )
    detect([gibberish, _make_repetition_detector(window=5)], rollout)
    assert rollout.detections == {"gibberish": True, "repetition": False}


def _drop(rollout, drop=(), zero_advantage=True):
    return drop_reasons(rollout, drop_detections=list(drop), drop_zero_advantage=zero_advantage)


def test_detection_only_drops_when_asked():
    rollout = _make_rollout(completion_ids=[1], completion_logprobs=[-1.0])
    rollout.detections = {"gibberish": True}
    rollout.assign_advantages(0.5)
    assert _drop(rollout) == []  # measured, but nothing asked for it to drop
    assert _drop(rollout, drop=["gibberish"]) == ["gibberish"]


def test_zero_credit_drops_by_default():
    rollout = _make_rollout(completion_ids=[1], completion_logprobs=[-1.0])
    rollout.assign_advantages(0.0)
    assert _drop(rollout) == ["zero_advantage"]
    assert _drop(rollout, zero_advantage=False) == []


def test_unscored_rollout_is_not_zero_credit():
    """opd/opsd assign no credit at all and train through reference KL — dropping them as
    zero-advantage would ship an empty batch for every distillation run."""
    rollout = _make_rollout(completion_ids=[1], completion_logprobs=[-1.0])
    assert rollout.advantages is None
    assert _drop(rollout) == []


def test_reasons_accumulate():
    rollout = _make_rollout(completion_ids=[1], completion_logprobs=[-1.0])
    rollout.detections = {"gibberish": True, "repetition": True}
    rollout.assign_advantages(0.0)
    assert _drop(rollout, drop=["gibberish", "repetition"]) == ["gibberish", "repetition", "zero_advantage"]
