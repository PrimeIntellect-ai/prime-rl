import math

import verifiers.v1 as vf
from verifiers.v1.configs.agent import WireAgentConfig

from prime_rl.orchestrator.degeneracy import (
    REPETITION_PROB,
    REPETITION_WINDOW,
    TOKEN_ID_THRESHOLD,
    drop_reasons,
    is_gibberish,
    is_repetitive,
    measure,
)
from prime_rl.orchestrator.types import TrainRollout

VOCAB_SIZE = 128_000
GIBBERISH_LOGPROB = -math.log(VOCAB_SIZE) - 2.0 - 1.0  # comfortably under the entropy threshold
RARE_TOKEN = TOKEN_ID_THRESHOLD + 1
REPEAT_LOGPROB = math.log(REPETITION_PROB) + 0.001  # just above the confidence threshold


def _assistant_node(token_ids: list[int], logprobs: list[float]) -> vf.MessageNode:
    """An assistant node whose tokens are all model-sampled (the measurements read each node's
    masked-True tokens + logprobs)."""
    return vf.MessageNode(
        message=vf.AssistantMessage(content="x"),
        token_ids=token_ids,
        mask=[True] * len(token_ids),
        logprobs=logprobs,
    )


def _make_rollout(nodes: list[vf.MessageNode]) -> TrainRollout:
    rollout = TrainRollout[vf.TaskData](
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="")),
        agent=vf.AgentInfo(config=WireAgentConfig()),
        nodes=nodes,
        rewards={"reward": vf.Reward(score=1.0)},
    )
    rollout.env_name = "test"
    return rollout


def _sampled(token_ids: list[int], logprobs: list[float], *, multi_step: bool = False) -> TrainRollout:
    """A rollout carrying these completion tokens, optionally split across two model turns."""
    if not multi_step:
        return _make_rollout([_assistant_node(token_ids, logprobs)])
    mid = len(token_ids) // 2
    return _make_rollout(
        [
            _assistant_node(token_ids[:mid], logprobs[:mid]),
            _assistant_node(token_ids[mid:], logprobs[mid:]),
        ]
    )


# --- gibberish ---


def test_gibberish_detects_rare_low_prob_token():
    assert is_gibberish(_sampled([50, RARE_TOKEN, 80], [-1.0, GIBBERISH_LOGPROB, -0.5]), VOCAB_SIZE)


def test_gibberish_ignores_normal_tokens():
    assert not is_gibberish(_sampled([10, 200, 5000], [-1.0, -2.0, -3.0]), VOCAB_SIZE)


def test_gibberish_ignores_high_prob_rare_token():
    assert not is_gibberish(_sampled([RARE_TOKEN], [-0.5]), VOCAB_SIZE)


def test_gibberish_works_across_trajectory_steps():
    rollout = _sampled([50, 60, RARE_TOKEN, 80], [-1.0, -0.5, GIBBERISH_LOGPROB, -0.5], multi_step=True)
    assert is_gibberish(rollout, VOCAB_SIZE)


def test_gibberish_aligns_logprobs_under_generation_prompt_scaffold():
    """Regression: the assistant node carries a generation-prompt scaffold (mask=False) and
    suffix-only logprobs, and the gibberish token is the LAST completion token. A per-node
    ``zip(token_ids, logprobs, mask)`` truncates at len(logprobs) and never examines it; reading
    the aligned branch streams finds it."""
    node = vf.MessageNode(
        message=vf.AssistantMessage(content="x"),
        token_ids=[1, 1, 50, 80, RARE_TOKEN],
        mask=[False, False, True, True, True],
        logprobs=[-1.0, -0.5, GIBBERISH_LOGPROB],  # the sampled suffix only, as vLLM returns it
    )
    assert is_gibberish(_make_rollout([node]), VOCAB_SIZE)


# --- repetition ---


def test_repetition_triggers_at_the_window():
    n = REPETITION_WINDOW
    assert is_repetitive(_sampled(list(range(n)), [REPEAT_LOGPROB] * n))


def test_repetition_no_trigger_below_the_window():
    n = REPETITION_WINDOW - 1
    assert not is_repetitive(_sampled(list(range(n)), [REPEAT_LOGPROB] * n))


def test_repetition_resets_on_a_low_probability_token():
    """The streak has to be consecutive: one unconfident token in the middle breaks it, so nearly
    twice the window's worth of confident tokens either side of it is not a loop."""
    half = [REPEAT_LOGPROB] * (REPETITION_WINDOW - 1)
    logprobs = [*half, -2.0, *half]
    assert not is_repetitive(_sampled(list(range(len(logprobs))), logprobs))


def test_repetition_does_not_run_a_streak_across_branches():
    """Each branch is measured on its own — a flat walk over ``nodes`` would join two turns'
    streaks into one that never happened."""
    half = [REPEAT_LOGPROB] * (REPETITION_WINDOW - 1)
    logprobs = [*half, *half]
    assert not is_repetitive(_sampled(list(range(len(logprobs))), logprobs, multi_step=True))


# --- measuring is unconditional ---


def test_measure_records_every_measurement():
    """Every trace is measured for all of them, so a rate is a rate: one hit does not shadow the
    others, and nothing decided in advance which traces to look at."""
    rollout = _sampled([RARE_TOKEN] * 5, [GIBBERISH_LOGPROB] * 5)
    measure(rollout, VOCAB_SIZE)
    assert rollout.degeneracy == {"gibberish": True, "repetition": False}


def test_measure_handles_a_trace_with_no_sampled_tokens():
    rollout = _make_rollout([])
    measure(rollout, VOCAB_SIZE)
    assert rollout.degeneracy == {"gibberish": False, "repetition": False}


# --- the drop policy ---


def _drop(rollout, drop=(), zero_advantage=True):
    return drop_reasons(rollout, drop=list(drop), drop_zero_advantage=zero_advantage)


def test_a_measurement_only_drops_when_asked():
    rollout = _sampled([1], [-1.0])
    rollout.degeneracy = {"gibberish": True}
    rollout.assign_advantages(0.5)
    assert _drop(rollout) == []  # measured, but nothing asked for it to drop
    assert _drop(rollout, drop=["gibberish"]) == ["gibberish"]


def test_zero_credit_drops_by_default():
    rollout = _sampled([1], [-1.0])
    rollout.assign_advantages(0.0)
    assert _drop(rollout) == ["zero_advantage"]
    assert _drop(rollout, zero_advantage=False) == []


def test_unscored_rollout_is_not_zero_credit():
    """opd/opsd assign no credit at all and train through reference KL — dropping them as
    zero-advantage would ship an empty batch for every distillation run."""
    rollout = _sampled([1], [-1.0])
    assert rollout.advantages is None
    assert _drop(rollout) == []


def test_reasons_accumulate():
    rollout = _sampled([1], [-1.0])
    rollout.degeneracy = {"gibberish": True, "repetition": True}
    rollout.assign_advantages(0.0)
    assert _drop(rollout, drop=["gibberish", "repetition"]) == ["gibberish", "repetition", "zero_advantage"]
