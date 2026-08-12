import uuid

import verifiers.v1 as vf

from prime_rl.orchestrator.filters import (
    REPETITION_WINDOW,
    detect_gibberish,
    detect_repetition,
    gibberish_logprob_threshold,
    has_zero_advantage,
)
from prime_rl.orchestrator.types import Rollout

GIBBERISH_THRESHOLD = gibberish_logprob_threshold(vocab_size=128_000)


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
) -> Rollout:
    """Build a ``Rollout`` (a message-graph trace) carrying the completion tokens — enough for
    the detectors to inspect each node's sampled tokens / logprobs."""
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


# --- detect_gibberish tests ---


def test_gibberish_detects_rare_low_prob_token():
    rollout = _make_rollout(
        completion_ids=[50, 120_000, 80],
        completion_logprobs=[-1.0, GIBBERISH_THRESHOLD - 1.0, -0.5],
    )
    assert detect_gibberish(rollout, GIBBERISH_THRESHOLD) is True


def test_gibberish_ignores_normal_tokens():
    rollout = _make_rollout(
        completion_ids=[10, 200, 5000],
        completion_logprobs=[-1.0, -2.0, -3.0],
    )
    assert detect_gibberish(rollout, GIBBERISH_THRESHOLD) is False


def test_gibberish_ignores_high_prob_rare_token():
    rollout = _make_rollout(
        completion_ids=[120_000],
        completion_logprobs=[-0.5],
    )
    assert detect_gibberish(rollout, GIBBERISH_THRESHOLD) is False


def test_gibberish_works_across_trajectory_steps():
    rollout = _make_rollout(
        completion_ids=[50, 60, 120_000, 80],
        completion_logprobs=[-1.0, -0.5, GIBBERISH_THRESHOLD - 1.0, -0.5],
        multi_step=True,
    )
    assert detect_gibberish(rollout, GIBBERISH_THRESHOLD) is True


def test_gibberish_aligns_logprobs_under_generation_prompt_scaffold():
    """Regression: the assistant node carries a generation-prompt scaffold (mask=False) and
    suffix-only logprobs, and the gibberish token is the LAST completion token. The old
    per-node ``zip(token_ids, logprobs, mask)`` truncated at len(logprobs) and never examined
    it; reading the aligned branch streams detects it."""
    rollout = Rollout[vf.TaskData](
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="")),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=[_scaffold_assistant_node([50, 80, 120_000], [-1.0, -0.5, GIBBERISH_THRESHOLD - 1.0])],
        rewards={"reward": vf.Reward(score=1.0)},
    )
    assert detect_gibberish(rollout, GIBBERISH_THRESHOLD) is True


# --- detect_repetition tests ---


def test_repetition_triggers_after_window():
    rollout = _make_rollout(
        completion_ids=list(range(REPETITION_WINDOW)),
        completion_logprobs=[-0.001] * REPETITION_WINDOW,
    )
    assert detect_repetition(rollout) is True


def test_repetition_no_trigger_below_window():
    rollout = _make_rollout(
        completion_ids=list(range(REPETITION_WINDOW - 1)),
        completion_logprobs=[-0.001] * (REPETITION_WINDOW - 1),
    )
    assert detect_repetition(rollout) is False


def test_repetition_resets_on_low_prob():
    logprobs = [-0.001] * (REPETITION_WINDOW - 1) + [-2.0] + [-0.001] * (REPETITION_WINDOW - 1)
    rollout = _make_rollout(
        completion_ids=list(range(len(logprobs))),
        completion_logprobs=logprobs,
    )
    assert detect_repetition(rollout) is False


# --- has_zero_advantage tests ---


def test_zero_advantage_without_advantages():
    rollout = _make_rollout(completion_ids=[1, 2], completion_logprobs=[-1.0, -1.0])
    assert has_zero_advantage(rollout) is False


def test_zero_advantage_all_zero():
    rollout = _make_rollout(completion_ids=[1, 2], completion_logprobs=[-1.0, -1.0])
    rollout.advantages = [0.0, 0.0]
    assert has_zero_advantage(rollout) is True


def test_zero_advantage_nonzero():
    rollout = _make_rollout(completion_ids=[1, 2], completion_logprobs=[-1.0, -1.0])
    rollout.advantages = [0.5, 0.0]
    assert has_zero_advantage(rollout) is False
