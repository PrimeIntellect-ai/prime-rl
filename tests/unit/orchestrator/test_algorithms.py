import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pydantic
import pytest
import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, ToolMessage, UserMessage

from prime_rl.configs.algorithm import AlgoConfig, FrozenModelConfig
from prime_rl.orchestrator.algo import (
    EchoAlgorithm,
    OPSDAlgorithm,
    RLCSDAlgorithm,
    RLRTAlgorithm,
    RLSDAlgorithm,
    stamp_advantages,
    stamp_loss_routing,
)
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.orchestrator.types import Rollout
from prime_rl.transport.types import TrainingSample
from prime_rl.utils.client import PrefillScores

FROZEN = {"name": "org/ref-model", "base_url": ["http://ref:8001/v1"]}

_ALGO = pydantic.TypeAdapter(AlgoConfig)


def _build(**kwargs) -> AlgoConfig:
    """Validate an algorithm config — ``algo.type`` is the discriminator (the
    bundle IS the algorithm)."""
    return _ALGO.validate_python(kwargs)


def _ref_kind(ref):
    """Collapse a resolved reference to a comparable marker."""
    return "frozen" if isinstance(ref, FrozenModelConfig) else ref


# The vetted default of each algorithm: which model it samples from and which
# loss component its action tokens feed. opd alone names a frozen ``teacher``;
# sft samples from a frozen ``sampling.source``; the rest run on the policy.
@pytest.mark.parametrize(
    ("algorithm_type", "build_kwargs", "source", "action_loss_type"),
    [
        ("grpo", {}, "policy", "rl"),
        ("max_rl", {}, "policy", "rl"),
        ("opd", {"teacher": FROZEN}, "policy", "ref_kl"),
        ("sft", {"sampling": {"source": FROZEN}}, "frozen", "ce"),
        ("opsd", {}, "policy", "ref_kl"),
        ("rlsd", {}, "policy", "rl"),
        ("rlrt", {}, "policy", "rl"),
        ("rlcsd", {}, "policy", "rl"),
        ("echo", {}, "policy", "rl"),
    ],
)
def test_type_defaults_are_the_vetted_algorithms(algorithm_type, build_kwargs, source, action_loss_type):
    algo = _build(type=algorithm_type, **build_kwargs)
    assert algo.type == algorithm_type
    assert _ref_kind(algo.sampling.source) == source
    assert algo.action_loss_type == action_loss_type


def test_echo_role_table():
    # Default: tool-response bodies at alpha 0.1, every other role off.
    default = _build(type="echo")
    assert default.roles.tool.alpha == 0.1
    assert default.roles.system is None
    assert default.roles.user is None
    assert default.roles.assistant is None
    # Setting any role replaces the whole table — the tool default is gone.
    replaced = _build(type="echo", roles={"user": {"alpha": 0.5}})
    assert replaced.roles.user.alpha == 0.5
    assert replaced.roles.tool is None


def test_echo_roles_require_at_least_one():
    with pytest.raises(ValueError, match="at least one role"):
        _build(type="echo", roles={})


def test_opd_teacher_must_be_a_frozen_endpoint():
    # opd needs a teacher, and it must be frozen: a missing teacher is a
    # structural error, and "policy" can't even be set — opd.teacher is typed
    # FrozenModelConfig (the KL against the policy itself would be zero).
    with pytest.raises(ValueError, match="Field required"):
        _build(type="opd")
    with pytest.raises(ValueError, match="FrozenModelConfig"):
        _build(type="opd", teacher="policy")


def test_sft_requires_teacher():
    with pytest.raises(ValueError, match="needs a teacher to sample rollouts from"):
        _build(type="sft")


def test_rl_loss_type_incompatible_with_frozen_sampling():
    with pytest.raises(ValueError, match="sampling.source is a frozen model"):
        _build(type="grpo", sampling={"source": FROZEN})


# --------------------------------------------------------------------------
# Routing / advantage stamping over the FLAT TrainingSample data model.
#
# A sample is a single flat token sequence: ``mask`` marks the trainable
# (model-sampled) tokens; the streams (rl/ce/ref_kl/advantages) are all
# full-length-N (= len(token_ids)), 0.0 on non-trainable positions.
# --------------------------------------------------------------------------


def _make_sample(ce_weights: list[float] | None = None) -> TrainingSample:
    # 2 prompt tokens (mask False), then a 4-token completion with one
    # env-provided observation token (position 4, mask False) interleaved.
    return TrainingSample(
        token_ids=[1, 2, 3, 4, 5, 6],
        mask=[False, False, True, True, False, True],
        logprobs=[0.0, 0.0, -0.1, -0.2, 0.0, -0.3],
        temperatures=[],
        env_name="test-env",
        ce_weights=ce_weights,
    )


def test_stamp_loss_routing_uniform_rl():
    sample = _make_sample()
    stamp_loss_routing(sample, "rl")
    # Hot path: absent streams mean rl weight 1.0 on the loss mask
    assert sample.rl_weights is None
    assert sample.ce_weights is None
    assert sample.ref_kl_weights is None


def test_stamp_loss_routing_ref_kl_action():
    sample = _make_sample()
    stamp_loss_routing(sample, "ref_kl")
    # Action tokens (mask True) feed the ref_kl component; rl is off
    assert sample.rl_weights == [0.0] * 6
    assert sample.ref_kl_weights == [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    assert sample.ce_weights is None


def test_stamp_loss_routing_ce_action():
    sample = _make_sample()
    stamp_loss_routing(sample, "ce")
    assert sample.rl_weights == [0.0] * 6
    assert sample.ce_weights == [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    assert sample.ref_kl_weights is None


def test_stamp_loss_routing_keeps_algorithm_written_ce_stream():
    # Echo writes ce_weights directly at group time (observation at position
    # 4, outside the loss mask); rl routing must not clobber it — the rl
    # component still ships no streams (hot path).
    sample = _make_sample(ce_weights=[0.0, 0.0, 0.0, 0.0, 0.1, 0.0])
    stamp_loss_routing(sample, "rl")
    assert sample.rl_weights is None
    assert sample.ce_weights == [0.0, 0.0, 0.0, 0.0, 0.1, 0.0]
    assert sample.ref_kl_weights is None


def test_stamp_loss_routing_merges_action_weights_into_ce_stream():
    # A ce-action algorithm that also weighted observation tokens: action
    # tokens merge into the existing stream instead of replacing it.
    sample = _make_sample(ce_weights=[0.0, 0.0, 0.0, 0.0, 0.1, 0.0])
    stamp_loss_routing(sample, "ce")
    assert sample.rl_weights == [0.0] * 6
    assert sample.ce_weights == [0.0, 0.0, 1.0, 1.0, 0.1, 1.0]
    assert sample.ref_kl_weights is None


def _make_rollout(
    samples: list[TrainingSample],
    advantages: list[float] | None = None,
) -> Rollout:
    rollout = Rollout(task=vf.Task(idx=0, prompt=None), nodes=[], rewards={}, env_name="test-env")
    rollout.samples = samples
    rollout.advantages = advantages
    return rollout


def test_stamp_advantages_full_length_stream():
    # The advantage stream is full-length-N: 0.0 on prompt + non-trainable
    # positions, the rl credit on trainable (mask True) tokens.
    rollout = _make_rollout([_make_sample()], advantages=[0.0, 0.0, 0.5, -0.5, 0.0, 1.0])
    stamp_advantages(rollout)
    assert rollout.samples[0].advantages == [0.0, 0.0, 0.5, -0.5, 0.0, 1.0]


def test_stamp_advantages_slices_across_samples():
    samples = [_make_sample(), _make_sample()]
    rollout = _make_rollout(samples, advantages=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    stamp_advantages(rollout)
    assert rollout.samples[0].advantages == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert rollout.samples[1].advantages == [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]


def test_stamp_advantages_no_credit_ships_none():
    rollout = _make_rollout([_make_sample()])
    stamp_advantages(rollout)
    assert rollout.samples[0].advantages is None


def test_stamp_advantages_rejects_misaligned():
    rollout = _make_rollout([_make_sample()], advantages=[0.5])
    with pytest.raises(ValueError, match="align"):
        stamp_advantages(rollout)


def test_assign_advantages_scalar_broadcasts_over_mask():
    rollout = _make_rollout([_make_sample()])
    rollout.assign_advantages(1.0)
    assert rollout.advantages == [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]


def test_assign_advantages_list_rejects_misaligned():
    rollout = _make_rollout([_make_sample()])
    with pytest.raises(ValueError, match="align"):
        rollout.assign_advantages([0.5])


# --------------------------------------------------------------------------
# Reward-grounded self-distillation variants.
#
# RLSD/RLRT/RLCSD keep the RL loss and rewrite the per-token advantage stream.
# These tests use a fake renderer/pool so no model server is needed.
# --------------------------------------------------------------------------


class _FakeRenderer:
    def render_ids(self, messages, add_generation_prompt=False):
        content = messages[0]["content"]
        if "bad" in content:
            return [202]
        return [101]


class _FakePool:
    model_name = "fake-policy"

    async def score(self, token_ids, top_k=None):
        marker = token_ids[0]
        if marker == 101:
            scores = {11: -1.0, 12: -3.0}
        elif marker == 202:
            scores = {11: -3.0, 12: -1.0}
        else:
            scores = {}
        logprobs = [0.0] + [scores.get(token_id, 0.0) for token_id in token_ids[1:]]
        if top_k is None:
            return logprobs
        return PrefillScores(
            logprobs=logprobs,
            topk_ids=[
                [marker * 1000 + position * 100 + rank for rank in range(top_k)] for position in range(len(token_ids))
            ],
            topk_logprobs=[[-float(rank + 1) for rank in range(top_k)] for _ in token_ids],
        )


class _LongHintRenderer:
    def render_ids(self, messages, add_generation_prompt=False):
        return list(range(10))


def _rl_sample() -> TrainingSample:
    return TrainingSample(
        token_ids=[10, 11, 12],
        mask=[False, True, True],
        logprobs=[0.0, -2.0, -2.0],
        temperatures=[],
        env_name="test-env",
    )


def _rl_rollout(reward: float, *, hint: str = "good hint") -> Rollout:
    rollout = Rollout(task=vf.Task(idx=0, prompt=None), nodes=[], rewards={"reward": reward}, env_name="test-env")
    rollout.samples = [_rl_sample()]
    rollout.info["demonstration"] = hint
    return rollout


def _self_distill_kwargs() -> dict:
    return {
        "token_weight_clip_min": 0.01,
        "token_weight_clip_max": 100.0,
        "normalize_token_weights": True,
    }


def _encoded_array(tensor) -> np.ndarray:
    return np.frombuffer(tensor.data, dtype=tensor.dtype).reshape(tensor.shape)


@pytest.mark.parametrize("algorithm_type", ["rlsd", "rlcsd"])
def test_self_distilled_rl_diagnostic_topk_default(algorithm_type):
    assert _build(type=algorithm_type).diag_top_k == 64


def test_rlsd_collects_measurement_only_teacher_topk_with_truncation_padding():
    rollout = _rl_rollout(1.0)
    algo = RLSDAlgorithm(_build(type="rlsd", diag_top_k=2, max_score_tokens=4), _FakePool())
    algo.scorer.renderer = _FakeRenderer()

    asyncio.run(algo.score_rollout(rollout))

    sample = rollout.samples[0]
    assert sample.ref_logprobs == [0.0, -1.0, 0.0]
    assert sample.mask == [False, True, False]
    assert sample.diag_topk_token_ids.shape == [3, 2]
    assert sample.diag_topk_logprobs.shape == [3, 2]
    np.testing.assert_array_equal(_encoded_array(sample.diag_topk_token_ids)[-1], [0, 0])
    np.testing.assert_array_equal(_encoded_array(sample.diag_topk_logprobs)[-1], [-1e9, -1e9])


def test_rlsd_diagnostics_do_not_change_objective():
    enabled = [_rl_rollout(1.0), _rl_rollout(0.0)]
    disabled = [_rl_rollout(1.0), _rl_rollout(0.0)]

    for diag_top_k, group in ((2, enabled), (None, disabled)):
        algo = RLSDAlgorithm(_build(type="rlsd", diag_top_k=diag_top_k, **_self_distill_kwargs()), _FakePool())
        algo.scorer.renderer = _FakeRenderer()
        for rollout in group:
            asyncio.run(algo.score_rollout(rollout))
        asyncio.run(algo.score_group(group))

    for enabled_rollout, disabled_rollout in zip(enabled, disabled, strict=True):
        assert enabled_rollout.advantages == pytest.approx(disabled_rollout.advantages)
        assert enabled_rollout.samples[0].ref_logprobs == disabled_rollout.samples[0].ref_logprobs
        assert enabled_rollout.samples[0].diag_topk_token_ids is not None
        assert disabled_rollout.samples[0].diag_topk_token_ids is None


def test_rlsd_uses_teacher_student_ratio_as_reward_signed_magnitude():
    good = _rl_rollout(1.0)
    bad = _rl_rollout(0.0)
    algo = RLSDAlgorithm(_build(type="rlsd", **_self_distill_kwargs()), _FakePool())
    algo.scorer.renderer = _FakeRenderer()

    asyncio.run(algo.score_rollout(good))
    asyncio.run(algo.score_rollout(bad))
    asyncio.run(algo.score_group([good, bad]))

    # GRPO scalar is +/-0.5. The good-hint teacher likes token 11 more than
    # token 12, so RLSD gives token 11 larger magnitude while preserving sign.
    assert good.advantages[0] == 0.0
    assert good.advantages[1] > good.advantages[2] > 0.0
    assert bad.advantages[1] < bad.advantages[2] < 0.0
    assert (good.advantages[1] + good.advantages[2]) / 2 == pytest.approx(0.5)


def test_opsd_filters_rollout_when_hint_alone_exceeds_score_window():
    rollout = _rl_rollout(1.0)
    algo = OPSDAlgorithm(_build(type="opsd", max_score_tokens=5), _FakePool())
    algo.renderer = _LongHintRenderer()

    asyncio.run(algo.score_rollout(rollout))

    assert rollout.is_filtered is True
    assert rollout.filter_results["opsd_hint_overflow"] is True
    assert rollout.samples[0].ref_logprobs is None


def test_opsd_tool_sequence_plan_withholds_argument_values():
    rollout = _rl_rollout(1.0, hint='[["search", {"query": "secret", "limit": 3}], ["submit", {}]]')
    algo = OPSDAlgorithm(_build(type="opsd", demo_transform="tool_sequence_plan"), _FakePool())

    plan = algo._demonstration(rollout)

    assert plan == "1. Call `search` with `query`, `limit`.\n2. Call `submit` with no arguments."
    assert "secret" not in plan
    assert "3" not in plan


def test_opsd_tool_sequence_plan_rejects_non_json_demonstration():
    rollout = _rl_rollout(1.0, hint="not json")
    algo = OPSDAlgorithm(_build(type="opsd", demo_transform="tool_sequence_plan"), _FakePool())

    with pytest.raises(ValueError, match="JSON tool-call chain"):
        algo._demonstration(rollout)


def test_opsd_loads_template_path(tmp_path: Path):
    template_path = tmp_path / "gepa-plan.txt"
    template_path.write_text("Optimized plan:\n{demonstration}")

    algo = OPSDAlgorithm(_build(type="opsd", template_path=str(template_path)), _FakePool())

    assert algo.template == "Optimized plan:\n{demonstration}"


def test_opsd_rejects_template_without_demonstration(tmp_path: Path):
    template_path = tmp_path / "bad.txt"
    template_path.write_text("missing placeholder")

    with pytest.raises(ValueError, match="must contain"):
        OPSDAlgorithm(_build(type="opsd", template_path=str(template_path)), _FakePool())


def test_rlsd_filters_rollout_when_hint_alone_exceeds_score_window():
    rollout = _rl_rollout(1.0)
    algo = RLSDAlgorithm(_build(type="rlsd", max_score_tokens=5), _FakePool())
    algo.scorer.renderer = _LongHintRenderer()

    asyncio.run(algo.score_rollout(rollout))

    assert rollout.is_filtered is True
    assert rollout.filter_results["rlsd_hint_overflow"] is True
    assert rollout.samples[0].ref_logprobs == [0.0, 0.0, 0.0]


def test_rlrt_reverses_teacher_signal_only_for_positive_advantage_rollouts():
    good = _rl_rollout(1.0)
    bad = _rl_rollout(0.0)
    algo = RLRTAlgorithm(_build(type="rlrt", **_self_distill_kwargs()), _FakePool())
    algo.scorer.renderer = _FakeRenderer()

    asyncio.run(algo.score_rollout(good))
    asyncio.run(algo.score_rollout(bad))
    asyncio.run(algo.score_group([good, bad]))

    # RLRT upweights successful tokens the student chose against the teacher:
    # here token 12 has lower teacher probability than token 11, so it gets the
    # larger positive advantage. The failed rollout stays plain GRPO.
    assert 0.0 < good.advantages[1] < good.advantages[2]
    assert bad.advantages == [0.0, -0.5, -0.5]


def _assistant_rollout(
    reward: float,
    content: str | None,
    *,
    reasoning_content: str | None = None,
    tool_calls: list[dict] | None = None,
) -> Rollout:
    nodes = [
        _node(UserMessage(content="q"), parent=None, sampled=False, token_ids=[10]),
        _node(
            AssistantMessage(content=content, reasoning_content=reasoning_content, tool_calls=tool_calls),
            parent=0,
            sampled=True,
            token_ids=[11, 12],
            logprobs=[-2.0, -2.0],
        ),
    ]
    rollout = Rollout(task=vf.Task(idx=0, prompt=None), nodes=nodes, rewards={"reward": reward}, env_name="test-env")
    rollout.samples = trace_to_samples(rollout, env_name="test-env")
    return rollout


def test_rlcsd_uses_correct_minus_incorrect_sibling_hint_contrast():
    success = _assistant_rollout(1.0, "good solution")
    failure = _assistant_rollout(0.0, "bad solution")
    other_failure = _assistant_rollout(0.0, "bad alternate")
    algo = RLCSDAlgorithm(_build(type="rlcsd", **_self_distill_kwargs()), _FakePool())
    algo.scorer.renderer = _FakeRenderer()

    asyncio.run(algo.score_group([success, failure, other_failure]))

    # Only one success exists, so the success cannot use a non-self positive
    # hint and stays plain GRPO. It still records positive-teacher diagnostics.
    # Failures get contrastive reweighting from the success minus the other
    # failure.
    assert success.advantages == pytest.approx([0.0, 2 / 3, 2 / 3])
    assert success.samples[0].ref_logprobs is not None
    assert success.samples[0].diag_topk_token_ids is not None
    assert failure.advantages[0] == 0.0
    assert failure.advantages[1] < failure.advantages[2] < 0.0
    assert (failure.advantages[1] + failure.advantages[2]) / 2 == pytest.approx(-1 / 3)


def test_rlcsd_diagnostics_capture_positive_teacher_without_changing_objective():
    enabled = [
        _assistant_rollout(1.0, "good solution"),
        _assistant_rollout(0.0, "bad solution"),
        _assistant_rollout(0.0, "bad alternate"),
    ]
    disabled = [
        _assistant_rollout(1.0, "good solution"),
        _assistant_rollout(0.0, "bad solution"),
        _assistant_rollout(0.0, "bad alternate"),
    ]

    for diag_top_k, group in ((2, enabled), (None, disabled)):
        algo = RLCSDAlgorithm(
            _build(type="rlcsd", diag_top_k=diag_top_k, **_self_distill_kwargs()),
            _FakePool(),
        )
        algo.scorer.renderer = _FakeRenderer()
        asyncio.run(algo.score_group(group))

    for enabled_rollout, disabled_rollout in zip(enabled, disabled, strict=True):
        assert enabled_rollout.advantages == pytest.approx(disabled_rollout.advantages)

    measured = enabled[1].samples[0]
    unmeasured = disabled[1].samples[0]
    assert measured.ref_logprobs == unmeasured.ref_logprobs
    assert measured.diag_topk_token_ids.shape == [3, 2]
    assert measured.diag_topk_logprobs.shape == [3, 2]
    assert unmeasured.diag_topk_token_ids is None
    # The first diagnostic row comes after the positive hint row. A negative
    # teacher write would have a different marker and overwrite these ids.
    np.testing.assert_array_equal(_encoded_array(measured.diag_topk_token_ids)[0], [101100, 101101])


def test_rlcsd_uses_reasoning_and_tool_calls_when_assistant_content_is_none():
    group = [
        _assistant_rollout(
            1.0,
            None,
            reasoning_content="correct plan",
            tool_calls=[{"id": "good", "name": "tools_finish", "arguments": '{"ok": true}'}],
        ),
        _assistant_rollout(
            1.0,
            None,
            reasoning_content="correct alternate",
            tool_calls=[{"id": "good2", "name": "tools_finish", "arguments": '{"ok": true}'}],
        ),
        _assistant_rollout(
            0.0,
            None,
            reasoning_content="wrong plan",
            tool_calls=[{"id": "bad", "name": "tools_finish", "arguments": '{"ok": false}'}],
        ),
        _assistant_rollout(
            0.0,
            None,
            reasoning_content="wrong alternate",
            tool_calls=[{"id": "bad2", "name": "tools_finish", "arguments": '{"ok": false}'}],
        ),
    ]
    algo = RLCSDAlgorithm(_build(type="rlcsd", diag_top_k=2, **_self_distill_kwargs()), _FakePool())
    algo.scorer.renderer = _FakeRenderer()

    asyncio.run(algo.score_group(group))

    for rollout in group:
        sample = rollout.samples[0]
        assert sample.ref_logprobs is not None
        assert sample.diag_topk_token_ids is not None
        assert sample.diag_topk_logprobs is not None


# --------------------------------------------------------------------------
# Echo: weighted CE on env-provided observation tokens of later turns.
#
# Provenance is structural under v1 — within a branch, the non-sampled nodes
# that follow the first sampled (model) node are the env-provided observations
# (tool output, user feedback). Each such node's token span gets its message
# role's weight; the initial prompt (before the first response) is excluded.
# --------------------------------------------------------------------------


def _echo_algorithm(roles: dict | None = None, filter_fn=None) -> EchoAlgorithm:
    kwargs: dict = {"type": "echo"}
    if roles is not None:
        kwargs["roles"] = roles
    algo = EchoAlgorithm(_build(**kwargs), MagicMock())
    algo.filter_fn = filter_fn
    return algo


def _node(message, *, parent, sampled, token_ids, logprobs=None, is_content=None) -> MessageNode:
    return MessageNode(
        parent=parent,
        message=message,
        sampled=sampled,
        token_ids=token_ids,
        mask=[sampled] * len(token_ids),
        is_content=is_content if is_content is not None else [],
        logprobs=logprobs if logprobs is not None else ([0.0] * len(token_ids) if sampled else []),
    )


def _two_turn_rollout(observation_role: str = "tool") -> Rollout:
    """A single linear branch: user prompt, an assistant response, an
    env-provided observation (tool output / user feedback), then a second
    assistant response. Tokens: prompt [1,2], action [3,4], observation
    [5,6], action [7,8]."""
    if observation_role == "tool":
        obs_message = ToolMessage(tool_call_id="t", content="T")
    else:
        obs_message = UserMessage(content="feedback")
    nodes = [
        _node(UserMessage(content="U"), parent=None, sampled=False, token_ids=[1, 2]),
        _node(AssistantMessage(content="A"), parent=0, sampled=True, token_ids=[3, 4], logprobs=[-0.1, -0.2]),
        _node(obs_message, parent=1, sampled=False, token_ids=[5, 6]),
        _node(AssistantMessage(content="A2"), parent=2, sampled=True, token_ids=[7, 8], logprobs=[-0.3, -0.4]),
    ]
    rollout = Rollout(task=vf.Task(idx=0, prompt=None), nodes=nodes, rewards={"r": 1.0}, env_name="test-env")
    rollout.samples = trace_to_samples(rollout, env_name="test-env")
    return rollout


def test_echo_weights_observations_by_role():
    # The observation node [5,6] follows the first sampled node, so it is
    # weighted; the initial prompt [1,2] precedes it and is excluded.
    rollout = _two_turn_rollout()
    algo = _echo_algorithm()  # the default table: tool bodies at 0.1
    asyncio.run(algo.score_rollout(rollout))
    sample = rollout.samples[0]
    assert sample.token_ids == [1, 2, 3, 4, 5, 6, 7, 8]
    assert sample.mask == [False, False, True, True, False, False, True, True]
    # [3,4] step-1 action, [5,6] observation (weighted), [7,8] step-2 action
    assert sample.ce_weights == [0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0]

    # A user-feedback observation under a role table that weights users.
    rollout = _two_turn_rollout(observation_role="user")
    algo = _echo_algorithm(roles={"tool": {"alpha": 0.1}, "user": {"alpha": 0.05}})
    asyncio.run(algo.score_rollout(rollout))
    assert rollout.samples[0].ce_weights == [0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.0, 0.0]

    # A role not in the table leaves the observation unweighted: no ce stream.
    rollout = _two_turn_rollout(observation_role="user")
    algo = _echo_algorithm()  # tool only
    asyncio.run(algo.score_rollout(rollout))
    assert rollout.samples[0].ce_weights is None


def test_echo_weights_only_content_tokens_when_is_content_present():
    # The observation node [5,6] carries per-token is_content: the first token is
    # template scaffold (False), the second is message body (True). Only the body
    # token gets the role weight — the scaffold is excluded (content granularity).
    nodes = [
        _node(UserMessage(content="U"), parent=None, sampled=False, token_ids=[1, 2]),
        _node(AssistantMessage(content="A"), parent=0, sampled=True, token_ids=[3, 4], logprobs=[-0.1, -0.2]),
        _node(
            ToolMessage(tool_call_id="t", content="T"),
            parent=1,
            sampled=False,
            token_ids=[5, 6],
            is_content=[False, True],
        ),
        _node(AssistantMessage(content="A2"), parent=2, sampled=True, token_ids=[7, 8], logprobs=[-0.3, -0.4]),
    ]
    rollout = Rollout(task=vf.Task(idx=0, prompt=None), nodes=nodes, rewards={"r": 1.0}, env_name="test-env")
    rollout.samples = trace_to_samples(rollout, env_name="test-env")
    algo = _echo_algorithm()  # tool bodies at 0.1
    asyncio.run(algo.score_rollout(rollout))
    # Only position 5 (the body token) is weighted; the scaffold token at position 4 is not.
    assert rollout.samples[0].ce_weights == [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]


def test_echo_filter_narrows_selection():
    # A per-branch keep-mask drops observation position 5 (the second tool
    # token), narrowing the role selection.
    def keep_drop_one(trace):
        # One keep-mask per trainable branch, spanning that branch's tokens.
        return [[True, True, True, True, True, False, True, True]]

    rollout = _two_turn_rollout()
    algo = _echo_algorithm(filter_fn=keep_drop_one)
    asyncio.run(algo.score_rollout(rollout))
    assert rollout.samples[0].ce_weights == [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]

    # Shape violations fail loudly: wrong branch count, wrong per-branch length.
    rollout = _two_turn_rollout()
    with pytest.raises(ValueError, match="per trainable branch"):
        asyncio.run(_echo_algorithm(filter_fn=lambda trace: []).score_rollout(rollout))
    rollout = _two_turn_rollout()
    with pytest.raises(ValueError, match="span the branch's tokens"):
        asyncio.run(_echo_algorithm(filter_fn=lambda trace: [[True] * 6]).score_rollout(rollout))
