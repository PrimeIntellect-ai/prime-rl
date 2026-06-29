import asyncio
from unittest.mock import MagicMock

import pydantic
import pytest
import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, ToolMessage, UserMessage

from prime_rl.configs.algorithm import AlgoConfig, FrozenModelConfig
from prime_rl.orchestrator.algo import EchoAlgorithm, stamp_advantages, stamp_loss_routing
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.orchestrator.types import Rollout
from prime_rl.transport.types import TrainingSample

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
