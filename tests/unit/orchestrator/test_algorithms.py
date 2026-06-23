import asyncio
from unittest.mock import MagicMock

import pydantic
import pytest
import verifiers.v1 as vf

from prime_rl.configs.algorithm import AlgorithmConfig, FrozenModelConfig
from prime_rl.orchestrator.algo import EchoAlgorithm, stamp_advantages, stamp_loss_routing
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.orchestrator.types import Rollout, RolloutView
from prime_rl.transport.types import TrainingSample

FROZEN = {"name": "org/ref-model", "base_url": ["http://ref:8001/v1"]}

_ALGO = pydantic.TypeAdapter(AlgorithmConfig)


def _task(idx: int = 0) -> vf.Task:
    return vf.Task(idx=idx, prompt="")


def _build(**kwargs) -> AlgorithmConfig:
    """Validate an algorithm config — ``algo.type`` is the discriminator (the
    bundle IS the algorithm)."""
    return _ALGO.validate_python(kwargs)


def _ref_kind(ref):
    """Collapse a resolved reference to a comparable marker."""
    return "frozen" if isinstance(ref, FrozenModelConfig) else ref


@pytest.mark.parametrize(
    ("algorithm_type", "teacher", "source", "model_ref", "action_loss_type"),
    [
        ("grpo", None, "policy", None, "rl"),
        ("max_rl", None, "policy", None, "rl"),
        ("opd", FROZEN, "policy", "frozen", "ref_kl"),
        ("sft", FROZEN, "frozen", None, "ce"),
        ("opsd", None, "policy", "policy", "ref_kl"),
        ("echo", None, "policy", None, "rl"),
    ],
)
def test_type_defaults_are_the_vetted_algorithms(algorithm_type, teacher, source, model_ref, action_loss_type):
    kwargs = {"type": algorithm_type}
    if teacher is not None:
        kwargs["teacher"] = teacher
    algo = _build(**kwargs)
    assert _ref_kind(algo.sampling.source) == source
    assert algo.type == algorithm_type
    assert _ref_kind(getattr(algo, "model", None)) == model_ref
    assert algo.action_loss_type == action_loss_type


def test_echo_roles_replace_the_default_table():
    algo = _build(type="echo", roles={"user": {"alpha": 0.5}})
    assert algo.type == "echo"
    assert algo.roles.user.alpha == 0.5
    # Setting any role replaces the whole table: the tool default is gone
    assert algo.roles.tool is None


def test_echo_defaults_to_tool_bodies():
    algo = _build(type="echo")
    assert algo.roles.tool.alpha == 0.1
    assert algo.roles.system is None
    assert algo.roles.user is None
    assert algo.roles.assistant is None


def test_echo_roles_require_at_least_one():
    with pytest.raises(ValueError, match="at least one role"):
        _build(type="echo", roles={})


def test_opd_requires_teacher():
    with pytest.raises(ValueError, match="needs a teacher"):
        _build(type="opd")


def test_sft_requires_teacher():
    with pytest.raises(ValueError, match="needs a teacher to sample rollouts from"):
        _build(type="sft")


def test_teacher_folds_into_model():
    algo = _build(type="opd", teacher=FROZEN)
    assert isinstance(algo.model, FrozenModelConfig)
    assert algo.model.name == "org/ref-model"


def test_teacher_without_target_errors():
    with pytest.raises(ValueError, match="references no model"):
        _build(type="grpo", teacher=FROZEN)


def test_teacher_redundant_but_consistent_is_accepted():
    algo = _build(type="opd", teacher=FROZEN, model=FROZEN)
    assert isinstance(algo.model, FrozenModelConfig)


def test_opd_rejects_policy():
    with pytest.raises(ValueError, match="degenerate"):
        _build(type="opd", model="policy")


def test_rl_loss_type_incompatible_with_frozen_sampling():
    with pytest.raises(ValueError, match="sampling.source is a frozen model"):
        _build(type="grpo", sampling={"source": FROZEN})


def _make_sample(ce_weights: list[float] | None = None) -> TrainingSample:
    return TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=[3, 4, 5, 6],
        completion_mask=[True, True, False, True],
        completion_logprobs=[-0.1, -0.2, 0.0, -0.3],
        completion_temperatures=[],
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
    # Action tokens (completion_mask True) feed the ref_kl component; rl is off
    assert sample.rl_weights == [0.0] * 6
    assert sample.ref_kl_weights == [0.0, 0.0] + [1.0, 1.0, 0.0, 1.0]
    assert sample.ce_weights is None


def test_stamp_loss_routing_ce_action():
    sample = _make_sample()
    stamp_loss_routing(sample, "ce")
    assert sample.rl_weights == [0.0] * 6
    assert sample.ce_weights == [0.0, 0.0] + [1.0, 1.0, 0.0, 1.0]
    assert sample.ref_kl_weights is None


def test_stamp_loss_routing_keeps_algorithm_written_ce_stream():
    # Echo writes ce_weights directly at group time (observation at
    # completion index 2, outside completion_mask); rl routing must not
    # clobber it — the rl component still ships no streams (hot path).
    sample = _make_sample(ce_weights=[0.0, 0.0] + [0.0, 0.0, 0.1, 0.0])
    stamp_loss_routing(sample, "rl")
    assert sample.rl_weights is None
    assert sample.ce_weights == [0.0, 0.0] + [0.0, 0.0, 0.1, 0.0]
    assert sample.ref_kl_weights is None


def test_stamp_loss_routing_merges_action_weights_into_ce_stream():
    # A ce-action algorithm that also weighted observation tokens: action
    # tokens merge into the existing stream instead of replacing it.
    sample = _make_sample(ce_weights=[0.0, 0.0] + [0.0, 0.0, 0.1, 0.0])
    stamp_loss_routing(sample, "ce")
    assert sample.rl_weights == [0.0] * 6
    assert sample.ce_weights == [0.0, 0.0] + [1.0, 1.0, 0.1, 1.0]
    assert sample.ref_kl_weights is None


def _make_rollout(
    samples: list[TrainingSample],
    advantages: list[float] | None = None,
) -> Rollout:
    rollout = Rollout[vf.Task](task=_task(), rewards={"reward": 1.0})
    rollout.env_name = "test-env"
    rollout.samples = samples
    rollout.advantages = advantages
    return rollout


def test_stamp_advantages_pads_prompt():
    rollout = _make_rollout([_make_sample()], advantages=[0.5, -0.5, 0.0, 1.0])
    stamp_advantages(rollout)
    # 2 prompt positions padded with 0.0 + 4 completion-aligned advantages
    assert rollout.samples[0].advantages == [0.0, 0.0, 0.5, -0.5, 0.0, 1.0]


def test_stamp_advantages_slices_across_samples():
    samples = [_make_sample(), _make_sample()]
    rollout = _make_rollout(samples, advantages=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    stamp_advantages(rollout)
    assert rollout.samples[0].advantages == [0.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    assert rollout.samples[1].advantages == [0.0, 0.0, 5.0, 6.0, 7.0, 8.0]


def test_stamp_advantages_no_credit_ships_none():
    rollout = _make_rollout([_make_sample()])
    stamp_advantages(rollout)
    assert rollout.samples[0].advantages is None


def test_stamp_advantages_rejects_misaligned():
    rollout = _make_rollout([_make_sample()], advantages=[0.5])
    with pytest.raises(ValueError, match="align"):
        stamp_advantages(rollout)


def _echo_algorithm(roles: dict | None = None) -> EchoAlgorithm:
    kwargs: dict = {"type": "echo"}
    if roles is not None:
        kwargs["roles"] = roles
    return EchoAlgorithm(_build(**kwargs), MagicMock(), MagicMock())


def _node(
    *,
    parent: int | None,
    message: vf.Message,
    token_ids: list[int],
    sampled: bool = False,
) -> vf.MessageNode:
    return vf.MessageNode(
        parent=parent,
        message=message,
        sampled=sampled,
        token_ids=token_ids,
        mask=[sampled] * len(token_ids),
        logprobs=[-0.1] * len(token_ids) if sampled else [],
    )


def _echo_rollout(observation: vf.Message | None = None) -> Rollout:
    nodes = [
        _node(parent=None, message=vf.UserMessage(content="U"), token_ids=[1, 2]),
        _node(parent=0, message=vf.AssistantMessage(content="A"), token_ids=[3, 4], sampled=True),
    ]
    if observation is not None:
        nodes.append(_node(parent=1, message=observation, token_ids=[5, 6]))
        nodes.append(_node(parent=2, message=vf.AssistantMessage(content="B"), token_ids=[7, 8], sampled=True))
    rollout = Rollout[vf.Task](task=_task(), nodes=nodes, rewards={"reward": 1.0})
    rollout.env_name = "test-env"
    rollout.samples = trace_to_samples(rollout, env_name="test-env")
    return rollout


def test_echo_weights_observations_by_role():
    rollout = _echo_rollout(vf.ToolMessage(tool_call_id="call", content="tool output"))
    algo = _echo_algorithm()
    asyncio.run(algo.score_rollout(RolloutView(rollout)))
    sample = rollout.samples[0]
    assert sample.completion_ids == [1, 2, 3, 4, 5, 6, 7, 8]
    assert sample.ce_weights == [0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0]
    assert sample.completion_mask == [False, False, True, True, False, False, True, True]

    rollout = _echo_rollout(vf.UserMessage(content="follow-up"))
    algo = _echo_algorithm(roles={"user": {"alpha": 0.05}})
    asyncio.run(algo.score_rollout(RolloutView(rollout)))
    assert rollout.samples[0].ce_weights == [0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.0, 0.0]


def test_echo_rejects_v0_filter_shape():
    config = _build(type="echo", filter={"import_path": "package.module.filter_fn"})
    with pytest.raises(ValueError, match="filters use the old trajectory-step mask shape"):
        EchoAlgorithm(config, MagicMock(), MagicMock())


def test_trace_to_samples_preserves_branch_sampled_mask():
    rollout = _echo_rollout(vf.ToolMessage(tool_call_id="call", content="tool output"))
    samples = rollout.samples
    assert samples[0].completion_ids == [1, 2, 3, 4, 5, 6, 7, 8]
    assert samples[0].completion_mask == [False, False, True, True, False, False, True, True]
    assert samples[0].ce_weights is None
