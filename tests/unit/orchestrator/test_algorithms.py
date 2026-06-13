import uuid
from unittest.mock import MagicMock

import pytest
import verifiers as vf

from prime_rl.configs.algorithm import AlgorithmConfig, FrozenModelConfig
from prime_rl.orchestrator.algo import EchoAlgorithm, stamp_advantages, stamp_loss_routing
from prime_rl.orchestrator.trajectories import interleave_rollout
from prime_rl.orchestrator.types import RolloutView, TrainRollout
from prime_rl.transport.types import TrainingSample

FROZEN = {"name": "org/ref-model", "base_url": ["http://ref:8001/v1"]}


def _ref_kind(ref):
    """Collapse a resolved reference to a comparable marker."""
    return "frozen" if isinstance(ref, FrozenModelConfig) else ref


@pytest.mark.parametrize(
    ("advantage_type", "model", "source", "advantage_model", "action_loss_type"),
    [
        ("grpo", None, "policy", None, "rl"),
        ("max_rl", None, "policy", None, "rl"),
        ("opd", FROZEN, "policy", "frozen", "ref_kl"),
        ("sft", FROZEN, "frozen", None, "ce"),
        ("opsd", None, "policy", "policy", "ref_kl"),
        ("echo", None, "policy", None, "rl"),
    ],
)
def test_type_defaults_are_the_vetted_algorithms(advantage_type, model, source, advantage_model, action_loss_type):
    algo = AlgorithmConfig(advantage={"type": advantage_type}, model=model)
    assert _ref_kind(algo.sampling.source) == source
    assert algo.advantage.type == advantage_type
    assert _ref_kind(getattr(algo.advantage, "model", None)) == advantage_model
    assert algo.advantage.action_loss_type == action_loss_type


def test_echo_roles_replace_the_default_table():
    algo = AlgorithmConfig(advantage={"type": "echo", "roles": {"user": {"alpha": 0.5}}})
    assert algo.advantage.type == "echo"
    assert algo.advantage.roles.user.alpha == 0.5
    # Setting any role replaces the whole table: the tool default is gone
    assert algo.advantage.roles.tool is None


def test_echo_defaults_to_tool_bodies():
    algo = AlgorithmConfig(advantage={"type": "echo"})
    assert algo.advantage.roles.tool.alpha == 0.1
    assert algo.advantage.roles.system is None
    assert algo.advantage.roles.user is None
    assert algo.advantage.roles.assistant is None


def test_echo_roles_require_at_least_one():
    with pytest.raises(ValueError, match="at least one role"):
        AlgorithmConfig(advantage={"type": "echo", "roles": {}})


def test_opd_requires_teacher():
    with pytest.raises(ValueError, match="needs a teacher"):
        AlgorithmConfig(advantage={"type": "opd"})


def test_sft_requires_teacher():
    with pytest.raises(ValueError, match="needs a teacher to sample rollouts from"):
        AlgorithmConfig(advantage={"type": "sft"})


def test_teacher_aliases_model_shorthand():
    algo = AlgorithmConfig.model_validate({"advantage": {"type": "opd"}, "teacher": FROZEN})
    assert isinstance(algo.advantage.model, FrozenModelConfig)
    assert algo.advantage.model.name == "org/ref-model"


def test_model_shorthand_without_target_errors():
    with pytest.raises(ValueError, match="no component reference accepts it"):
        AlgorithmConfig(model=FROZEN)


def test_model_shorthand_redundant_but_consistent_is_accepted():
    algo = AlgorithmConfig(model=FROZEN, advantage={"type": "opd", "model": FROZEN})
    assert isinstance(algo.advantage.model, FrozenModelConfig)


def test_opd_rejects_policy():
    with pytest.raises(ValueError, match="degenerate"):
        AlgorithmConfig(advantage={"type": "opd"}, model="policy")


def test_rl_loss_type_incompatible_with_frozen_sampling():
    with pytest.raises(ValueError, match="sampling.source is a frozen model"):
        AlgorithmConfig(sampling={"source": FROZEN}, advantage={"type": "grpo"})


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
    raw: vf.RolloutOutput | None = None,
) -> TrainRollout:
    return TrainRollout(
        raw=raw if raw is not None else {},
        env_name="test-env",
        example_id=0,
        group_id=uuid.uuid4(),
        policy_version=0,
        off_policy_steps=0,
        samples=samples,
        advantages=advantages,
    )


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


def _echo_algorithm(roles: dict | None = None, filter_fn=None) -> EchoAlgorithm:
    advantage: dict = {"type": "echo"}
    if roles is not None:
        advantage["roles"] = roles
    algo = EchoAlgorithm(AlgorithmConfig(advantage=advantage).advantage, MagicMock(), MagicMock())
    algo.filter_fn = filter_fn
    return algo


def _two_step_rollout(attribution: dict | None = None) -> vf.RolloutOutput:
    def step(prompt_ids, completion_ids, logprobs, prompt_attribution=None):
        tokens = vf.TrajectoryStepTokens(
            prompt_ids=prompt_ids,
            prompt_mask=[0] * len(prompt_ids),
            completion_ids=completion_ids,
            completion_mask=[1] * len(completion_ids),
            completion_logprobs=logprobs,
            overlong_prompt=False,
            is_truncated=False,
        )
        if prompt_attribution is not None:
            tokens["prompt_attribution"] = prompt_attribution
        return vf.TrajectoryStep(
            prompt=[{"role": "user", "content": "U"}],
            completion=[{"role": "assistant", "content": "A"}],
            response=MagicMock(),
            tokens=tokens,
            reward=None,
            advantage=None,
            is_truncated=False,
            trajectory_id="1",
            extras={},
        )

    # Renderer rollouts carry attribution on every step; the first step's
    # prompt never lands as observation tokens, so a minimal one suffices.
    first_attribution = {"message_indices": [0, 0], "message_roles": ["user"]} if attribution is not None else None
    return vf.RolloutOutput(
        example_id=0,
        reward=1.0,
        trajectory=[
            step([1, 2], [3, 4], [-0.1, -0.2], prompt_attribution=first_attribution),
            # Extension: prompt re-includes [1,2,3,4]; tokens [5,6] are the
            # env's observation; [7,8] the next action.
            step([1, 2, 3, 4, 5, 6], [7, 8], [-0.3, -0.4], prompt_attribution=attribution),
        ],
        error=None,
    )


def _echo_rollout(output: vf.RolloutOutput) -> TrainRollout:
    samples = interleave_rollout(output, env_name="test-env")
    assert samples is not None
    return _make_rollout(samples, raw=output)


def test_echo_weights_observations_by_role():
    # Span tokens [5,6] (positions 4,5) belong to a tool message; is_content
    # excludes the wrap token, so only the body token gets the tool weight.
    attribution = {
        "message_indices": [0, 0, 1, 1, 2, 2],
        "message_roles": ["user", "assistant", "tool"],
        "is_content": [True, True, True, True, False, True],
    }
    rollout = _echo_rollout(_two_step_rollout(attribution))
    algo = _echo_algorithm()  # the default table: tool bodies at 0.1
    algo.score_rollout(RolloutView(rollout))
    sample = rollout.samples[0]
    assert sample.completion_ids == [3, 4, 5, 6, 7, 8]
    # [3,4] step-1 action, [5,6] observation, [7,8] step-2 action
    assert sample.ce_weights == [0.0, 0.0] + [0.0, 0.0, 0.0, 0.1, 0.0, 0.0]
    assert sample.completion_mask == [True, True, False, False, True, True]

    # Without is_content, whole messages count; each role carries its own weight.
    attribution = {"message_indices": [0, 0, 1, 1, 2, 3], "message_roles": ["user", "assistant", "tool", "user"]}
    rollout = _echo_rollout(_two_step_rollout(attribution))
    algo = _echo_algorithm(roles={"tool": {"alpha": 0.1}, "user": {"alpha": 0.05}})
    algo.score_rollout(RolloutView(rollout))
    assert rollout.samples[0].ce_weights == [0.0, 0.0] + [0.0, 0.0, 0.1, 0.05, 0.0, 0.0]

    # MITO rollouts carry no attribution: loud error, not a silent no-op.
    with pytest.raises(ValueError, match="attribution"):
        _echo_algorithm().score_rollout(RolloutView(_echo_rollout(_two_step_rollout())))


def test_echo_filter_narrows_selection():
    attribution = {"message_indices": [0, 0, 1, 1, 2, 2], "message_roles": ["user", "assistant", "tool"]}

    def keep_last_only(output):
        # One keep-mask per step over prompt+completion; drops span position 4.
        return [[True] * 4, [True, True, True, True, False, True, True, True]]

    rollout = _echo_rollout(_two_step_rollout(attribution))
    algo = _echo_algorithm(filter_fn=keep_last_only)
    algo.score_rollout(RolloutView(rollout))
    assert rollout.samples[0].ce_weights == [0.0, 0.0] + [0.0, 0.0, 0.0, 0.1, 0.0, 0.0]

    # Shape violations fail loudly: wrong step count, wrong per-step length.
    rollout = _echo_rollout(_two_step_rollout(attribution))
    with pytest.raises(ValueError, match="per trajectory step"):
        _echo_algorithm(filter_fn=lambda output: [[True] * 4]).score_rollout(RolloutView(rollout))
    with pytest.raises(ValueError, match="prompt\\+completion"):
        _echo_algorithm(filter_fn=lambda output: [[True] * 4, [True] * 6]).score_rollout(RolloutView(rollout))


def test_interleave_records_obs_spans():
    samples = interleave_rollout(_two_step_rollout(), env_name="test-env")
    assert samples is not None
    # The step-2 prompt extension [5,6] lands at completion positions 2-3,
    # sourced from step 1's prompt at offset 4, length 2.
    assert samples[0].obs_spans == [[2, 1, 4, 2]]
    # Provenance only — no algorithm wrote a ce stream.
    assert samples[0].ce_weights is None
