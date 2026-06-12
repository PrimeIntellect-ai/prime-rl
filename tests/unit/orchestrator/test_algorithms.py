import uuid
from unittest.mock import MagicMock

import pytest
import verifiers as vf

from prime_rl.configs.algorithm import AlgorithmConfig, FrozenModelConfig
from prime_rl.orchestrator.algo import EchoAlgorithm, spread_token_advantages, stamp_loss_routing
from prime_rl.orchestrator.trajectories import interleave_rollout
from prime_rl.orchestrator.types import TrainRollout
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
        ("rlcsd", None, "policy", "policy", "rl"),
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


def _make_sample(obs_weights: list[float] | None) -> TrainingSample:
    return TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=[3, 4, 5, 6],
        completion_mask=[True, True, False, True],
        completion_logprobs=[-0.1, -0.2, 0.0, -0.3],
        completion_temperatures=[],
        env_name="test-env",
        completion_obs_weights=obs_weights,
    )


def test_stamp_loss_routing_uniform_rl():
    sample = _make_sample(obs_weights=None)
    stamp_loss_routing(sample, "rl")
    # Hot path: absent streams mean rl weight 1.0 on the loss mask
    assert sample.rl_weights is None
    assert sample.ce_weights is None
    assert sample.ref_kl_weights is None


def test_stamp_loss_routing_ref_kl_action():
    sample = _make_sample(obs_weights=None)
    stamp_loss_routing(sample, "ref_kl")
    # Action tokens (completion_mask True) feed the ref_kl component; rl is off
    assert sample.rl_weights == [0.0] * 6
    assert sample.ref_kl_weights == [0.0, 0.0] + [1.0, 1.0, 0.0, 1.0]
    assert sample.ce_weights is None


def test_stamp_loss_routing_ce_action():
    sample = _make_sample(obs_weights=None)
    stamp_loss_routing(sample, "ce")
    assert sample.rl_weights == [0.0] * 6
    assert sample.ce_weights == [0.0, 0.0] + [1.0, 1.0, 0.0, 1.0]
    assert sample.ref_kl_weights is None


def test_stamp_loss_routing_echo_observations():
    # Token at completion index 2 is an env observation (masked out today)
    sample = _make_sample(obs_weights=[0.0, 0.0, 0.1, 0.0])
    stamp_loss_routing(sample, "rl")

    assert sample.completion_obs_weights is None  # cleared, never ships
    # The observation token trains on the ce component with its role's
    # weight; it stays out of completion_mask (the rl mask), so the rl
    # component and its denominator never see it.
    assert sample.completion_mask == [True, True, False, True]
    assert sample.rl_weights is None
    assert sample.ce_weights == [0.0, 0.0] + [0.0, 0.0, 0.1, 0.0]
    assert sample.ref_kl_weights is None


def test_stamp_loss_routing_clears_obs_weights_when_all_zero():
    sample = _make_sample(obs_weights=[0.0, 0.0, 0.0, 0.0])
    stamp_loss_routing(sample, "rl")
    assert sample.completion_obs_weights is None
    assert sample.ce_weights is None
    assert sample.completion_mask == [True, True, False, True]


def _make_rollout(samples: list[TrainingSample], token_advantages: list[float] | None) -> TrainRollout:
    return TrainRollout(
        raw={},
        env_name="test-env",
        example_id=0,
        group_id=uuid.uuid4(),
        policy_version=0,
        off_policy_steps=0,
        samples=samples,
        token_advantages=token_advantages,
    )


def test_spread_token_advantages_pads_prompt():
    rollout = _make_rollout([_make_sample(obs_weights=None)], token_advantages=[0.5, -0.5, 0.0, 1.0])
    spread_token_advantages(rollout)
    # 2 prompt positions padded with 0.0 + 4 completion-aligned advantages
    assert rollout.samples[0].token_advantages == [0.0, 0.0, 0.5, -0.5, 0.0, 1.0]


def test_spread_token_advantages_rejects_misaligned():
    rollout = _make_rollout([_make_sample(obs_weights=None)], token_advantages=[0.5])
    with pytest.raises(ValueError, match="align"):
        spread_token_advantages(rollout)


def test_spread_token_advantages_rejects_multi_sample_rollouts():
    samples = [_make_sample(obs_weights=None), _make_sample(obs_weights=None)]
    rollout = _make_rollout(samples, token_advantages=[0.5, -0.5, 0.0, 1.0])
    with pytest.raises(ValueError, match="exactly one training sample"):
        spread_token_advantages(rollout)


def _echo_algorithm(roles: dict | None = None, filter_fn=None) -> EchoAlgorithm:
    advantage: dict = {"type": "echo"}
    if roles is not None:
        advantage["roles"] = roles
    algo = EchoAlgorithm(AlgorithmConfig(advantage=advantage), MagicMock(), MagicMock())
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
        trajectory=[
            step([1, 2], [3, 4], [-0.1, -0.2], prompt_attribution=first_attribution),
            # Extension: prompt re-includes [1,2,3,4]; tokens [5,6] are the
            # env's observation; [7,8] the next action.
            step([1, 2, 3, 4, 5, 6], [7, 8], [-0.3, -0.4], prompt_attribution=attribution),
        ],
        error=None,
    )


def test_interleave_tags_observation_weights_by_role():
    # Span tokens [5,6] (positions 4,5) belong to a tool message; is_content
    # excludes the wrap token, so only the body token gets the tool weight.
    attribution = {
        "message_indices": [0, 0, 1, 1, 2, 2],
        "message_roles": ["user", "assistant", "tool"],
        "is_content": [True, True, True, True, False, True],
    }
    rollout = _two_step_rollout(attribution)
    algo = _echo_algorithm()  # the default table: tool bodies at 0.1
    samples = algo.build_samples(rollout, env_name="test-env")
    assert samples is not None and len(samples) == 1
    sample = samples[0]
    assert sample.completion_ids == [3, 4, 5, 6, 7, 8]
    # [3,4] step-1 action, [5,6] observation, [7,8] step-2 action
    assert sample.completion_obs_weights == [0.0, 0.0, 0.0, 0.1, 0.0, 0.0]
    assert sample.completion_mask == [True, True, False, False, True, True]

    # Without is_content, whole messages count; each role carries its own weight.
    attribution = {"message_indices": [0, 0, 1, 1, 2, 3], "message_roles": ["user", "assistant", "tool", "user"]}
    rollout = _two_step_rollout(attribution)
    algo = _echo_algorithm(roles={"tool": {"alpha": 0.1}, "user": {"alpha": 0.05}})
    samples = algo.build_samples(rollout, env_name="test-env")
    assert samples is not None
    assert samples[0].completion_obs_weights == [0.0, 0.0, 0.1, 0.05, 0.0, 0.0]

    # MITO rollouts carry no attribution: loud error, not a silent no-op.
    with pytest.raises(ValueError, match="attribution"):
        _echo_algorithm().observation_weights(_two_step_rollout())


def test_interleave_echo_filter_narrows_selection():
    attribution = {"message_indices": [0, 0, 1, 1, 2, 2], "message_roles": ["user", "assistant", "tool"]}
    rollout = _two_step_rollout(attribution)

    def keep_last_only(output):
        # One keep-mask per step over prompt+completion; drops span position 4.
        return [[True] * 4, [True, True, True, True, False, True, True, True]]

    algo = _echo_algorithm(filter_fn=keep_last_only)
    samples = algo.build_samples(rollout, env_name="test-env")
    assert samples is not None
    assert samples[0].completion_obs_weights == [0.0, 0.0, 0.0, 0.1, 0.0, 0.0]

    # Shape violations fail loudly: wrong step count, wrong per-step length.
    with pytest.raises(ValueError, match="per trajectory step"):
        _echo_algorithm(filter_fn=lambda output: [[True] * 4]).observation_weights(rollout)
    with pytest.raises(ValueError, match="prompt\\+completion"):
        _echo_algorithm(filter_fn=lambda output: [[True] * 4, [True] * 6]).observation_weights(rollout)


def test_rlcsd_contrastive_signal_is_log_mean_exp():
    from prime_rl.orchestrator.algo.rlcsd import _contrastive_signal

    # One negative hint: plain logprob difference
    assert _contrastive_signal([-1.0], [[-2.0]])[0] == pytest.approx(1.0)
    # K negatives: the negative branch is the log of the MEAN probability,
    # not the mean logprob
    import math

    expected = -1.0 - math.log((math.exp(-1.0) + math.exp(-3.0)) / 2)
    assert _contrastive_signal([-1.0], [[-1.0], [-3.0]])[0] == pytest.approx(expected)


def test_rlcsd_hint_pools_gap_band():
    from types import SimpleNamespace

    from prime_rl.orchestrator.algo.rlcsd import _hint_pools

    group = [SimpleNamespace(reward=r) for r in (1.0, 0.6, 0.45, 0.1)]
    # gap 0 reproduces the plain threshold split
    correct, wrong = _hint_pools(group, 0.5, 0.0)
    assert [r.reward for r in correct] == [1.0, 0.6]
    assert [r.reward for r in wrong] == [0.45, 0.1]
    # the band [threshold - gap, threshold) serves as neither hint
    correct, wrong = _hint_pools(group, 0.5, 0.3)
    assert [r.reward for r in correct] == [1.0, 0.6]
    assert [r.reward for r in wrong] == [0.1]
    # binary rewards: any gap in (0, 1] reduces to the paper's partition
    binary = [SimpleNamespace(reward=r) for r in (1.0, 0.0, 1.0)]
    correct, wrong = _hint_pools(binary, 1.0, 0.5)
    assert len(correct) == 2 and len(wrong) == 1


def test_rlcsd_modulation_two_path_weights_and_clamp():
    from prime_rl.orchestrator.algo.rlcsd import _modulated_token_advantages

    knobs = dict(lam=0.5, tau=0.02, delta=0.02, eta=1.0)
    # Token 0 carries a saturating contrast (tanh -> 1, r = lam), token 1 none.
    # Paths normalize independently: each path's weight is L / |path|.
    out = _modulated_token_advantages([10.0, 0.0], 1.0, [True, True], **knobs)
    assert out[0] == pytest.approx((1.0 + 0.5) * 2.0)  # modulated path, |M| = 1
    assert out[1] == pytest.approx(1.0 * 2.0)  # plain path, |U| = 1

    # Sign-preserving clamp: modulation never flips the verifier's direction
    out = _modulated_token_advantages([10.0], -0.2, [True], **knobs)
    assert out == [0.0]

    # Below the mask threshold everything stays plain GRPO at unit weight
    assert _modulated_token_advantages([0.0], 1.0, [True], **knobs) == [1.0]

    # No trainable tokens -> no per-token advantages
    assert _modulated_token_advantages([10.0], 1.0, [False], **knobs) is None


def test_interleave_obs_weights_off_by_default():
    samples = interleave_rollout(_two_step_rollout(), env_name="test-env")
    assert samples is not None
    assert samples[0].completion_obs_weights is None
