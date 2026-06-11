import uuid
from unittest.mock import MagicMock

import pytest
import verifiers as vf

from prime_rl.configs.algorithm import AlgorithmConfig, FrozenModelConfig
from prime_rl.orchestrator.algo import spread_token_advantages, stamp_loss_routing
from prime_rl.orchestrator.trajectories import interleave_rollout
from prime_rl.orchestrator.types import TrainRollout
from prime_rl.transport.types import TrainingSample

FROZEN = {"name": "org/ref-model", "base_url": ["http://ref:8001/v1"]}


def _ref_kind(ref):
    """Collapse a resolved reference to a comparable marker."""
    return "frozen" if isinstance(ref, FrozenModelConfig) else ref


@pytest.mark.parametrize(
    ("name", "model", "source", "advantage_type", "advantage_model", "action_loss_type"),
    [
        ("grpo", None, "policy", "group_norm", None, "rl"),
        ("opd", FROZEN, "policy", "ref_kl", "frozen", "ref_kl"),
        ("sft_distill", FROZEN, "frozen", "supervised", None, "ce"),
        ("self_distill", None, "policy", "demo_ref_kl", "policy", "ref_kl"),
        ("echo", None, "policy", "echo", None, "rl"),
    ],
)
def test_preset_expansion(name, model, source, advantage_type, advantage_model, action_loss_type):
    algo = AlgorithmConfig(name=name, model=model)
    assert _ref_kind(algo.sampling.source) == source
    assert algo.advantage.type == advantage_type
    assert _ref_kind(getattr(algo.advantage, "model", None)) == advantage_model
    assert algo.advantage.action_loss_type == action_loss_type


def test_preset_with_component_override_is_rejected():
    with pytest.raises(ValueError, match="presets are atomic"):
        AlgorithmConfig(name="echo", advantage={"observation_weight": 0.5})
    with pytest.raises(ValueError, match="presets are atomic"):
        AlgorithmConfig(name="opd", model=FROZEN, advantage={"max_concurrent": 64})
    with pytest.raises(ValueError, match="presets are atomic"):
        AlgorithmConfig(name="grpo", sampling={"source": "policy"})


def test_assembled_components_without_preset_name():
    algo = AlgorithmConfig(advantage={"type": "echo", "observation_weight": 0.5})
    assert algo.advantage.type == "echo"
    assert algo.advantage.observation_weight == 0.5


def test_ref_kl_requires_model_reference():
    with pytest.raises(ValueError, match="needs a teacher"):
        AlgorithmConfig(name="opd")


def test_frozen_sampling_requires_model_reference():
    with pytest.raises(ValueError, match="needs a teacher to sample rollouts from"):
        AlgorithmConfig(name="sft_distill")


def test_teacher_aliases_model_shorthand():
    algo = AlgorithmConfig.model_validate({"name": "opd", "teacher": FROZEN})
    assert isinstance(algo.advantage.model, FrozenModelConfig)
    assert algo.advantage.model.name == "org/ref-model"


def test_model_shorthand_without_target_errors():
    with pytest.raises(ValueError, match="no component reference accepts it"):
        AlgorithmConfig(name="grpo", model=FROZEN)


def test_model_shorthand_redundant_but_consistent_is_accepted():
    algo = AlgorithmConfig(model=FROZEN, advantage={"type": "ref_kl", "model": FROZEN})
    assert isinstance(algo.advantage.model, FrozenModelConfig)


def test_ref_kl_rejects_policy():
    with pytest.raises(ValueError, match="degenerate"):
        AlgorithmConfig(name="opd", model="policy")


def test_rl_loss_type_incompatible_with_frozen_sampling():
    with pytest.raises(ValueError, match="sampling.source is a frozen model"):
        AlgorithmConfig(sampling={"source": FROZEN}, advantage={"type": "group_norm"})


def _make_sample(obs_mask: list[bool] | None) -> TrainingSample:
    return TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=[3, 4, 5, 6],
        completion_mask=[True, True, False, True],
        completion_logprobs=[-0.1, -0.2, 0.0, -0.3],
        completion_temperatures=[],
        env_name="test-env",
        completion_obs_mask=obs_mask,
    )


def test_stamp_loss_routing_uniform_rl():
    sample = _make_sample(obs_mask=None)
    stamp_loss_routing(sample, "rl", None)
    # Hot path: absent streams mean rl weight 1.0 on the loss mask
    assert sample.rl_weights is None
    assert sample.ce_weights is None
    assert sample.ref_kl_weights is None


def test_stamp_loss_routing_ref_kl_action():
    sample = _make_sample(obs_mask=None)
    stamp_loss_routing(sample, "ref_kl", None)
    # Action tokens (completion_mask True) feed the ref_kl component; rl is off
    assert sample.rl_weights == [0.0] * 6
    assert sample.ref_kl_weights == [0.0, 0.0] + [1.0, 1.0, 0.0, 1.0]
    assert sample.ce_weights is None


def test_stamp_loss_routing_ce_action():
    sample = _make_sample(obs_mask=None)
    stamp_loss_routing(sample, "ce", None)
    assert sample.rl_weights == [0.0] * 6
    assert sample.ce_weights == [0.0, 0.0] + [1.0, 1.0, 0.0, 1.0]
    assert sample.ref_kl_weights is None


def test_stamp_loss_routing_echo_observations():
    # Token at completion index 2 is an env observation (masked out today)
    sample = _make_sample(obs_mask=[False, False, True, False])
    stamp_loss_routing(sample, "rl", 0.1)

    assert sample.completion_obs_mask is None  # cleared, never ships
    # The observation token trains on the ce component with the configured
    # weight; it stays out of completion_mask (the rl mask), so the rl
    # component and its denominator never see it.
    assert sample.completion_mask == [True, True, False, True]
    assert sample.rl_weights is None
    assert sample.ce_weights == [0.0, 0.0] + [0.0, 0.0, 0.1, 0.0]
    assert sample.ref_kl_weights is None


def test_stamp_loss_routing_clears_obs_mask_when_unused():
    sample = _make_sample(obs_mask=[False, False, True, False])
    stamp_loss_routing(sample, "rl", None)
    assert sample.completion_obs_mask is None
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
    rollout = _make_rollout([_make_sample(obs_mask=None)], token_advantages=[0.5, -0.5, 0.0, 1.0])
    spread_token_advantages(rollout)
    # 2 prompt positions padded with 0.0 + 4 completion-aligned advantages
    assert rollout.samples[0].token_advantages == [0.0, 0.0, 0.5, -0.5, 0.0, 1.0]


def test_spread_token_advantages_rejects_misaligned():
    rollout = _make_rollout([_make_sample(obs_mask=None)], token_advantages=[0.5])
    with pytest.raises(ValueError, match="align"):
        spread_token_advantages(rollout)


def test_spread_token_advantages_rejects_multi_sample_rollouts():
    samples = [_make_sample(obs_mask=None), _make_sample(obs_mask=None)]
    rollout = _make_rollout(samples, token_advantages=[0.5, -0.5, 0.0, 1.0])
    with pytest.raises(ValueError, match="exactly one training sample"):
        spread_token_advantages(rollout)


def _two_step_rollout() -> vf.RolloutOutput:
    def step(prompt_ids, completion_ids, logprobs):
        return vf.TrajectoryStep(
            prompt=[{"role": "user", "content": "U"}],
            completion=[{"role": "assistant", "content": "A"}],
            response=MagicMock(),
            tokens=vf.TrajectoryStepTokens(
                prompt_ids=prompt_ids,
                prompt_mask=[0] * len(prompt_ids),
                completion_ids=completion_ids,
                completion_mask=[1] * len(completion_ids),
                completion_logprobs=logprobs,
                overlong_prompt=False,
                is_truncated=False,
            ),
            reward=None,
            advantage=None,
            is_truncated=False,
            trajectory_id="1",
            extras={},
        )

    return vf.RolloutOutput(
        example_id=0,
        trajectory=[
            step([1, 2], [3, 4], [-0.1, -0.2]),
            # Extension: prompt re-includes [1,2,3,4]; tokens [5,6] are the
            # env's observation; [7,8] the next action.
            step([1, 2, 3, 4, 5, 6], [7, 8], [-0.3, -0.4]),
        ],
        error=None,
    )


def test_interleave_tags_observation_tokens():
    samples = interleave_rollout(_two_step_rollout(), env_name="test-env", tag_observation_tokens=True)
    assert samples is not None and len(samples) == 1
    sample = samples[0]
    assert sample.completion_ids == [3, 4, 5, 6, 7, 8]
    # [3,4] step-1 action, [5,6] observation, [7,8] step-2 action
    assert sample.completion_obs_mask == [False, False, True, True, False, False]
    assert sample.completion_mask == [True, True, False, False, True, True]


def test_interleave_obs_mask_off_by_default():
    samples = interleave_rollout(_two_step_rollout(), env_name="test-env")
    assert samples is not None
    assert samples[0].completion_obs_mask is None
