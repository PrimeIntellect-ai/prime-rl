from unittest.mock import MagicMock

import pytest
import verifiers as vf

from prime_rl.configs.algorithm import AlgorithmConfig, LossRoutingConfig
from prime_rl.orchestrator.algorithms import stamp_loss_routing
from prime_rl.orchestrator.trajectories import interleave_rollout
from prime_rl.transport.types import LOSS_CORE_CE, LOSS_CORE_REF_KL, LOSS_CORE_RL, TrainingSample


@pytest.mark.parametrize(
    ("name", "model", "source", "advantage_type", "advantage_model", "action_core", "observation"),
    [
        ("grpo", None, "policy", "group_norm", None, "rl", "none"),
        ("opd", "ref", "policy", "ref_kl", "ref", "ref_kl", "none"),
        ("sft_distill", "ref", "ref", "supervised", None, "ce", "none"),
        ("self_distill", "policy", "policy", "demo_ref_kl", "policy", "ref_kl", "none"),
        ("echo", None, "policy", "group_norm", None, "rl", "ce"),
    ],
)
def test_preset_expansion(name, model, source, advantage_type, advantage_model, action_core, observation):
    algo = AlgorithmConfig(name=name, model=model)
    assert algo.sampling.source == source
    assert algo.advantage.type == advantage_type
    assert getattr(algo.advantage, "model", None) == advantage_model
    assert algo.advantage.action_core == action_core
    assert algo.loss.observation == observation


def test_preset_component_override():
    algo = AlgorithmConfig(name="echo", loss={"observation_weight": 0.5})
    assert algo.loss.observation == "ce"  # unset loss fields inherit from the preset
    assert algo.loss.observation_weight == 0.5
    assert algo.advantage.type == "group_norm"  # untouched components still inherit the preset


def test_ref_kl_requires_model_reference():
    with pytest.raises(ValueError, match="needs a reference model"):
        AlgorithmConfig(name="opd")


def test_frozen_sampling_requires_model_reference():
    with pytest.raises(ValueError, match="samples rollouts from a frozen model"):
        AlgorithmConfig(name="sft_distill")


def test_model_shorthand_without_target_errors():
    with pytest.raises(ValueError, match="no component needs it"):
        AlgorithmConfig(name="grpo", model="ref")


def test_ref_kl_rejects_policy():
    with pytest.raises(ValueError, match="degenerate"):
        AlgorithmConfig(name="opd", model="policy")


def test_rl_core_incompatible_with_frozen_sampling():
    with pytest.raises(ValueError, match="sampling.source='ref'"):
        AlgorithmConfig(name="sft_distill", model="ref", advantage={"type": "group_norm"})


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
    stamp_loss_routing(sample, LOSS_CORE_RL, LossRoutingConfig())
    assert sample.loss_core == LOSS_CORE_RL
    assert sample.token_loss_cores is None
    assert sample.token_loss_weights is None


def test_stamp_loss_routing_primary_core():
    sample = _make_sample(obs_mask=None)
    stamp_loss_routing(sample, LOSS_CORE_REF_KL, LossRoutingConfig())
    assert sample.loss_core == LOSS_CORE_REF_KL
    assert sample.token_loss_cores is None


def test_stamp_loss_routing_echo_observations():
    # Token at completion index 2 is an env observation (masked out today)
    sample = _make_sample(obs_mask=[False, False, True, False])
    stamp_loss_routing(sample, LOSS_CORE_RL, LossRoutingConfig(observation="ce", observation_weight=0.1))

    assert sample.completion_obs_mask is None  # cleared, never ships
    assert sample.loss_core == LOSS_CORE_RL
    # Observation token flips trainable on the CE core with weight lambda
    assert sample.completion_mask == [True, True, True, True]
    # Full-sequence arrays: 2 prompt tokens + 4 completion tokens
    assert sample.token_loss_cores == [LOSS_CORE_RL, LOSS_CORE_RL] + [
        LOSS_CORE_RL,
        LOSS_CORE_RL,
        LOSS_CORE_CE,
        LOSS_CORE_RL,
    ]
    assert sample.token_loss_weights == [1.0, 1.0] + [1.0, 1.0, 0.1, 1.0]


def test_stamp_loss_routing_clears_obs_mask_when_unused():
    sample = _make_sample(obs_mask=[False, False, True, False])
    stamp_loss_routing(sample, LOSS_CORE_RL, LossRoutingConfig(observation="none"))
    assert sample.completion_obs_mask is None
    assert sample.token_loss_cores is None
    assert sample.completion_mask == [True, True, False, True]


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
