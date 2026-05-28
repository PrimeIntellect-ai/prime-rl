import pytest

from prime_rl.orchestrator.orchestrator import collect_training_unit_samples
from prime_rl.transport import TrainingSample


def _sample(prompt_len: int, completion_mask: list[bool]) -> TrainingSample:
    return TrainingSample(
        prompt_ids=list(range(prompt_len)),
        prompt_mask=[False] * prompt_len,
        completion_ids=list(range(len(completion_mask))),
        completion_mask=completion_mask,
        completion_logprobs=[0.0] * len(completion_mask),
        completion_temperatures=[1.0] * len(completion_mask),
        env_name="unset",
    )


def test_collect_training_unit_samples_uses_units_for_training_and_rollouts_for_metrics():
    units = [
        {
            "env_name": "debate",
            "reward": 0.8,
            "advantage": 0.3,
            "is_filtered": False,
        },
        {
            "env_name": "debate",
            "reward": 0.2,
            "advantage": -0.1,
            "is_filtered": True,
        },
    ]
    unit_results = [
        [_sample(prompt_len=3, completion_mask=[True, True])],
        [_sample(prompt_len=5, completion_mask=[True])],
    ]

    batch = collect_training_unit_samples(
        training_units=units,
        unit_results=unit_results,
        rollout_to_unit_idxs=[[0, 1]],
        num_rollouts=1,
        training_mode="rl",
    )

    assert len(batch.examples) == 1
    assert batch.examples[0].reward == 0.8
    assert batch.examples[0].advantage == 0.3
    assert batch.examples[0].env_name == "debate"
    assert batch.examples[0].training_mode == "rl"
    assert batch.rollout_prefill_lens == [8]
    assert batch.rollout_decode_lens == [3]
    assert batch.rollout_samples_per_rollout == [2]
    assert batch.num_prefill_tokens == 8
    assert batch.num_decode_tokens == 3


def test_collect_training_unit_samples_fails_on_duplicate_mapping():
    with pytest.raises(ValueError, match="multiple rollout rows"):
        collect_training_unit_samples(
            training_units=[
                {
                    "env_name": "debate",
                    "reward": 1.0,
                    "advantage": 0.5,
                    "is_filtered": False,
                }
            ],
            unit_results=[[]],
            rollout_to_unit_idxs=[[0], [0]],
            num_rollouts=2,
            training_mode="rl",
        )
