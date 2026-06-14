import base64
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.trajectories import interleave_rollout, prune_train_rollout_payload
from prime_rl.orchestrator.types import TrainRollout
from prime_rl.transport import TrainingSample


def test_prune_train_rollout_payload_compacts_heavy_token_arrays():
    raw = {
        "trajectory": [
            {
                "prompt": [{"role": "user", "content": "hello"}],
                "completion": [{"role": "assistant", "content": "world"}],
                "response": object(),
                "tokens": {
                    "prompt_ids": [1, 2, 3],
                    "prompt_mask": [0, 0, 0],
                    "completion_ids": [4, 5],
                    "completion_mask": [1, 1],
                    "completion_logprobs": [-0.1, -0.2],
                    "routed_experts": {
                        "data": "large-base64-payload",
                        "shape": [5, 78, 8],
                        "dtype": "uint8",
                        "start": 0,
                    },
                    "overlong_prompt": False,
                    "is_truncated": False,
                },
            }
        ]
    }

    prune_train_rollout_payload(raw)

    step = raw["trajectory"][0]
    assert "response" not in step
    assert raw["trajectory_payload_pruned"] is True
    assert step["tokens"] == {
        "prompt_ids_len": 3,
        "prompt_mask_len": 3,
        "completion_ids_len": 2,
        "completion_mask_len": 2,
        "completion_logprobs_len": 2,
        "has_routed_experts": True,
        "routed_experts_shape": [5, 78, 8],
        "routed_experts_dtype": "uint8",
        "routed_experts_start": 0,
        "overlong_prompt": False,
        "is_truncated": False,
    }


def test_interleave_rollout_prunes_raw_payload_during_preparation():
    raw = {
        "example_id": 0,
        "error": None,
        "trajectory": [
            {
                "tokens": {
                    "prompt_ids": [1, 2],
                    "prompt_mask": [0, 0],
                    "completion_ids": [3],
                    "completion_mask": [1],
                    "completion_logprobs": [-0.1],
                    "routed_experts": {
                        "data": base64.b64encode(bytes([7, 8])).decode(),
                        "shape": [2, 1, 1],
                        "dtype": "uint8",
                        "start": 0,
                    },
                    "overlong_prompt": False,
                    "is_truncated": False,
                },
                "response": object(),
            }
        ],
    }

    samples = interleave_rollout(raw, env_name="test-env", prune_raw_payload=True)

    assert samples is not None
    assert samples[0].routed_experts is not None
    assert samples[0].routed_experts.shape == [3, 1, 1]
    assert raw["trajectory_payload_pruned"] is True
    step = raw["trajectory"][0]
    assert "response" not in step
    assert step["tokens"]["has_routed_experts"] is True
    assert step["tokens"]["routed_experts_shape"] == [2, 1, 1]
    assert "routed_experts" not in step["tokens"]


@pytest.mark.asyncio
async def test_process_rollout_prunes_raw_payload_immediately(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    raw = {
        "error": None,
        "trajectory": [
            {
                "tokens": {
                    "prompt_ids": [1, 2, 3],
                    "prompt_mask": [0, 0, 0],
                    "completion_ids": [4, 5],
                    "completion_mask": [1, 1],
                    "completion_logprobs": [-0.1, -0.2],
                    "routed_experts": {
                        "data": "large-base64-payload",
                        "shape": [5, 78, 8],
                        "dtype": "uint8",
                        "start": 0,
                    },
                },
            }
        ],
    }
    sample = TrainingSample(
        prompt_ids=[1, 2, 3],
        prompt_mask=[False, False, False],
        completion_ids=[4, 5],
        completion_mask=[True, True],
        completion_logprobs=[-0.1, -0.2],
        completion_temperatures=[],
        env_name="test-env",
    )

    def fake_interleave_rollout(output, *args, prune_raw_payload: bool, **kwargs):
        assert prune_raw_payload is True
        prune_train_rollout_payload(output)
        return [sample]

    monkeypatch.setattr("prime_rl.orchestrator.train_sink.interleave_rollout", fake_interleave_rollout)
    monkeypatch.setattr("prime_rl.orchestrator.train_sink.offload_images_to_disk", lambda *args, **kwargs: 0)

    sink = TrainSink(
        SimpleNamespace(
            output_dir=tmp_path,
            training_mode="rl",
        ),
        tokenizer=None,
        renderer=None,
        train_envs=SimpleNamespace(),
        mm_token_type_ids_mapping=None,
        batch_size=1,
        token_batch_size=None,
        pre_filters=[],
        post_filters=[],
    )
    rollout = TrainRollout(
        raw=raw,
        env_name="test-env",
        example_id=0,
        group_id=uuid.uuid4(),
        policy_version=0,
        off_policy_steps=0,
    )

    await sink.process_rollout(rollout)

    assert rollout.samples == [sample]
    assert raw["trajectory_payload_pruned"] is True
    tokens = raw["trajectory"][0]["tokens"]
    assert "routed_experts" not in tokens
    assert tokens["has_routed_experts"] is True
    assert tokens["routed_experts_shape"] == [5, 78, 8]
