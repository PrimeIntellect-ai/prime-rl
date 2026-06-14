from prime_rl.orchestrator.train_sink import prune_train_rollout_payload


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
