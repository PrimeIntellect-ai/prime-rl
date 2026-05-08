from prime_rl.orchestrator.vf_utils import get_model_completion_len


def test_get_model_completion_len_uses_final_output_tokens():
    output = {
        "trajectory": [{"tokens": None}],
        "token_usage": {
            "output_tokens": 11,
            "final_output_tokens": 7,
        },
    }

    assert get_model_completion_len(output) == 7


def test_get_model_completion_len_falls_back_to_trajectory_tokens():
    output = {
        "trajectory": [
            {"tokens": {"completion_ids": [1, 2, 3]}},
            {"tokens": {"completion_ids": [4, 5]}},
        ],
    }

    assert get_model_completion_len(output) == 5
