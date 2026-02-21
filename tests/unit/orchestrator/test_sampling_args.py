from prime_rl.orchestrator.config import EvalSamplingConfig, SamplingConfig
from prime_rl.orchestrator.eval_utils import get_eval_sampling_args
from prime_rl.orchestrator.utils import get_sampling_args
from prime_rl.utils.config import ClientConfig


def test_get_sampling_args_enforces_token_outputs_for_non_token_client():
    sampling_config = SamplingConfig(
        min_tokens=4,
        repetition_penalty=1.1,
        extra_body={"custom_flag": True},
    )
    client_config = ClientConfig(client_type="openai_chat_completions")

    sampling_args = get_sampling_args(sampling_config, temperature=0.7, client_config=client_config)

    assert sampling_args["logprobs"] is True
    assert sampling_args["extra_body"]["return_token_ids"] is True
    assert sampling_args["extra_body"]["custom_flag"] is True
    assert sampling_args["temperature"] == 0.7


def test_get_sampling_args_applies_client_overrides():
    sampling_config = SamplingConfig(extra_body={"top_k": 32, "custom": "from-sampling"})
    client_config = ClientConfig(
        sampling_overrides={"seed": 123, "logprobs": False},
        extra_body_overrides={"top_k": 8, "return_token_ids": False, "trace": "enabled"},
    )

    sampling_args = get_sampling_args(sampling_config, temperature=1.0, client_config=client_config)

    assert sampling_args["seed"] == 123
    assert sampling_args["extra_body"]["top_k"] == 8
    assert sampling_args["extra_body"]["trace"] == "enabled"
    assert sampling_args["extra_body"]["custom"] == "from-sampling"
    assert sampling_args["logprobs"] is True
    assert sampling_args["extra_body"]["return_token_ids"] is True


def test_get_eval_sampling_args_applies_client_overrides():
    eval_sampling_config = EvalSamplingConfig(
        top_k=4,
        extra_body={"trace": "from-eval"},
    )
    client_config = ClientConfig(
        sampling_overrides={"seed": 99},
        extra_body_overrides={"trace": "from-client", "return_token_ids": True},
    )

    sampling_args = get_eval_sampling_args(eval_sampling_config, client_config)

    assert sampling_args["seed"] == 99
    assert sampling_args["extra_body"]["top_k"] == 4
    assert sampling_args["extra_body"]["trace"] == "from-client"
    assert sampling_args["extra_body"]["return_token_ids"] is True
