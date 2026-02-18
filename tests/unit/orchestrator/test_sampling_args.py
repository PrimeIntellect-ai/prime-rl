from prime_rl.orchestrator.config import SamplingConfig
from prime_rl.orchestrator.utils import get_sampling_args
from prime_rl.orchestrator.vf_utils import with_program_id


def test_get_sampling_args_omits_auto_program_id_field():
    sampling_config = SamplingConfig(
        min_tokens=3,
        repetition_penalty=1.1,
        extra_body={"foo": "bar"},
        auto_program_id=True,
    )

    sampling_args = get_sampling_args(sampling_config, temperature=0.7)

    assert "auto_program_id" not in sampling_args
    assert sampling_args["temperature"] == 0.7
    assert sampling_args["extra_body"]["foo"] == "bar"
    assert sampling_args["extra_body"]["min_tokens"] == 3
    assert sampling_args["extra_body"]["repetition_penalty"] == 1.1


def test_with_program_id_does_not_mutate_input_sampling_args():
    sampling_args = {"temperature": 1.0, "extra_body": {"top_k": -1}}

    updated_args = with_program_id(sampling_args, "program-123")

    assert sampling_args == {"temperature": 1.0, "extra_body": {"top_k": -1}}
    assert updated_args["extra_body"]["top_k"] == -1
    assert updated_args["extra_body"]["program_id"] == "program-123"
