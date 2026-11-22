import json
import shlex

from glyphbench import eval_defaults

EXPECTED_SAMPLING_ARGS = (
    '{"top_p":1.0,"presence_penalty":0.0,"frequency_penalty":0.0,'
    '"extra_body":{"top_k":0,"min_p":0.0,"repetition_penalty":1.0,'
    '"chat_template_kwargs":{"enable_thinking":true}}}'
)


def test_eval_defaults_match_submitted_profile() -> None:
    assert eval_defaults.DEFAULT_MODEL == "Qwen/Qwen3.5-4B"
    assert eval_defaults.DEFAULT_PROVIDER == "vllm"
    assert eval_defaults.DEFAULT_EPISODES_PER_ENV == 3
    assert eval_defaults.DEFAULT_ROLLOUTS_PER_EXAMPLE == 1
    assert eval_defaults.DEFAULT_SEED == 42
    assert eval_defaults.DEFAULT_N_FRAMES == 0
    assert eval_defaults.DEFAULT_MAX_TOKENS == 4096
    assert eval_defaults.DEFAULT_MAX_OUTPUT_TOKENS == 4096
    assert eval_defaults.DEFAULT_MEMORY_UPDATE_MAX_TOKENS == 4096
    assert eval_defaults.DEFAULT_CONTEXT_WINDOW_TOKENS == 65536
    assert eval_defaults.DEFAULT_MAX_MODEL_LEN == 65536
    assert eval_defaults.DEFAULT_MAX_NUM_BATCHED_TOKENS == 65536
    assert eval_defaults.DEFAULT_MAX_CONCURRENT is None
    assert eval_defaults.DEFAULT_MAX_NUM_SEQS is None
    assert eval_defaults.protocol_dict()["context_window_tokens"] == 65536
    craftax_settings = eval_defaults.eval_settings_for_task_ids(
        ["glyphbench/craftax-craft-ironset-v0"]
    )
    assert craftax_settings["max_tokens"] == 4096
    assert craftax_settings["max_output_tokens"] == 4096
    assert craftax_settings["memory_update_max_tokens"] == 4096
    assert craftax_settings["max_model_len"] == 65536
    assert craftax_settings["max_num_batched_tokens"] == 65536
    assert craftax_settings["max_concurrent"] == 1
    assert craftax_settings["max_num_seqs"] == 1
    assert craftax_settings["env_request_timeout_seconds"] is None
    full_settings = eval_defaults.eval_settings_for_task_ids(
        ["glyphbench/craftaxfull-v0"]
    )
    assert full_settings["max_tokens"] == 32768
    assert full_settings["max_output_tokens"] == 32768
    assert full_settings["memory_update_max_tokens"] == 32768
    assert full_settings["max_model_len"] == 196608
    assert full_settings["max_num_batched_tokens"] == 32768
    assert full_settings["max_concurrent"] == 1
    assert full_settings["max_num_seqs"] == 1
    assert full_settings["env_request_timeout_seconds"] == 172800
    assert eval_defaults.DEFAULT_TEMPERATURE == 1.0
    assert eval_defaults.sampling_args_json() == EXPECTED_SAMPLING_ARGS


def test_eval_defaults_shell_assignments_round_trip() -> None:
    parsed: dict[str, str] = {}
    for line in eval_defaults.shell_assignments().splitlines():
        key, value = line.split("=", 1)
        parsed[key] = shlex.split(value)[0]

    assert parsed["GLYPHBENCH_EVAL_MODEL"] == eval_defaults.DEFAULT_MODEL
    assert parsed["GLYPHBENCH_EVAL_SAMPLING_ARGS"] == EXPECTED_SAMPLING_ARGS
    assert json.loads(parsed["GLYPHBENCH_EVAL_EXCLUDE_SUITES"]) == [
        "atari",
        "craftaxfull",
        "nethack",
        "agentick",
    ]
