from __future__ import annotations

import pathlib
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[2]
GENERATED = ROOT / "configs" / "debate" / "generated"
GPQA_OE_JUDGE_PROMPT = ROOT / "deps" / "verifiers" / "verifiers" / "utils" / "judge_prompts" / "gpqa_oe.yaml"


def test_generated_debate_envs_wire_gt_grader() -> None:
    for path in sorted(GENERATED.glob("*.toml")):
        config = tomllib.loads(path.read_text())
        train_envs = config["orchestrator"]["train"]["env"]
        eval_envs = config["orchestrator"]["eval"]["env"]
        debate_envs = [env for env in [*train_envs, *eval_envs] if env["id"] == "gpqa-open-ended-debate"]
        assert debate_envs, path

        for env in debate_envs:
            args = env["args"]
            assert args["judge_base_url"] == "https://openrouter.ai/api/v1", path
            assert args["judge_model"] == "deepseek/deepseek-v4-flash", path
            assert args["judge_api_key_var"] == "OPENROUTER_API_KEY", path


def test_generated_single_agent_eval_uses_deepseek_provider_args() -> None:
    for path in sorted(GENERATED.glob("*.toml")):
        config = tomllib.loads(path.read_text())
        (single_eval,) = [env for env in config["orchestrator"]["eval"]["env"] if env["id"] == "hf-singleturn"]
        args = single_eval["args"]
        sampling = args["judge_sampling_args"]
        extra_body = sampling["extra_body"]
        provider = extra_body["provider"]

        assert args["judge_prompt_pack"] == "gpqa_oe", path
        assert args["judge_model"] == "deepseek/deepseek-v4-flash", path
        assert sampling["max_completion_tokens"] == 8192, path
        assert extra_body["reasoning"] == {"effort": "high", "exclude": True}, path
        assert provider["only"] == ["AtlasCloud"], path
        assert provider["allow_fallbacks"] is False, path
        assert provider["require_parameters"] is True, path
        assert provider["zdr"] is True, path
        assert provider["data_collection"] == "deny", path


def test_debate_prompt_pack_uses_same_deepseek_sampling_policy() -> None:
    text = GPQA_OE_JUDGE_PROMPT.read_text()

    assert text.count("model: deepseek/deepseek-v4-flash") == 2
    assert text.count("max_completion_tokens: 8192") == 2
    assert text.count("effort: high") == 2
    assert text.count("exclude: true") == 2
    assert text.count("allow_fallbacks: false") == 2
    assert text.count("require_parameters: true") == 2
    assert text.count("zdr: true") == 2
    assert text.count("data_collection: deny") == 2
    assert text.count("- AtlasCloud") == 2
