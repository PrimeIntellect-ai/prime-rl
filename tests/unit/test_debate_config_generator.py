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


def test_generated_single_agent_eval_uses_deepseek_grader() -> None:
    # The grader sampling/provider policy was hoisted out of the per-config TOML
    # into the shared gpqa_oe prompt pack (single source of truth); the eval now
    # references it via ``judge_prompt_pack``. The policy itself is asserted in
    # ``test_debate_prompt_pack_uses_same_deepseek_sampling_policy``.
    for path in sorted(GENERATED.glob("*.toml")):
        config = tomllib.loads(path.read_text())
        (single_eval,) = [env for env in config["orchestrator"]["eval"]["env"] if env["id"] == "hf-singleturn"]
        args = single_eval["args"]

        assert args["judge_prompt_pack"] == "gpqa_oe", path
        assert args["judge_model"] == "deepseek/deepseek-v4-flash", path
        assert args["judge_base_url"] == "https://openrouter.ai/api/v1", path
        assert args["judge_api_key_var"] == "OPENROUTER_API_KEY", path


def test_debate_prompt_pack_uses_same_deepseek_sampling_policy() -> None:
    text = GPQA_OE_JUDGE_PROMPT.read_text()

    assert text.count("model: deepseek/deepseek-v4-flash") == 2
    assert text.count("max_completion_tokens: 8192") == 2
    assert text.count("effort: high") == 2
    assert text.count("exclude: true") == 2
    # ZDR-constrained provider POOL with fallbacks (was: single pinned provider,
    # allow_fallbacks false). zdr:true + data_collection:deny still filter every
    # provider in the pool, so the privacy guarantee holds while a single dead
    # provider no longer fails every grade (the grader_error=1.0 incident).
    assert text.count("allow_fallbacks: true") == 2
    assert text.count("require_parameters: true") == 2
    assert text.count("zdr: true") == 2
    assert text.count("data_collection: deny") == 2
    assert text.count("- AtlasCloud") == 2
