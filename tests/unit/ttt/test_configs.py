"""The TTT config TOMLs parse and launch: service configs validate against
TTTServiceConfig (top-level keys must not hide under [optim]), and the scaleswe base
config's vLLM LoRA sizing covers the TTT service's adapters."""

import tomllib
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from prime_rl.configs.ttt import TTTServiceConfig  # noqa: E402

CONFIGS = Path(__file__).parents[3] / "configs" / "ttt"


def load(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


@pytest.mark.parametrize("name", ["ttt.toml", "scaleswe/ttt_service.toml"])
def test_service_configs_validate(name: str):
    TTTServiceConfig.model_validate(load(CONFIGS / name))


def test_scaleswe_lora_sizing():
    base = load(CONFIGS / "scaleswe" / "base.toml")["inference"]
    service = load(CONFIGS / "scaleswe" / "ttt_service.toml")
    # vLLM requires max_cpu_loras >= max_loras (its default of 100 would abort the launch).
    assert base["max_cpu_loras"] >= base["max_loras"]
    # The engine can only serve adapters whose rank fits max_lora_rank.
    assert base["max_lora_rank"] >= service["lora"]["rank"]


# --- RLConfig.validate_ttt launch hardening ---

from prime_rl.configs.rl import RLConfig  # noqa: E402
from prime_rl.utils.config import cli  # noqa: E402


def rl_payload(ttt: dict | None, **inference_overrides) -> dict:
    """Minimal RLConfig payload with one (legacy-id) train env carrying a ``ttt`` block."""
    env: dict = {"id": "dummy-env"}
    if ttt is not None:
        env["ttt"] = ttt
    inference = {"enable_lora": True, "max_loras": 16, "max_cpu_loras": 16, **inference_overrides}
    return {
        "model": {"name": "Qwen/Qwen3-0.6B"},
        "trainer": {},
        "orchestrator": {"batch_size": 16, "group_size": 1, "train": {"env": [env]}},
        "inference": inference,
    }


def test_disabled_ttt_imposes_no_constraints():
    """enabled=false is the master switch: no LoRA-serving requirement, no ttt_replay."""
    config = RLConfig.model_validate(
        rl_payload({"base_url": "http://localhost:8092", "enabled": False}, enable_lora=False)
    )
    assert config.trainer.ttt_replay is False


def test_active_ttt_sets_ttt_replay():
    config = RLConfig.model_validate(rl_payload({"base_url": "http://localhost:8092"}))
    assert config.trainer.ttt_replay is True


def test_recycle_and_meta_lessons_mutually_exclusive():
    """A4 (recycle_to_policy) and A5 (meta_lessons) must stay disentangled."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        RLConfig.model_validate(
            rl_payload({"base_url": "http://localhost:8092", "qa": {"recycle_to_policy": True, "meta_lessons": True}})
        )


def test_inflight_exceeding_max_loras_rejected():
    """Every in-flight rollout holds a vLLM adapter slot — inflight must fit max_loras."""
    with pytest.raises(ValueError, match="max_loras"):
        RLConfig.model_validate(rl_payload({"base_url": "http://localhost:8092"}, max_loras=8))


def test_inflight_equal_to_max_loras_ok():
    # batch_size=16 resolves max_inflight_rollouts=16 == max_loras.
    RLConfig.model_validate(rl_payload({"base_url": "http://localhost:8092"}, max_loras=16))


def eval_only_ttt_payload(**inference_overrides) -> dict:
    """RLConfig payload where only the EVAL env carries a ``ttt`` block."""
    payload = rl_payload(None, **inference_overrides)
    payload["orchestrator"]["eval"] = {
        "env": [{"id": "dummy-eval-env", "ttt": {"base_url": "http://localhost:8092"}}]
    }
    return payload


def test_eval_env_ttt_requires_enable_lora():
    """Eval envs run the identical TTT inference regime, so they impose the same
    LoRA-serving requirement on inference as train envs."""
    with pytest.raises(ValueError, match="enable_lora"):
        RLConfig.model_validate(eval_only_ttt_payload(enable_lora=False))


def test_eval_only_ttt_does_not_set_ttt_replay():
    """Eval adapters are dismissed per rollout, never replayed — no trainer replay hooks."""
    config = RLConfig.model_validate(eval_only_ttt_payload())
    assert config.trainer.ttt_replay is False


@pytest.mark.parametrize(
    "arm",
    [
        "arm_a0_no_compaction.toml",
        "arm_a1_compaction.toml",
        "arm_a2_ttt.toml",
        "arm_a3_qa.toml",
        "arm_a4_recycle.toml",
        "arm_a5_meta.toml",
    ],
)
def test_scaleswe_arms_launchable(arm: str):
    """Pins the invariant that every shipped base+arm overlay still validates end-to-end."""
    base = CONFIGS / "scaleswe" / "base.toml"
    cli(RLConfig, args=["@", str(base), "@", str(CONFIGS / "scaleswe" / arm)])
