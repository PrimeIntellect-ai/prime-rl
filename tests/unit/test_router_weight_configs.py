import os

import pytest
from pydantic import ValidationError

from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.inference.server import setup_vllm_env


def full_trainer(**updates):
    config = {
        "enable_router_replay": True,
        "router_replay_mode": "ids_and_weights",
        "model": {"freeze_moe_router": True, "ep": 1, "cp": 1, "impl": "custom"},
    }
    config.update(updates)
    return config


def test_legacy_replay_defaults_unchanged():
    config = TrainerConfig(enable_router_replay=True)
    assert config.router_replay_mode == "ids"
    assert not config.model.freeze_moe_router
    assert not InferenceConfig().vllm.enable_return_routed_expert_weights


def test_full_replay_config():
    config = TrainerConfig.model_validate(full_trainer())
    assert config.router_replay_mode == "ids_and_weights"
    assert config.model.freeze_moe_router


@pytest.mark.parametrize(
    "override, message",
    [
        ({"enable_router_replay": False}, "enable_router_replay"),
        ({"model": {"freeze_moe_router": False, "ep": 1}}, "freeze_moe_router"),
        ({"model": {"freeze_moe_router": True, "ep": 2}}, "model.cp=1"),
        ({"model": {"freeze_moe_router": True, "ep": 1, "cp": 2}}, "model.cp=1"),
        ({"model": {"freeze_moe_router": True, "ep": 1, "impl": "hf"}}, "custom"),
    ],
)
def test_full_replay_rejects_unsupported_config(override, message):
    with pytest.raises(ValidationError, match=message):
        TrainerConfig.model_validate(full_trainer(**override))


def test_managed_full_replay_enables_paired_capture():
    config = RLConfig.model_validate(
        {
            "model": {"name": "Qwen/Qwen3-30B-A3B"},
            "trainer": full_trainer(),
            "orchestrator": {},
            "inference": {},
        }
    )
    assert config.inference.vllm.enable_return_routed_experts
    assert config.inference.vllm.enable_return_routed_expert_weights


def test_managed_ids_only_does_not_enable_weights():
    config = RLConfig.model_validate(
        {
            "trainer": {"enable_router_replay": True},
            "orchestrator": {},
            "inference": {},
        }
    )
    assert config.inference.vllm.enable_return_routed_experts
    assert not config.inference.vllm.enable_return_routed_expert_weights


def test_standalone_full_capture_requires_ids():
    with pytest.raises(ValidationError, match="enable_return_routed_experts"):
        InferenceConfig(vllm={"enable_return_routed_expert_weights": True})


def test_weights_require_v2_and_reject_override(monkeypatch):
    config = InferenceConfig(
        vllm={
            "enable_return_routed_experts": True,
            "enable_return_routed_expert_weights": True,
        }
    )
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    setup_vllm_env(config)
    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    with pytest.raises(ValueError, match="V2"):
        setup_vllm_env(config)


def test_full_capture_rejects_pd_transfer():
    with pytest.raises(ValidationError, match="disaggregated"):
        InferenceConfig(
            use_pd_kv_transfer=True,
            vllm={"enable_return_routed_experts": True, "enable_return_routed_expert_weights": True},
        )


def test_full_capture_guard_rechecked_after_parent_auto_configuration():
    # Parent config auto-enables fields after nested validation; the guard must
    # still execute when the previously valid inference config has KV offload.
    inference = InferenceConfig().model_copy(update={"kv_cache_offload": object()})
    inference.vllm.enable_return_routed_experts = True
    inference.vllm.enable_return_routed_expert_weights = True
    with pytest.raises(ValueError, match="offload"):
        inference.validate_router_weight_capture()
