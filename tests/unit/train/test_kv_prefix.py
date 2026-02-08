import pytest
import torch

from prime_rl.trainer.config import KVPrefixConfig, LoRAConfig, ModelConfig
from prime_rl.trainer.kv_prefix import strip_kv_prefix_from_state_dict
from prime_rl.trainer.rl.config import RLTrainerConfig


def test_model_config_rejects_kv_prefix_with_sdpa():
    with pytest.raises(ValueError, match="requires flash attention"):
        ModelConfig(attn="sdpa", kv_prefix=KVPrefixConfig())


def test_model_config_rejects_kv_prefix_with_cp():
    with pytest.raises(ValueError, match="does not support context parallelism"):
        ModelConfig(cp=2, kv_prefix=KVPrefixConfig())


def test_rl_trainer_config_allows_kv_prefix_multi_run():
    cfg = RLTrainerConfig(
        max_concurrent_runs=4,
        model=ModelConfig(lora=LoRAConfig(), kv_prefix=KVPrefixConfig()),
    )
    assert cfg.max_concurrent_runs == 4


def test_strip_kv_prefix_from_state_dict():
    state_dict = {
        "model.layers.0.self_attn.kv_prefix_key": torch.ones(1),
        "model.layers.0.self_attn.kv_prefix_value": torch.ones(1),
        "model.layers.0.self_attn.q_proj.weight": torch.ones(1),
    }
    stripped = strip_kv_prefix_from_state_dict(state_dict)
    assert "model.layers.0.self_attn.kv_prefix_key" not in stripped
    assert "model.layers.0.self_attn.kv_prefix_value" not in stripped
    assert "model.layers.0.self_attn.q_proj.weight" in stripped
