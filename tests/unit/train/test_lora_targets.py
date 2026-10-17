import pytest
import torch

from prime_rl.configs.trainer import LoRAConfig
from prime_rl.trainer.lora import _find_target_modules
from prime_rl.trainer.models.glm_moe_dsa import GlmMoeDsaConfig, GlmMoeDsaForCausalLM
from prime_rl.trainer.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

GLM_MOE_DSA = GlmMoeDsaConfig(
    vocab_size=256,
    pad_token_id=0,
    hidden_size=128,
    intermediate_size=256,
    moe_intermediate_size=64,
    num_hidden_layers=2,
    first_k_dense_replace=1,
    num_attention_heads=4,
    num_key_value_heads=4,
    n_routed_experts=4,
    num_experts_per_tok=2,
    q_lora_rank=64,
    kv_lora_rank=32,
    qk_nope_head_dim=32,
    qk_rope_head_dim=16,
    v_head_dim=32,
    index_n_heads=2,
    index_head_dim=32,
)

QWEN3_5 = Qwen3_5TextConfig(
    vocab_size=256,
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=2,
    layer_types=["linear_attention", "full_attention"],
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    linear_key_head_dim=32,
    linear_value_head_dim=32,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
)


@pytest.mark.parametrize(
    ("model_cls", "config", "expected"),
    [
        (
            GlmMoeDsaForCausalLM,
            GLM_MOE_DSA,
            {"q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj", "gate_proj", "up_proj", "down_proj", "experts"},
        ),
        (
            Qwen3_5ForCausalLM,
            QWEN3_5,
            {"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj"}
            | {"gate_proj", "up_proj", "down_proj"},
        ),
    ],
    ids=["glm_moe_dsa", "qwen3_5"],
)
def test_default_lora_targets_cover_attention(model_cls, config, expected):
    with torch.device("meta"):
        model = model_cls(config)
    targets = _find_target_modules(model, LoRAConfig().target_modules)
    assert {name.rsplit(".", 1)[-1] for name in targets} == expected
