import pytest
import torch

from prime_rl.trainer.models import (
    AutoModelForCausalLMPrimeRL,
    GlmMoeDsaConfig,
    Qwen3_5TextConfig,
    Qwen3Config,
)
from prime_rl.trainer.perf import FlopCounter

N = 256  # document length; sparse attention reads 32 tokens
DIMS = dict(
    hidden_size=64, intermediate_size=128, vocab_size=128, num_attention_heads=4, num_key_value_heads=2, pad_token_id=0
)
MOE = dict(moe_intermediate_size=32, num_experts_per_tok=2)
# Softmax attention of 4 heads of qk/v dim 16 over all N keys per query, 6 FLOPs per multiply-add.
FULL = 6 * 4 * (16 + 16) * N * N

MODELS = {
    "qwen3": (Qwen3Config(**DIMS, head_dim=16, num_hidden_layers=2), 2 * FULL),
    # Gated delta rule: 4 value heads, key/value head dim 8; three key * value products per token and head.
    "qwen3_5": (
        Qwen3_5TextConfig(
            **DIMS,
            head_dim=16,
            layer_types=["linear_attention", "full_attention"],
            num_hidden_layers=2,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
        ),
        6 * 3 * 4 * 8 * 8 * N + FULL,
    ),
    # MLA with qk = 16 nope + 8 rope, v = 16, reading index_topk = 32 tokens; the indexer (2 heads of dim 8)
    # scores all N tokens forward only (no_grad) on the "full" layer, and the "shared" layer reuses its indices.
    "glm_moe_dsa": (
        GlmMoeDsaConfig(
            **DIMS,
            **MOE,
            n_routed_experts=4,
            num_hidden_layers=2,
            q_lora_rank=32,
            kv_lora_rank=16,
            qk_nope_head_dim=16,
            qk_rope_head_dim=8,
            v_head_dim=16,
            index_n_heads=2,
            index_head_dim=8,
            index_topk=32,
            indexer_types=["full", "shared"],
            use_index_cache=True,
            dsa_backend="tilelang",
        ),
        2 * 6 * 4 * (24 + 16) * N * 32 + 2 * 2 * 8 * N * N,
    ),
}


@pytest.mark.parametrize("name", MODELS)
def test_attention_flops_match_hand_count(name: str):
    config, attention_flops = MODELS[name]
    with torch.device("meta"):
        model = AutoModelForCausalLMPrimeRL.from_config(config, attn_implementation="flash_attention_3")
    counter = FlopCounter(model)
    assert counter([N]) - N * counter.matmul_flops_per_token == attention_flops > 0


def test_active_params_count_attention_and_mlp_weights():
    with torch.device("meta"):
        model = AutoModelForCausalLMPrimeRL.from_config(MODELS["qwen3"][0], attn_implementation="flash_attention_3")
    # Per layer: q/o 64x64, k/v 64x32, q/k norms 16, MLP 3x64x128, two norms 64; then final norm and lm_head.
    layer = 2 * 64 * 64 + 2 * 64 * 32 + 2 * 16 + 3 * 64 * 128 + 2 * 64
    assert model.num_active_params() == 2 * layer + 64 + 128 * 64
