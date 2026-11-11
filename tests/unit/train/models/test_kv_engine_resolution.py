from types import SimpleNamespace

import pytest
import torch
from torch import nn

from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.model import configure_kv_precision, resolve_engine_kv_cache
from prime_rl.trainer.models.glm_moe_dsa.sparse_mla_attention import GlmMoeDsaAttention, SparseMlaAttentionArgs


@pytest.mark.parametrize(
    "engines, model_type, kv_scheme, expected",
    [
        (None, "qwen3_moe", False, None),  # no engines (SFT)
        ("auto", "qwen3_moe", False, "auto"),
        ("auto", "deepseek_v4", False, "fp8_ds_mla"),
        ("auto", "llama", True, "fp8"),  # a checkpoint kv_cache_scheme turns auto into FP8
        ("fp8_e4m3", "glm_moe_dsa", False, "fp8_e4m3"),
        ("bfloat16", "glm_moe_dsa", False, "bfloat16"),
    ],
)
def test_engine_kv_cache_resolves_like_vllm_auto(engines, model_type, kv_scheme, expected):
    assert resolve_engine_kv_cache(engines, model_type, kv_scheme) == expected


@pytest.mark.parametrize("model_type, choice", [("glm_moe_dsa", "fp8_e4m3"), ("deepseek_v41", "nvfp4_ds_mla")])
def test_gpu_dependent_auto_must_be_explicit(model_type, choice):
    with pytest.raises(ValueError, match=f"depends on the engines' GPU.*{choice}"):
        resolve_engine_kv_cache("auto", model_type, False)


def glm_attention() -> GlmMoeDsaAttention:
    args = SparseMlaAttentionArgs(
        hidden_size=64,
        num_attention_heads=2,
        kv_lora_rank=32,
        q_lora_rank=16,
        qk_rope_head_dim=8,
        qk_nope_head_dim=16,
        qk_head_dim=24,
        v_head_dim=16,
        attention_bias=False,
        rms_norm_eps=1e-6,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=4,
        dsa_backend="tilelang",
    )
    return GlmMoeDsaAttention(args)


def test_glm_dsa_rounds_query_and_kv_like_the_fp8_mla_cache():
    torch.manual_seed(0)
    model = nn.Module()
    model.config = SimpleNamespace(model_type="glm_moe_dsa")
    model.attn = glm_attention().to(torch.bfloat16)
    config = ModelConfig(engine_kv_cache_dtype="fp8_e4m3")
    configure_kv_precision(model, config)
    assert model.attn.kv_precision == "fp8"

    hidden = torch.randn(1, 6, 64, dtype=torch.bfloat16)
    q_latent, k_normed, k_rope = model.attn.mla_latents(hidden)
    cos, sin = torch.ones(1, 6, 8, dtype=torch.bfloat16), torch.zeros(1, 6, 8, dtype=torch.bfloat16)
    rounded_q, rounded_kv, _ = model.attn.mla_up_proj(q_latent, k_normed, k_rope, (cos, sin))
    model.attn.kv_precision = "bf16"
    q, kv, _ = model.attn.mla_up_proj(q_latent, k_normed, k_rope, (cos, sin))
    # E4M3 at scale 1.0, saturating, on the query and on the latent + RoPE KV (values included)
    for exact, rounded in ((q, rounded_q), (kv, rounded_kv)):
        reference = exact.float().clamp(-448, 448).to(torch.float8_e4m3fn).float().to(exact.dtype)
        assert torch.equal(rounded, reference)
        assert not torch.equal(rounded, exact)

    with pytest.raises(ValueError, match="depends on the engines' GPU"):
        configure_kv_precision(model, ModelConfig(engine_kv_cache_dtype="auto"))
