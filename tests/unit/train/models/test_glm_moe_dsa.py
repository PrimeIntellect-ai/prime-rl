import pytest
import torch

from prime_rl.trainer.models.glm_moe_dsa import GlmMoeDsaConfig, GlmMoeDsaForCausalLM

pytestmark = [pytest.mark.gpu]


def test_glm_moe_dsa_init_buffers_post_meta():
    config = GlmMoeDsaConfig(
        pad_token_id=0,
        hidden_size=32,
        intermediate_size=64,
        max_position_embeddings=128,
        moe_intermediate_size=32,
        norm_topk_prob=True,
        num_attention_heads=4,
        num_key_value_heads=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        num_hidden_layers=2,
        rope_theta=1000000.0,
        first_k_dense_replace=1,
        use_grouped_mm=False,
        vocab_size=64,
    )
    with torch.device("meta"):
        model = GlmMoeDsaForCausalLM(config)
    model.to_empty(device="cuda")

    model.init_buffers_post_meta()

    for name, buffer in model.named_buffers():
        assert torch.isfinite(buffer).all(), f"buffer {name} is not finite after init_buffers_post_meta"
