import pytest
import torch

from prime_rl.trainer.models.laguna import LagunaConfig, LagunaForCausalLM

pytestmark = [pytest.mark.gpu]


def test_laguna_init_buffers_post_meta():
    config = LagunaConfig(
        pad_token_id=0,
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts_per_tok=2,
        num_experts=4,
        sliding_window=None,
        layer_types=["full_attention", "full_attention"],
    )
    with torch.device("meta"):
        model = LagunaForCausalLM(config)
    model.to_empty(device="cuda")

    model.init_buffers_post_meta()

    for name, buffer in model.named_buffers():
        assert torch.isfinite(buffer).all(), f"buffer {name} is not finite after init_buffers_post_meta"
