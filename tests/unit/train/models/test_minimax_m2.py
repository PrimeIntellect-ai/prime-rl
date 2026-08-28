import pytest
import torch

from prime_rl.trainer.models.minimax_m2 import MiniMaxM2Config, MiniMaxM2ForCausalLM

pytestmark = [pytest.mark.gpu]


def test_minimax_m2_init_buffers_post_meta():
    config = MiniMaxM2Config(
        pad_token_id=0,
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        num_local_experts=4,
        num_experts_per_tok=2,
        use_grouped_mm=False,
    )
    with torch.device("meta"):
        model = MiniMaxM2ForCausalLM(config)
    model.to_empty(device="cuda")

    model.init_buffers_post_meta()

    for name, buffer in model.named_buffers():
        assert torch.isfinite(buffer).all(), f"buffer {name} is not finite after init_buffers_post_meta"
