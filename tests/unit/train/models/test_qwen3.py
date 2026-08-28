import pytest
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from prime_rl.trainer.models.qwen3 import Qwen3ForCausalLM

pytestmark = [pytest.mark.gpu]


def test_qwen3_init_buffers_post_meta():
    config = Qwen3Config(
        pad_token_id=0,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=2,
        vocab_size=64,
    )
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config)
    model.to_empty(device="cuda")

    model.init_buffers_post_meta()

    for name, buffer in model.named_buffers():
        assert torch.isfinite(buffer).all(), f"buffer {name} is not finite after init_buffers_post_meta"
