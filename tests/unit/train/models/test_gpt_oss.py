import pytest
import torch
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

from prime_rl.trainer.models.gpt_oss import GptOssForCausalLM

pytestmark = [pytest.mark.gpu]


def test_gpt_oss_init_buffers_post_meta():
    config = GptOssConfig(
        pad_token_id=0,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_local_experts=4,
        num_experts_per_tok=2,
        vocab_size=64,
        sliding_window=16,
        layer_types=["sliding_attention", "full_attention"],
    )
    with torch.device("meta"):
        model = GptOssForCausalLM(config)
    model.to_empty(device="cuda")

    model.init_buffers_post_meta()

    for name, buffer in model.named_buffers():
        assert torch.isfinite(buffer).all(), f"buffer {name} is not finite after init_buffers_post_meta"
