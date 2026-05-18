import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from prime_rl.trainer.models.qwen3 import Qwen3ForCausalLM


def _tiny_qwen3_config() -> Qwen3Config:
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        hidden_act="silu",
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        attention_bias=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_cache=False,
        tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000},
    )
    config._attn_implementation = "sdpa"
    config.layer_types = ["full_attention"]
    return config


def test_qwen3_forward_with_unwrapped_lm_head():
    model = Qwen3ForCausalLM(_tiny_qwen3_config())
    input_ids = torch.randint(0, model.config.vocab_size, (1, 4))

    outputs = model(input_ids=input_ids)

    assert outputs["logits"].shape == (1, 4, model.config.vocab_size)
