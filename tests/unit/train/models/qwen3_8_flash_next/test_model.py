import pytest
import torch

from prime_rl.trainer.models.conversion_ops import apply_hf_to_prime, apply_prime_to_hf
from prime_rl.trainer.models.qwen3_8_flash_next import (
    Qwen3_8FlashNextConfig,
    Qwen3_8FlashNextForCausalLM,
    Qwen3_8FlashNextTextConfig,
    Qwen3_8FlashNextVisionConfig,
)
from prime_rl.utils.vlm import get_language_model, get_vision_encoder


@pytest.fixture(params=[False, True], ids=["text", "composite"])
def config(request):
    text = Qwen3_8FlashNextTextConfig(
        vocab_size=32,
        bos_token_id=31,
        eos_token_id=31,
        hidden_size=16,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        hc_count=2,
        hc_lowrank=4,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        ple_layer_ids=[1],
        ple_embed_dim=16,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=8,
        split_ngram_parts=3,
    )
    if not request.param:
        return text
    vision = Qwen3_8FlashNextVisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=16,
        num_heads=2,
        out_hidden_size=16,
        num_position_embeddings=16,
    )
    return Qwen3_8FlashNextConfig(text_config=text, vision_config=vision, image_token_id=30)


@pytest.mark.parametrize("shard_count", [3, 128])
def test_config_and_checkpoint_roundtrip(config, shard_count):
    getattr(config, "text_config", config).split_ngram_parts = shard_count
    restored = type(config).from_dict(config.to_dict(), attn_implementation="flash_attention_2")
    model = Qwen3_8FlashNextForCausalLM(restored)
    model.init_buffers_post_meta()
    assert len(get_language_model(model).layers) == 1
    assert model.cp_support(restored).styles == frozenset({"ulysses"})
    assert model.supports_packed_multimodal_training == model.is_vlm
    assert (get_vision_encoder(model) is not None) == model.is_vlm
    original = {
        name: torch.randn_like(value) if value.is_floating_point() else value.clone()
        for name, value in model.state_dict().items()
    }
    converted = dict(original)
    operations = model.conversion_chain(restored)
    apply_prime_to_hf(converted, operations)
    assert model.is_hf_state_dict(converted)
    assert any(name.startswith("model.visual.") for name in converted) == model.is_vlm
    apply_hf_to_prime(converted, operations)
    assert converted.keys() == original.keys()
    for name, value in original.items():
        torch.testing.assert_close(converted[name], value, rtol=0, atol=0)
