import pytest

from prime_rl.trainer.models import get_custom_vlm_cls, supports_custom_impl
from prime_rl.trainer.models.muse_glimmer import MuseGlimmerConfig, MuseGlimmerForCausalLM, MuseGlimmerTextConfig
from prime_rl.trainer.vlm import get_language_model, get_layer_prefix


def _text_config(**overrides) -> MuseGlimmerTextConfig:
    values = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        sliding_window=8,
        max_position_embeddings=256,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    values.update(overrides)
    return MuseGlimmerTextConfig(**values)


def test_config_defaults_match_checkpoint_layout():
    config = MuseGlimmerTextConfig()
    full_layers = [idx for idx, layer_type in enumerate(config.layer_types) if layer_type == "full_attention"]
    assert full_layers == list(range(3, 52, 4))
    assert config.query_scale == pytest.approx(3.87)
    assert MuseGlimmerTextConfig(qk_scale_factor=43.7840518911).query_scale == pytest.approx(3.87, abs=1e-3)


def test_composite_config_roundtrip_keeps_vision_config():
    config = MuseGlimmerConfig(text_config=_text_config().to_dict(), vision_config={"hidden_size": 32})
    restored = MuseGlimmerConfig.from_dict(config.to_dict())
    assert isinstance(restored.text_config, MuseGlimmerTextConfig)
    assert restored.text_config.layer_types == config.text_config.layer_types
    assert restored.vision_config == {"hidden_size": 32}
    assert supports_custom_impl(restored)
    assert get_custom_vlm_cls(restored) is MuseGlimmerForCausalLM


def test_state_dict_uses_checkpoint_names():
    config = MuseGlimmerConfig(text_config=_text_config(num_hidden_layers=1).to_dict())
    config._attn_implementation = "flash_attention_2"
    model = MuseGlimmerForCausalLM(config)
    assert len(get_language_model(model).layers) == 1
    assert get_layer_prefix(model) == "model.language_model.layers."
    layer = "model.language_model.layers.0"
    assert set(model.state_dict()) == {
        "lm_head.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        *(
            f"{layer}.{name}.weight"
            for name in (
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "self_attn.gate_proj",
            )
        ),
    }
