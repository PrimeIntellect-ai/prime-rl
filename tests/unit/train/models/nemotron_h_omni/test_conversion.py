import torch

from prime_rl.trainer.models.conversion_ops import apply_hf_to_prime, apply_prime_to_hf
from prime_rl.trainer.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
from prime_rl.trainer.models.nemotron_h_omni.converting_nemotron_h_omni import conversion_chain


def _config() -> object:
    class VisionConfig:
        num_hidden_layers = 1

    class Config:
        llm_config = NemotronHConfig(
            hidden_size=8,
            vocab_size=16,
            layers_block_type=["moe"],
            n_routed_experts=2,
            num_experts_per_tok=1,
            moe_intermediate_size=4,
            moe_shared_expert_intermediate_size=4,
            moe_latent_size=4,
        )
        vision_config = VisionConfig()

    return Config()


def test_composite_conversion_maps_language_vision_and_projector_weights():
    state_dict = {
        "language_model.backbone.embeddings.weight": torch.arange(16),
        "language_model.backbone.layers.0.mixer.gate.weight": torch.arange(8),
        "language_model.backbone.layers.0.mixer.experts.0.up_proj.weight": torch.ones(4, 4),
        "language_model.backbone.layers.0.mixer.experts.1.up_proj.weight": torch.full((4, 4), 2.0),
        "language_model.lm_head.weight": torch.arange(16),
        "vision_model.radio_model.model.patch_generator.embedder.weight": torch.arange(96).reshape(8, 12),
        "vision_model.radio_model.model.blocks.0.attn.qkv.weight": torch.arange(48).reshape(24, 2),
        "vision_model.radio_model.model.blocks.0.attn.qkv.bias": torch.arange(24),
        "mlp1.0.weight": torch.ones(8),
        "vision_projector.vision_final_layernorm.weight": torch.ones(8),
        "language_model.mtp.layer.weight": torch.ones(1),
        "backbone.embeddings.weight": torch.zeros(16),
    }

    apply_hf_to_prime(state_dict, conversion_chain(_config()))

    assert "model.language_model.embed_tokens.weight" in state_dict
    assert "model.language_model.layers.0.mlp.router.gate.weight" in state_dict
    assert state_dict["model.language_model.layers.0.mlp.experts.up_proj"].shape == (2, 4, 4)
    assert "lm_head.weight" in state_dict
    assert "model.vision_model.embeddings.patch_projection.weight" in state_dict
    assert "model.vision_model.encoder.layer.0.attention.attention.query.weight" in state_dict
    assert "model.vision_model.encoder.layer.0.attention.attention.key.weight" in state_dict
    assert "model.vision_model.encoder.layer.0.attention.attention.value.weight" in state_dict
    assert "model.vision_projector.mlp1.norm.weight" in state_dict
    assert "model.vision_projector.vision_final_layernorm.weight" in state_dict
    experts = state_dict["model.language_model.layers.0.mlp.experts.up_proj"]
    torch.testing.assert_close(experts[0], torch.ones(4, 4))
    torch.testing.assert_close(experts[1], torch.full((4, 4), 2.0))
    qkv = torch.arange(48).reshape(24, 2)
    for name, expected in zip(
        ("query", "key", "value"),
        qkv.chunk(3),
        strict=True,
    ):
        torch.testing.assert_close(
            state_dict[f"model.vision_model.encoder.layer.0.attention.attention.{name}.weight"],
            expected,
        )
    assert not any(name.startswith(("language_model.mtp.", "backbone.")) for name in state_dict)


def test_composite_conversion_roundtrip_uses_canonical_checkpoint_names():
    prime_state_dict = {
        "model.language_model.embed_tokens.weight": torch.arange(16),
        "model.vision_model.encoder.layer.0.attention.attention.query.weight": torch.ones(2, 2),
        "model.vision_model.encoder.layer.0.attention.attention.key.weight": torch.full((2, 2), 2.0),
        "model.vision_model.encoder.layer.0.attention.attention.value.weight": torch.full((2, 2), 3.0),
        "model.vision_projector.mlp1.linear1.weight": torch.ones(4, 8),
        "lm_head.weight": torch.arange(16),
    }

    apply_prime_to_hf(prime_state_dict, conversion_chain(_config()))

    assert "language_model.backbone.embeddings.weight" in prime_state_dict
    assert "vision_model.radio_model.model.blocks.0.attn.qkv.weight" in prime_state_dict
    torch.testing.assert_close(
        prime_state_dict["vision_model.radio_model.model.blocks.0.attn.qkv.weight"],
        torch.cat((torch.ones(2, 2), torch.full((2, 2), 2.0), torch.full((2, 2), 3.0))),
    )
    assert "mlp1.1.weight" in prime_state_dict
    assert "language_model.lm_head.weight" in prime_state_dict
    assert not any(name.startswith("model.") for name in prime_state_dict)
