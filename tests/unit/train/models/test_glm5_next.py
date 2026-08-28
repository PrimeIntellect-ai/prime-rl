import torch

from prime_rl.trainer.models import get_custom_causal_lm_cls, supports_custom_impl
from prime_rl.trainer.models.conversion_ops import apply_hf_to_prime
from prime_rl.trainer.models.glm5_next import Glm5NextConfig, Glm5NextForCausalLM, Glm5NextTextConfig
from prime_rl.trainer.models.glm5_next.converting_glm5_next import conversion_chain


def _tiny_text_dict() -> dict:
    return {
        "vocab_size": 128,
        "pad_token_id": 0,
        "hidden_size": 8,
        "intermediate_size": 16,
        "moe_intermediate_size": 4,
        "num_hidden_layers": 4,
        "num_attention_heads": 1,
        "q_lora_rank": 4,
        "kv_lora_rank": 4,
        "qk_nope_head_dim": 4,
        "qk_rope_head_dim": 0,
        "v_head_dim": 4,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention"],
        "mlp_layer_types": ["dense", "dense", "dense", "sparse"],
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "n_shared_experts": 1,
        "num_nextn_predict_layers": 1,
        "mhc": True,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "linear_attn_config": {
            "num_heads": 1,
            "head_dim": 4,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
        },
    }


def test_glm5_next_config_folds_composite_text_config() -> None:
    config = Glm5NextConfig(text_config=_tiny_text_dict(), vision_config={"model_type": "glm5_next_vision"})

    assert config.model_type == "glm5_next"
    assert config.get_text_config().model_type == "glm5_next_text"
    assert config.hidden_size == 8
    assert config.glm5_layer_types[-1] == "deepseek_sparse_attention"
    assert config.layer_types[-1] == "sparse"
    assert config.num_experts_per_tok == 1
    assert config.mhc_num_residual_streams == 4
    assert config.mhc_sinkhorn_iterations == 20
    assert config.linear_head_dim == 4


def test_glm5_next_custom_registry_supports_top_level_config() -> None:
    config = Glm5NextConfig(text_config=_tiny_text_dict())

    assert supports_custom_impl(config)
    assert get_custom_causal_lm_cls(config) is Glm5NextForCausalLM


def test_glm5_next_conversion_normalizes_composite_checkpoint_keys() -> None:
    config = Glm5NextTextConfig(**_tiny_text_dict())
    state_dict = {
        "model.language_model.embed_tokens.weight": torch.zeros(128, 8),
        "model.language_model.layers.3.mlp.experts.0.gate_proj.weight": torch.zeros(4, 8),
        "model.language_model.layers.3.mlp.experts.0.down_proj.weight": torch.zeros(8, 4),
        "model.language_model.layers.3.mlp.experts.0.up_proj.weight": torch.zeros(4, 8),
        "model.language_model.layers.3.mlp.experts.1.gate_proj.weight": torch.ones(4, 8),
        "model.language_model.layers.3.mlp.experts.1.down_proj.weight": torch.ones(8, 4),
        "model.language_model.layers.3.mlp.experts.1.up_proj.weight": torch.ones(4, 8),
        "model.language_model.layers.3.mlp.gate.weight": torch.zeros(2, 8),
        "model.language_model.layers.3.mlp.gate.e_score_correction_bias": torch.zeros(2),
        "model.language_model.layers.3.mlp.shared_experts.gate_proj.weight": torch.zeros(4, 8),
        "model.language_model.layers.3.mlp.shared_experts.down_proj.weight": torch.zeros(8, 4),
        "model.language_model.layers.3.mlp.shared_experts.up_proj.weight": torch.zeros(4, 8),
        "model.language_model.layers.3.self_attn.indexer.index_kpool_compress_ape": torch.zeros(4, 8),
        "model.language_model.layers.3.self_attn.indexer.index_kpool_compress_gate": torch.zeros(8, 8),
        "model.language_model.layers.4.eh_proj.weight": torch.zeros(8, 8),
        "model.visual.blocks.0.norm.weight": torch.zeros(8),
        "lm_head.weight": torch.zeros(128, 8),
    }

    apply_hf_to_prime(state_dict, conversion_chain(config))

    assert set(state_dict) == {
        "lm_head.weight",
        "model.embed_tokens.weight",
        "model.layers.3.mlp.experts.down_proj",
        "model.layers.3.mlp.experts.gate_proj",
        "model.layers.3.mlp.experts.up_proj",
        "model.layers.3.mlp.router.gate.weight",
        "model.layers.3.mlp.router.selection_bias",
        "model.layers.3.mlp.shared_expert.down_proj.weight",
        "model.layers.3.mlp.shared_expert.gate_proj.weight",
        "model.layers.3.mlp.shared_expert.up_proj.weight",
    }
    assert state_dict["model.layers.3.mlp.experts.gate_proj"].shape == (2, 4, 8)
