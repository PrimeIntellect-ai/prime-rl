import torch

from prime_rl.trainer.models.glm_moe_dsa.converting_glm_moe_dsa import convert_tt_layer_to_vllm_kernel


def _build_prime_layer_state(layer_idx: int = 0) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{layer_idx}"
    return {
        f"{prefix}.self_attn.q_a_proj.weight": torch.randn(4, 6),
        f"{prefix}.self_attn.kv_a_proj_with_mqa.weight": torch.randn(3, 6),
        f"{prefix}.self_attn.q_b_proj.weight": torch.randn(5, 6),
        f"{prefix}.self_attn.kv_b_proj.weight": torch.randn(7, 6),
        f"{prefix}.self_attn.o_proj.weight": torch.randn(6, 5),
        f"{prefix}.self_attn.indexer.wq_b.weight": torch.randn(4, 6),
        f"{prefix}.self_attn.indexer.wk.weight": torch.randn(4, 6),
        f"{prefix}.self_attn.indexer.k_norm.weight": torch.randn(6),
        f"{prefix}.self_attn.indexer.k_norm.bias": torch.randn(6),
        f"{prefix}.self_attn.indexer.weights_proj.weight": torch.randn(2, 6),
        f"{prefix}.mlp.gate_proj.weight": torch.randn(8, 6),
        f"{prefix}.mlp.up_proj.weight": torch.randn(8, 6),
        f"{prefix}.mlp.down_proj.weight": torch.randn(6, 8),
        f"{prefix}.mlp.router.gate.weight": torch.randn(4, 6),
        f"{prefix}.mlp.expert_bias": torch.randn(4),
        f"{prefix}.mlp.experts.w1": torch.randn(2, 3, 6),
        f"{prefix}.mlp.experts.w2": torch.randn(2, 6, 3),
        f"{prefix}.mlp.experts.w3": torch.randn(2, 3, 6),
        f"{prefix}.mlp.shared_expert.w1": torch.randn(1, 3, 6),
        f"{prefix}.mlp.shared_expert.w2": torch.randn(1, 6, 3),
        f"{prefix}.mlp.shared_expert.w3": torch.randn(1, 3, 6),
    }


def test_convert_tt_layer_to_vllm_kernel_no_fp8():
    state = _build_prime_layer_state()
    out = convert_tt_layer_to_vllm_kernel(state, layer_idx=0, quantize_fp8=False)

    assert "model.layers.0.self_attn.fused_qkv_a_proj.weight" in out
    assert out["model.layers.0.self_attn.fused_qkv_a_proj.weight"].shape == (7, 6)

    assert "model.layers.0.mlp.gate_up_proj.weight" in out
    assert out["model.layers.0.mlp.gate_up_proj.weight"].shape == (16, 6)

    assert "model.layers.0.mlp.experts.w13_weight" in out
    assert out["model.layers.0.mlp.experts.w13_weight"].shape == (2, 6, 6)
    assert "model.layers.0.mlp.experts.w2_weight" in out
    assert out["model.layers.0.mlp.experts.w2_weight"].shape == (2, 6, 3)

    assert "model.layers.0.mlp.gate.weight" in out
    assert "model.layers.0.mlp.gate.e_score_correction_bias" in out


def test_convert_tt_layer_to_vllm_kernel_with_fp8():
    state = _build_prime_layer_state()
    out = convert_tt_layer_to_vllm_kernel(state, layer_idx=0, quantize_fp8=True)

    assert out["model.layers.0.self_attn.fused_qkv_a_proj.weight"].dtype == torch.float8_e4m3fn
    assert out["model.layers.0.self_attn.fused_qkv_a_proj.weight_scale_inv"].dtype == torch.float32

    assert out["model.layers.0.mlp.experts.w13_weight"].dtype == torch.float8_e4m3fn
    assert out["model.layers.0.mlp.experts.w13_weight_scale_inv"].dtype == torch.float32
    assert out["model.layers.0.mlp.experts.w2_weight"].dtype == torch.float8_e4m3fn
    assert out["model.layers.0.mlp.experts.w2_weight_scale_inv"].dtype == torch.float32
