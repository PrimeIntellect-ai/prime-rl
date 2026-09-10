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
        f"{prefix}.mlp.router.selection_bias": torch.randn(4),
        f"{prefix}.mlp.experts.gate_proj": torch.randn(2, 3, 6),
        f"{prefix}.mlp.experts.up_proj": torch.randn(2, 3, 6),
        f"{prefix}.mlp.experts.down_proj": torch.randn(2, 6, 3),
        f"{prefix}.mlp.shared_expert.gate_proj.weight": torch.randn(3, 6),
        f"{prefix}.mlp.shared_expert.up_proj.weight": torch.randn(3, 6),
        f"{prefix}.mlp.shared_expert.down_proj.weight": torch.randn(6, 3),
    }


def test_convert_tt_layer_to_vllm_kernel_no_fp8():
    state = _build_prime_layer_state()
    out = convert_tt_layer_to_vllm_kernel(state, layer_idx=0, quantize_fp8=False)

    assert "model.layers.0.self_attn.fused_qkv_a_proj.weight" in out
    assert out["model.layers.0.self_attn.fused_qkv_a_proj.weight"].shape == (7, 6)
    assert "model.layers.0.self_attn.indexer.wk_weights_proj.weight" in out
    assert out["model.layers.0.self_attn.indexer.wk_weights_proj.weight"].shape == (6, 6)
    assert "model.layers.0.self_attn.indexer.wk.weight" not in out
    assert "model.layers.0.self_attn.indexer.weights_proj.weight" not in out

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
    assert out["model.layers.0.self_attn.indexer.wk_weights_proj.weight"].dtype == torch.float32
    assert "model.layers.0.self_attn.indexer.wk.weight_scale_inv" not in out

    assert out["model.layers.0.mlp.experts.w13_weight"].dtype == torch.float8_e4m3fn
    assert out["model.layers.0.mlp.experts.w13_weight_scale_inv"].dtype == torch.float32
    assert out["model.layers.0.mlp.experts.w2_weight"].dtype == torch.float8_e4m3fn
    assert out["model.layers.0.mlp.experts.w2_weight_scale_inv"].dtype == torch.float32


def test_glm4_kernel_conversion_preserves_glm_quantization_and_pads_ragged_input():
    from prime_rl.trainer.models.glm4_moe.kernel_conversion import convert_glm4_layer_to_vllm_kernel

    torch.manual_seed(7)
    prefix = "model.layers.0"
    state = {
        f"{prefix}.self_attn.{part}_proj.weight": torch.randn(rows, 128)
        for part, rows in (("q", 256), ("k", 128), ("v", 128))
    }
    state.update(
        {
            f"{prefix}.self_attn.q_norm.weight": torch.randn(128),
            f"{prefix}.self_attn.k_norm.weight": torch.randn(128),
            f"{prefix}.self_attn.o_proj.weight": torch.randn(128, 256),
            f"{prefix}.mlp.gate_proj.weight": torch.randn(176, 128),
            f"{prefix}.mlp.up_proj.weight": torch.randn(176, 128),
            f"{prefix}.mlp.down_proj.weight": torch.randn(128, 176),
        }
    )
    expected = convert_tt_layer_to_vllm_kernel(state, 0, quantize_fp8=True)
    actual = convert_glm4_layer_to_vllm_kernel(state, 0, quantize_fp8=True)
    for name, reference in expected.items():
        value = actual[name]
        if value.ndim == 2 and value.shape != reference.shape:
            assert value.shape == (128, 256)
            assert torch.count_nonzero(value[:, 176:].float()) == 0
            value = value[:, :176]
        torch.testing.assert_close(value.float(), reference.float(), rtol=0, atol=0)
    assert actual[f"{prefix}.self_attn.qkv_proj.weight"].shape == (512, 128)
    assert actual[f"{prefix}.self_attn.qkv_proj.weight"].dtype == torch.float8_e4m3fn
    torch.testing.assert_close(actual[f"{prefix}.self_attn.q_norm.weight"], state[f"{prefix}.self_attn.q_norm.weight"])
