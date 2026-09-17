import pytest
import torch

from prime_rl.trainer.models.conversion_ops import apply_prime_to_hf
from prime_rl.trainer.models.glm4_moe.converting_glm4_moe import glm_moe_layer_ops
from prime_rl.trainer.models.glm_moe_dsa.converting_glm_moe_dsa import quantize_tt_layer_to_vllm_fp8_checkpoint


def _build_prime_layer_state(layer_idx: int = 0) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{layer_idx}"
    return {
        f"{prefix}.self_attn.q_a_proj.weight": torch.randn(4, 6),
        f"{prefix}.self_attn.kv_a_proj_with_mqa.weight": torch.randn(3, 6),
        f"{prefix}.self_attn.q_b_proj.weight": torch.randn(5, 6),
        f"{prefix}.self_attn.kv_b_proj.weight": torch.randn(7, 6),
        f"{prefix}.self_attn.o_proj.weight": torch.randn(6, 5),
        f"{prefix}.self_attn.q_a_layernorm.weight": torch.randn(4),
        f"{prefix}.self_attn.kv_a_layernorm.weight": torch.randn(3),
        f"{prefix}.self_attn.indexer.wq_b.weight": torch.randn(4, 6),
        f"{prefix}.self_attn.indexer.wk.weight": torch.randn(4, 6),
        f"{prefix}.self_attn.indexer.k_norm.weight": torch.randn(6),
        f"{prefix}.self_attn.indexer.k_norm.bias": torch.randn(6),
        f"{prefix}.self_attn.indexer.weights_proj.weight": torch.randn(2, 6),
        f"{prefix}.input_layernorm.weight": torch.randn(6),
        f"{prefix}.post_attention_layernorm.weight": torch.randn(6),
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


def _hf_layer_state(layer_idx: int = 0) -> dict[str, torch.Tensor]:
    """One layer in HF checkpoint naming, as the transport layer produces it."""
    return apply_prime_to_hf(_build_prime_layer_state(layer_idx), glm_moe_layer_ops(layer_idx))


def test_fp8_checkpoint_wire_format_matches_disk_fp8_checkpoint():
    """The quantized wire stream must match what a blockwise-fp8 disk checkpoint
    of this architecture looks like — that is the only fp8 layout vLLM loads
    natively, and the engine consumes the stream through its checkpoint
    weight-loading path (TP/EP slicing, fused params, fp8 handling)."""
    out = quantize_tt_layer_to_vllm_fp8_checkpoint(_hf_layer_state(0), layer_idx=0)
    p = "model.layers.0"

    # Attention projections: separate q_a / kv_a tensors, fp8 + f32 blockwise
    # scales — NOT pre-fused (the engine fuses them itself via the fused_qkv_a_proj
    # weight_loader) and NOT transposed to any kernel layout.
    assert out[f"{p}.self_attn.q_a_proj.weight"].dtype == torch.float8_e4m3fn
    assert out[f"{p}.self_attn.q_a_proj.weight"].shape == (4, 6)
    assert out[f"{p}.self_attn.q_a_proj.weight_scale_inv"].dtype == torch.float32
    assert out[f"{p}.self_attn.kv_a_proj_with_mqa.weight"].dtype == torch.float8_e4m3fn
    assert f"{p}.self_attn.fused_qkv_a_proj.weight" not in out
    for proj in ("q_b_proj", "kv_b_proj", "o_proj"):
        assert out[f"{p}.self_attn.{proj}.weight"].dtype == torch.float8_e4m3fn
        assert out[f"{p}.self_attn.{proj}.weight_scale_inv"].dtype == torch.float32

    # Attention norms stay unquantized.
    for norm in (
        f"{p}.self_attn.q_a_layernorm.weight",
        f"{p}.self_attn.kv_a_layernorm.weight",
        f"{p}.input_layernorm.weight",
        f"{p}.post_attention_layernorm.weight",
    ):
        assert out[norm].dtype == torch.float32

    # Indexer: wq_b/wk are fp8 (the engine dequantizes wk into the fused bf16
    # wk_weights_proj while loading), k_norm and weights_proj stay unquantized.
    assert out[f"{p}.self_attn.indexer.wq_b.weight"].dtype == torch.float8_e4m3fn
    assert out[f"{p}.self_attn.indexer.wk.weight"].dtype == torch.float8_e4m3fn
    assert out[f"{p}.self_attn.indexer.weights_proj.weight"].dtype == torch.float32
    assert out[f"{p}.self_attn.indexer.k_norm.weight"].dtype == torch.float32

    # Dense MLP: plain checkpoint names, fp8.
    assert out[f"{p}.mlp.gate_proj.weight"].dtype == torch.float8_e4m3fn

    # Router stays unquantized, bias keeps fp32.
    assert out[f"{p}.mlp.gate.weight"].dtype == torch.float32
    assert out[f"{p}.mlp.gate.e_score_correction_bias"].dtype == torch.float32

    # Routed experts: per-expert gate/up/down checkpoint names, fp8 + scales —
    # NOT pre-stacked w13/w2 kernel tensors (the engine owns the TP/EP slicing
    # and the w13 stacking).
    assert out[f"{p}.mlp.experts.0.gate_proj.weight"].dtype == torch.float8_e4m3fn
    assert out[f"{p}.mlp.experts.0.gate_proj.weight"].shape == (3, 6)
    assert out[f"{p}.mlp.experts.0.down_proj.weight"].shape == (6, 3)
    assert out[f"{p}.mlp.experts.1.up_proj.weight"].dtype == torch.float8_e4m3fn
    assert f"{p}.mlp.experts.w13_weight" not in out
    assert f"{p}.mlp.experts.w2_weight" not in out

    # Shared experts: checkpoint names, fp8.
    assert out[f"{p}.mlp.shared_experts.gate_proj.weight"].dtype == torch.float8_e4m3fn


def test_fp8_checkpoint_scale_geometry():
    """Scales are the blockwise grid of the full [out, in] weight, [ceil/B, ceil/B]."""
    state = {
        "model.layers.0.self_attn.q_b_proj.weight": torch.randn(300, 260),
    }
    out = quantize_tt_layer_to_vllm_fp8_checkpoint(state, layer_idx=0)
    weight = out["model.layers.0.self_attn.q_b_proj.weight"]
    scale = out["model.layers.0.self_attn.q_b_proj.weight_scale_inv"]
    assert weight.dtype == torch.float8_e4m3fn
    assert weight.shape == (300, 260)
    assert scale.shape == (3, 3)  # ceil(300/128) x ceil(260/128)


def test_fp8_checkpoint_quantization_is_invertible():
    """Dequantizing the fp8 wire tensor with its scale recovers the weight."""
    torch.manual_seed(0)
    weight = torch.randn(256, 256) * 4
    state = {"model.layers.3.mlp.experts.0.down_proj.weight": weight.clone()}
    out = quantize_tt_layer_to_vllm_fp8_checkpoint(state, layer_idx=3)
    q = out["model.layers.3.mlp.experts.0.down_proj.weight"].float()
    s = out["model.layers.3.mlp.experts.0.down_proj.weight_scale_inv"]
    blocks = q.view(2, 128, 2, 128).permute(0, 2, 1, 3)
    dequant = (blocks * s[:, :, None, None]).permute(0, 2, 1, 3).reshape(256, 256)
    assert torch.allclose(dequant, weight, rtol=7e-2)


def test_fp8_checkpoint_rejects_foreign_layer_keys():
    state = {"model.layers.5.self_attn.q_a_proj.weight": torch.randn(4, 6)}
    with pytest.raises(ValueError, match="outside layer 0"):
        quantize_tt_layer_to_vllm_fp8_checkpoint(state, layer_idx=0)


def test_fp8_checkpoint_keeps_small_1d_tensors_unquantized():
    state = {"model.layers.0.mlp.gate.e_score_correction_bias": torch.randn(8)}
    out = quantize_tt_layer_to_vllm_fp8_checkpoint(state, layer_idx=0)
    assert "model.layers.0.mlp.gate.e_score_correction_bias" in out
    assert "model.layers.0.mlp.gate.e_score_correction_bias.weight_scale_inv" not in out
