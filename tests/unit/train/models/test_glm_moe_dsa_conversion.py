import pytest
import torch

from prime_rl.trainer.models.glm_moe_dsa.converting_glm_moe_dsa import convert_tt_layer_to_vllm_kernel

_BLOCK = 128

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="triton fp8 kernel requires CUDA")


def _attn_state(prefix: str, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        f"{prefix}.input_layernorm.weight": torch.randn(_BLOCK, dtype=torch.bfloat16, device=device),
        f"{prefix}.post_attention_layernorm.weight": torch.randn(_BLOCK, dtype=torch.bfloat16, device=device),
        f"{prefix}.self_attn.q_a_layernorm.weight": torch.randn(_BLOCK, dtype=torch.bfloat16, device=device),
        f"{prefix}.self_attn.kv_a_layernorm.weight": torch.randn(_BLOCK, dtype=torch.bfloat16, device=device),
        f"{prefix}.self_attn.q_a_proj.weight": torch.randn(_BLOCK, 2 * _BLOCK, dtype=torch.bfloat16, device=device),
        f"{prefix}.self_attn.kv_a_proj_with_mqa.weight": torch.randn(
            _BLOCK, 2 * _BLOCK, dtype=torch.bfloat16, device=device
        ),
        f"{prefix}.self_attn.q_b_proj.weight": torch.randn(_BLOCK, 2 * _BLOCK, dtype=torch.bfloat16, device=device),
        f"{prefix}.self_attn.kv_b_proj.weight": torch.randn(_BLOCK, 2 * _BLOCK, dtype=torch.bfloat16, device=device),
        f"{prefix}.self_attn.o_proj.weight": torch.randn(2 * _BLOCK, _BLOCK, dtype=torch.bfloat16, device=device),
        f"{prefix}.self_attn.indexer.wq_b.weight": torch.randn(_BLOCK, 2 * _BLOCK, dtype=torch.bfloat16, device=device),
        f"{prefix}.self_attn.indexer.wk.weight": torch.randn(_BLOCK, 2 * _BLOCK, dtype=torch.bfloat16, device=device),
        f"{prefix}.self_attn.indexer.k_norm.weight": torch.randn(_BLOCK, dtype=torch.bfloat16, device=device),
        f"{prefix}.self_attn.indexer.k_norm.bias": torch.randn(_BLOCK, dtype=torch.bfloat16, device=device),
        f"{prefix}.self_attn.indexer.weights_proj.weight": torch.randn(
            _BLOCK, 2 * _BLOCK, dtype=torch.bfloat16, device=device
        ),
    }


def test_sparse_layer_produces_vllm_keys():
    device = torch.device("cuda")
    prefix = "model.layers.1"
    num_experts = 2
    state = _attn_state(prefix, device)
    state.update(
        {
            f"{prefix}.mlp.router.gate.weight": torch.randn(num_experts, _BLOCK, dtype=torch.bfloat16, device=device),
            f"{prefix}.mlp.expert_bias": torch.randn(num_experts, dtype=torch.bfloat16, device=device),
            f"{prefix}.mlp.experts.w1": torch.randn(num_experts, _BLOCK, _BLOCK, dtype=torch.bfloat16, device=device),
            f"{prefix}.mlp.experts.w2": torch.randn(num_experts, _BLOCK, _BLOCK, dtype=torch.bfloat16, device=device),
            f"{prefix}.mlp.experts.w3": torch.randn(num_experts, _BLOCK, _BLOCK, dtype=torch.bfloat16, device=device),
            f"{prefix}.mlp.shared_expert.w1": torch.randn(1, _BLOCK, _BLOCK, dtype=torch.bfloat16, device=device),
            f"{prefix}.mlp.shared_expert.w2": torch.randn(1, _BLOCK, _BLOCK, dtype=torch.bfloat16, device=device),
            f"{prefix}.mlp.shared_expert.w3": torch.randn(1, _BLOCK, _BLOCK, dtype=torch.bfloat16, device=device),
        }
    )

    out = convert_tt_layer_to_vllm_kernel(state.copy(), layer_idx=1)

    # FP8 experts packed (num_experts, 2*moe_dim, dim) along cat_dim=1.
    assert out[f"{prefix}.mlp.experts.w13_weight"].dtype == torch.float8_e4m3fn
    assert out[f"{prefix}.mlp.experts.w13_weight"].shape == (num_experts, 2 * _BLOCK, _BLOCK)
    assert out[f"{prefix}.mlp.experts.w13_weight_scale_inv"].dtype == torch.float32
    assert out[f"{prefix}.mlp.experts.w2_weight"].dtype == torch.float8_e4m3fn

    # Shared experts squeezed to 2D.
    assert out[f"{prefix}.mlp.shared_experts.gate_up_proj.weight"].shape == (2 * _BLOCK, _BLOCK)

    # Fused QKV projection concatenated along dim=0.
    assert out[f"{prefix}.self_attn.fused_qkv_a_proj.weight"].shape == (2 * _BLOCK, 2 * _BLOCK)

    # Router + bias routed through without quantization.
    assert f"{prefix}.mlp.gate.weight" in out
    assert f"{prefix}.mlp.gate.e_score_correction_bias" in out


def test_dense_layer_produces_vllm_keys():
    device = torch.device("cuda")
    prefix = "model.layers.0"
    state = _attn_state(prefix, device)
    state.update(
        {
            f"{prefix}.mlp.gate_proj.weight": torch.randn(_BLOCK, _BLOCK, dtype=torch.bfloat16, device=device),
            f"{prefix}.mlp.up_proj.weight": torch.randn(_BLOCK, _BLOCK, dtype=torch.bfloat16, device=device),
            f"{prefix}.mlp.down_proj.weight": torch.randn(_BLOCK, _BLOCK, dtype=torch.bfloat16, device=device),
        }
    )

    out = convert_tt_layer_to_vllm_kernel(state.copy(), layer_idx=0)

    assert out[f"{prefix}.mlp.gate_up_proj.weight"].shape == (2 * _BLOCK, _BLOCK)
    assert out[f"{prefix}.mlp.gate_up_proj.weight"].dtype == torch.float8_e4m3fn
    assert out[f"{prefix}.mlp.down_proj.weight"].dtype == torch.float8_e4m3fn
    # Sparse-only keys must not leak into a dense conversion.
    assert f"{prefix}.mlp.experts.w13_weight" not in out
    assert f"{prefix}.mlp.gate.weight" not in out
