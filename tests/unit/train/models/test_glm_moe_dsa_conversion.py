import pytest
import torch

from prime_rl.trainer.models.fp8 import BLOCK_SIZE, ceil_div
from prime_rl.trainer.models.glm_moe_dsa.converting_glm_moe_dsa import (
    _BASE,
    _DENSE,
    _SPARSE,
    convert_tt_layer_to_vllm_kernel,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="triton fp8 kernel requires CUDA")

_BLOCK = 128


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


def _allocate_buffers(state_dict: dict[str, torch.Tensor], prefix: str, is_sparse: bool) -> dict[str, torch.Tensor]:
    specs = _BASE + (_SPARSE if is_sparse else _DENSE)
    buffers: dict[str, torch.Tensor] = {}
    for spec in specs:
        srcs = [state_dict[f"{prefix}.{s}"] for s in spec.sources]
        dst_shape = list(srcs[0].shape)
        dst_shape[spec.cat_dim] *= len(srcs)
        dtype = torch.float8_e4m3fn if spec.quantize else srcs[0].dtype
        buffers[f"{prefix}.{spec.dst}"] = torch.zeros(dst_shape, dtype=dtype, device=srcs[0].device)
        if spec.quantize:
            scale_shape = tuple(
                ceil_div(d, BLOCK_SIZE) if i >= len(dst_shape) - 2 else d for i, d in enumerate(dst_shape)
            )
            buffers[spec.scale_name(prefix)] = torch.zeros(scale_shape, dtype=torch.float32, device=srcs[0].device)
    return buffers


def test_sparse_layer_populates_all_buffers():
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
            f"{prefix}.mlp.shared_expert.w1": torch.randn(_BLOCK, _BLOCK, dtype=torch.bfloat16, device=device),
            f"{prefix}.mlp.shared_expert.w2": torch.randn(_BLOCK, _BLOCK, dtype=torch.bfloat16, device=device),
            f"{prefix}.mlp.shared_expert.w3": torch.randn(_BLOCK, _BLOCK, dtype=torch.bfloat16, device=device),
        }
    )

    buffers = _allocate_buffers(state, prefix, is_sparse=True)
    pointers = {k: v.data_ptr() for k, v in buffers.items()}
    convert_tt_layer_to_vllm_kernel(state, layer_idx=1, out_buffers=buffers)

    # In-place: every pre-registered buffer still points at the same storage.
    for k, ptr in pointers.items():
        assert buffers[k].data_ptr() == ptr, k

    # Expert tensors are cat'd along dim=1 (2*moe_dim) and quantized.
    assert buffers[f"{prefix}.mlp.experts.w13_weight"].shape == (num_experts, 2 * _BLOCK, _BLOCK)
    assert buffers[f"{prefix}.mlp.experts.w13_weight"].dtype == torch.float8_e4m3fn
    assert buffers[f"{prefix}.mlp.experts.w13_weight_scale_inv"].dtype == torch.float32

    # Shared experts stay 2D; fused QKV cats along dim=0.
    assert buffers[f"{prefix}.mlp.shared_experts.gate_up_proj.weight"].shape == (2 * _BLOCK, _BLOCK)
    assert buffers[f"{prefix}.self_attn.fused_qkv_a_proj.weight"].shape == (2 * _BLOCK, 2 * _BLOCK)


def test_dense_layer_populates_all_buffers():
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

    buffers = _allocate_buffers(state, prefix, is_sparse=False)
    convert_tt_layer_to_vllm_kernel(state, layer_idx=0, out_buffers=buffers)

    assert buffers[f"{prefix}.mlp.gate_up_proj.weight"].shape == (2 * _BLOCK, _BLOCK)
    assert buffers[f"{prefix}.mlp.gate_up_proj.weight"].dtype == torch.float8_e4m3fn
    # Sparse-only keys are not part of a dense allocation.
    assert f"{prefix}.mlp.experts.w13_weight" not in buffers
    assert f"{prefix}.mlp.gate.weight" not in buffers
