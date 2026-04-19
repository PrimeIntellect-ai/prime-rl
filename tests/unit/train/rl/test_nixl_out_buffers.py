"""Unit tests for the ``out=`` buffers behind the GLM MoE DSA -> vLLM FP8 conversion.

These tests don't require NIXL/RDMA hardware — they verify that the
pre-registered buffer path produces bit-identical outputs to the allocating
path. Bit-identical equality is the invariant that keeps live NIXL correctness
reducible to a single allocating-vs-inplace check.
"""

from __future__ import annotations

import pytest
import torch

from prime_rl.trainer.models.fp8 import BLOCK_SIZE, ceil_div, fp8_block_quantize, grouped_fp8_block_quantize

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="triton fp8 kernel requires CUDA")


@pytest.mark.parametrize("rows,cols", [(256, 256), (256, 384)])
def test_fp8_block_quantize_out_buffers_match_allocating_path(rows: int, cols: int) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    w = torch.randn(rows, cols, dtype=torch.bfloat16, device=device)

    # Allocating path.
    q_ref, s_ref = fp8_block_quantize(w)

    # In-place path with caller-provided buffers.
    out_q = torch.zeros(rows, cols, dtype=torch.float8_e4m3fn, device=device)
    out_sf = torch.zeros(ceil_div(rows, BLOCK_SIZE), ceil_div(cols, BLOCK_SIZE), dtype=torch.float32, device=device)
    fp8_block_quantize(w, out=out_q, sf=out_sf)

    assert torch.equal(q_ref.view(torch.uint8), out_q.view(torch.uint8))
    assert torch.equal(s_ref, out_sf)


def test_grouped_fp8_block_quantize_out_buffers_match_allocating_path() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    groups, rows, cols = 4, 256, 256
    w = torch.randn(groups, rows, cols, dtype=torch.bfloat16, device=device)

    q_ref, s_ref = grouped_fp8_block_quantize(w)

    out_q = torch.zeros(groups, rows, cols, dtype=torch.float8_e4m3fn, device=device)
    out_sf = torch.zeros(
        groups, ceil_div(rows, BLOCK_SIZE), ceil_div(cols, BLOCK_SIZE), dtype=torch.float32, device=device
    )
    grouped_fp8_block_quantize(w, out=out_q, sf=out_sf)

    assert torch.equal(q_ref.view(torch.uint8), out_q.view(torch.uint8))
    assert torch.equal(s_ref, out_sf)


def test_convert_glm_layer_out_buffers_match() -> None:
    """``convert_tt_layer_to_vllm_kernel`` with ``out_buffers`` must produce the
    same expert tensors as the allocating path (bitwise FP8 equality) and
    populate every destination into the caller's dict."""
    from prime_rl.trainer.models.glm_moe_dsa.converting_glm_moe_dsa import convert_tt_layer_to_vllm_kernel

    torch.manual_seed(0)
    device = torch.device("cuda")
    num_experts, moe_dim, dim = 4, 128, 256
    prefix = "model.layers.1"
    state_dict = {
        f"{prefix}.mlp.experts.w1": torch.randn(num_experts, moe_dim, dim, dtype=torch.bfloat16, device=device),
        f"{prefix}.mlp.experts.w2": torch.randn(num_experts, dim, moe_dim, dtype=torch.bfloat16, device=device),
        f"{prefix}.mlp.experts.w3": torch.randn(num_experts, moe_dim, dim, dtype=torch.bfloat16, device=device),
        f"{prefix}.mlp.router.gate.weight": torch.randn(num_experts, dim, dtype=torch.bfloat16, device=device),
        f"{prefix}.mlp.expert_bias": torch.randn(num_experts, dtype=torch.bfloat16, device=device),
        f"{prefix}.mlp.shared_expert.w1": torch.randn(1, moe_dim, dim, dtype=torch.bfloat16, device=device),
        f"{prefix}.mlp.shared_expert.w2": torch.randn(1, dim, moe_dim, dtype=torch.bfloat16, device=device),
        f"{prefix}.mlp.shared_expert.w3": torch.randn(1, moe_dim, dim, dtype=torch.bfloat16, device=device),
    }
    # Attention sources (required by _BASE mapping).
    for name, shape in (
        ("input_layernorm.weight", (dim,)),
        ("post_attention_layernorm.weight", (dim,)),
        ("self_attn.q_a_layernorm.weight", (dim,)),
        ("self_attn.kv_a_layernorm.weight", (dim,)),
        ("self_attn.q_a_proj.weight", (dim, dim)),
        ("self_attn.kv_a_proj_with_mqa.weight", (dim, dim)),
        ("self_attn.q_b_proj.weight", (dim, dim)),
        ("self_attn.kv_b_proj.weight", (dim, dim)),
        ("self_attn.o_proj.weight", (dim, dim)),
        ("self_attn.indexer.wq_b.weight", (dim, dim)),
        ("self_attn.indexer.wk.weight", (dim, dim)),
        ("self_attn.indexer.k_norm.weight", (dim,)),
        ("self_attn.indexer.k_norm.bias", (dim,)),
        ("self_attn.indexer.weights_proj.weight", (dim, dim)),
    ):
        state_dict[f"{prefix}.{name}"] = torch.randn(*shape, dtype=torch.bfloat16, device=device)

    ref = convert_tt_layer_to_vllm_kernel(dict(state_dict), layer_idx=1)

    # Pre-register the four expert slots; everything else should be auto-allocated.
    w13_shape = (num_experts, 2 * moe_dim, dim)
    w2_shape = (num_experts, dim, moe_dim)
    s_w13 = (ceil_div(2 * moe_dim, BLOCK_SIZE), ceil_div(dim, BLOCK_SIZE))
    s_w2 = (ceil_div(dim, BLOCK_SIZE), ceil_div(moe_dim, BLOCK_SIZE))
    out_buffers = {
        f"{prefix}.mlp.experts.w13_weight": torch.zeros(w13_shape, dtype=torch.float8_e4m3fn, device=device),
        f"{prefix}.mlp.experts.w2_weight": torch.zeros(w2_shape, dtype=torch.float8_e4m3fn, device=device),
        f"{prefix}.mlp.experts.w13_weight_scale_inv": torch.zeros(
            (num_experts, *s_w13), dtype=torch.float32, device=device
        ),
        f"{prefix}.mlp.experts.w2_weight_scale_inv": torch.zeros(
            (num_experts, *s_w2), dtype=torch.float32, device=device
        ),
    }
    pre_registered = {k: v.data_ptr() for k, v in out_buffers.items()}
    inp = convert_tt_layer_to_vllm_kernel(dict(state_dict), layer_idx=1, out_buffers=out_buffers)

    # Pre-registered buffers are reused in place (no realloc).
    for name, ptr in pre_registered.items():
        assert inp[name].data_ptr() == ptr, name

    # Every reference key is populated; FP8 tensors match bit-exactly.
    for name, ref_tensor in ref.items():
        assert name in inp, f"missing {name}"
        got = inp[name]
        if ref_tensor.dtype == torch.float8_e4m3fn:
            assert torch.equal(ref_tensor.view(torch.uint8), got.view(torch.uint8)), name
        else:
            assert torch.equal(ref_tensor, got), name
