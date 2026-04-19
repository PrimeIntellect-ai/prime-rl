"""Unit tests for the ``out=`` refactor behind NIXL weight transfer.

These tests don't require NIXL/RDMA hardware — they verify that the
pre-registered buffer path produces bit-identical outputs to the existing
allocating path. Bit-identical equality is the invariant we need to keep
live NIXL correctness reducible to a single allocating-vs-inplace check.
"""

from __future__ import annotations

import pytest
import torch

from prime_rl.trainer.models.fp8 import fp8_blockwise_scale_shape, quantize_to_fp8_blockwise


@pytest.mark.parametrize("rows,cols", [(256, 256), (300, 256), (256, 300), (256, 384)])
def test_quantize_out_buffers_match_allocating_path(rows: int, cols: int) -> None:
    torch.manual_seed(0)
    w = torch.randn(rows, cols, dtype=torch.bfloat16)
    scale_shape = fp8_blockwise_scale_shape(rows, cols, block_size=128)

    out_q = torch.zeros(rows, cols, dtype=torch.float8_e4m3fn)
    out_scale = torch.zeros(scale_shape, dtype=torch.float32)

    q_ref, s_ref = quantize_to_fp8_blockwise(w, block_size=128)
    q_inp, s_inp = quantize_to_fp8_blockwise(w, block_size=128, out_q=out_q, out_scale=out_scale)

    # Returned tensor IS the out= buffer (identity), not a copy.
    assert q_inp.data_ptr() == out_q.data_ptr()
    assert s_inp.data_ptr() == out_scale.data_ptr()

    # fp8 bitwise compare: cast both to uint8 view.
    assert torch.equal(q_ref.view(torch.uint8), out_q.view(torch.uint8))
    assert torch.equal(s_ref, out_scale)


def test_quantize_out_buffer_validates_shape() -> None:
    w = torch.randn(256, 256, dtype=torch.bfloat16)
    bad = torch.zeros(100, 100, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError):
        quantize_to_fp8_blockwise(w, out_q=bad)


def test_quantize_out_buffer_validates_dtype() -> None:
    w = torch.randn(256, 256, dtype=torch.bfloat16)
    bad = torch.zeros(256, 256, dtype=torch.bfloat16)  # wrong dtype
    with pytest.raises(ValueError):
        quantize_to_fp8_blockwise(w, out_q=bad)


def test_convert_glm_layer_out_buffers_match() -> None:
    """``convert_tt_layer_to_vllm_kernel`` with out_buffers must produce the same
    expert tensors as the allocating path (bitwise FP8 equality)."""
    from prime_rl.trainer.models.glm_moe_dsa.converting_glm_moe_dsa import convert_tt_layer_to_vllm_kernel

    torch.manual_seed(0)
    num_experts, moe_dim, dim = 4, 128, 256
    prefix = "model.layers.1"
    state_dict = {
        f"{prefix}.mlp.experts.w1": torch.randn(num_experts, moe_dim, dim, dtype=torch.bfloat16),
        f"{prefix}.mlp.experts.w2": torch.randn(num_experts, dim, moe_dim, dtype=torch.bfloat16),
        f"{prefix}.mlp.experts.w3": torch.randn(num_experts, moe_dim, dim, dtype=torch.bfloat16),
        f"{prefix}.mlp.router.gate.weight": torch.randn(num_experts, dim, dtype=torch.bfloat16),
    }

    # Reference: allocating path.
    ref = convert_tt_layer_to_vllm_kernel(dict(state_dict), layer_idx=1, quantize_fp8=True)

    # Inplace: pre-register w13/w2 slots + their scales.
    w13_shape = (num_experts, 2 * moe_dim, dim)
    w2_shape = (num_experts, dim, moe_dim)
    s_w13 = fp8_blockwise_scale_shape(2 * moe_dim, dim)
    s_w2 = fp8_blockwise_scale_shape(dim, moe_dim)
    out_buffers = {
        f"{prefix}.mlp.experts.w13_weight": torch.zeros(w13_shape, dtype=torch.float8_e4m3fn),
        f"{prefix}.mlp.experts.w2_weight": torch.zeros(w2_shape, dtype=torch.float8_e4m3fn),
        f"{prefix}.mlp.experts.w13_weight_scale_inv": torch.zeros((num_experts, *s_w13), dtype=torch.float32),
        f"{prefix}.mlp.experts.w2_weight_scale_inv": torch.zeros((num_experts, *s_w2), dtype=torch.float32),
    }
    inp = convert_tt_layer_to_vllm_kernel(
        dict(state_dict), layer_idx=1, quantize_fp8=True, out_buffers=out_buffers
    )

    # Identity check — the returned dict values should be the same buffers we passed in.
    for name in out_buffers:
        assert inp[name].data_ptr() == out_buffers[name].data_ptr()

    # Bitwise equality with the reference path.
    for name, ref_tensor in ref.items():
        if name in out_buffers:
            if ref_tensor.dtype == torch.float8_e4m3fn:
                assert torch.equal(ref_tensor.view(torch.uint8), inp[name].view(torch.uint8)), name
            else:
                assert torch.equal(ref_tensor, inp[name]), name
