"""Unit tests for the ``out=`` buffers behind the GLM MoE DSA -> vLLM FP8 conversion.

These tests don't require NIXL/RDMA hardware — they verify that the
pre-registered buffer path is written in place (zero-copy) and matches a
reference pass bit-exactly.
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

    q_ref, s_ref = fp8_block_quantize(w)

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


def test_convert_glm_layer_writes_into_preallocated_buffers() -> None:
    """Running the conversion twice with the same ``out_buffers`` must reuse the
    storage (no realloc) and match a fresh reference pass bit-exactly."""
    from tests.unit.train.models.test_glm_moe_dsa_conversion import _allocate_buffers, _attn_state

    from prime_rl.trainer.models.glm_moe_dsa.converting_glm_moe_dsa import convert_tt_layer_to_vllm_kernel

    torch.manual_seed(0)
    device = torch.device("cuda")
    num_experts, moe_dim, dim = 4, 128, 256
    prefix = "model.layers.1"
    state = _attn_state(prefix, device)
    # Override attn shapes with the ones the rest of this test uses.
    state.update(
        {
            f"{prefix}.mlp.experts.w1": torch.randn(num_experts, moe_dim, dim, dtype=torch.bfloat16, device=device),
            f"{prefix}.mlp.experts.w2": torch.randn(num_experts, dim, moe_dim, dtype=torch.bfloat16, device=device),
            f"{prefix}.mlp.experts.w3": torch.randn(num_experts, moe_dim, dim, dtype=torch.bfloat16, device=device),
            f"{prefix}.mlp.router.gate.weight": torch.randn(num_experts, dim, dtype=torch.bfloat16, device=device),
            f"{prefix}.mlp.expert_bias": torch.randn(num_experts, dtype=torch.bfloat16, device=device),
            f"{prefix}.mlp.shared_expert.w1": torch.randn(moe_dim, dim, dtype=torch.bfloat16, device=device),
            f"{prefix}.mlp.shared_expert.w2": torch.randn(dim, moe_dim, dtype=torch.bfloat16, device=device),
            f"{prefix}.mlp.shared_expert.w3": torch.randn(moe_dim, dim, dtype=torch.bfloat16, device=device),
        }
    )

    ref_buffers = _allocate_buffers(state, prefix, is_sparse=True)
    convert_tt_layer_to_vllm_kernel(state, layer_idx=1, out_buffers=ref_buffers)

    inp_buffers = _allocate_buffers(state, prefix, is_sparse=True)
    pointers = {k: v.data_ptr() for k, v in inp_buffers.items()}
    convert_tt_layer_to_vllm_kernel(state, layer_idx=1, out_buffers=inp_buffers)

    for k, ptr in pointers.items():
        assert inp_buffers[k].data_ptr() == ptr, k

    for name, ref in ref_buffers.items():
        got = inp_buffers[name]
        if ref.dtype == torch.float8_e4m3fn:
            assert torch.equal(ref.view(torch.uint8), got.view(torch.uint8)), name
        else:
            assert torch.equal(ref, got), name
