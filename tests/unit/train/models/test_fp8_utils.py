import pytest
import torch

from prime_rl.trainer.models.kernels.fp8_utils import (
    build_grouped_layout,
    grouped_per_channel_cast_to_fp8_rowmajor_triton,
    grouped_per_token_cast_to_fp8_triton,
    per_block_cast_to_fp8_tp_triton,
    per_block_cast_to_fp8_triton,
    per_token_cast_to_fp8_tp_triton,
    per_token_cast_to_fp8_triton,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
        reason="block-fp8 cast kernels use Triton fp8e4nv (e4m3), only supported on Hopper (SM90) and newer",
    ),
]


@pytest.mark.parametrize("rows,cols", [(256, 256), (256, 512), (512, 256), (1024, 768), (384, 128)])
def test_block_tp_cast_matches_materialized_transpose(rows, cols):
    """The fused transpose+cast is *bit-identical* to unfused."""
    torch.manual_seed(rows + cols)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16) * 0.3

    ref_q, ref_s = per_block_cast_to_fp8_triton(x.transpose(0, 1).contiguous(), False)
    tp_q, tp_s = per_block_cast_to_fp8_tp_triton(x, False)

    assert tp_q.shape == ref_q.shape == (cols, rows)
    assert tp_s.shape == ref_s.shape
    assert tp_q.is_contiguous()
    assert torch.equal(tp_q.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(tp_s, ref_s)


@pytest.mark.parametrize("rows,cols", [(256, 512), (512, 256), (128, 1024), (1024, 768), (384, 512)])
def test_token_tp_cast_matches_materialized_transpose(rows, cols):
    """The fused transpose+cast is *bit-identical* to unfused."""
    torch.manual_seed(rows + cols)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16) * 0.3

    ref_q, ref_s = per_token_cast_to_fp8_triton(x.transpose(0, 1).contiguous(), False)
    tp_q, tp_s = per_token_cast_to_fp8_tp_triton(x, False)

    assert tp_q.shape == ref_q.shape == (cols, rows)
    assert tp_s.shape == ref_s.shape
    assert tp_q.is_contiguous()
    assert torch.equal(tp_q.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(tp_s, ref_s)


@pytest.mark.parametrize(
    "rows,cols,input_scale",
    [(256, 256, 1.0), (256, 512, 0.3), (512, 256, 0.02), (1024, 768, 0.02), (384, 128, 0.005)],
)
def test_token_cast_matches_vllm_cuda_quant(rows, cols, input_scale):
    """The per-token cast is *bit-identical* to vLLM's production CUDA quantizer."""
    pytest.importorskip("vllm")
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

    torch.manual_seed(rows + cols)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16) * input_scale
    assert x.is_contiguous()

    ref_q, ref_s = per_token_group_quant_fp8(x, 128, use_ue8m0=False)
    q, s = per_token_cast_to_fp8_triton(x, False)

    assert q.shape == ref_q.shape == (rows, cols)
    assert s.shape == ref_s.shape == (rows, cols // 128)
    assert torch.equal(s, ref_s)
    assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))


MAGNITUDES = (1e-7, 1e-4, 1e-2, 1.0)
MAX_RELATIVE_ERROR = 0.03
GROUP_ENDS = (200, 328, 384, 704)
COLS = 512


def _grouped_layout():
    offs = torch.tensor(GROUP_ENDS, dtype=torch.int32, device="cuda")
    return build_grouped_layout(offs)


def _occupied_rows(actual_ms, block_starts):
    """Destination rows a group actually fills, skipping the pad up to 128."""
    return torch.cat(
        [
            torch.arange(start * 128, start * 128 + rows, device="cuda")
            for start, rows in zip(block_starts.tolist(), actual_ms.tolist())
        ]
    )


def _round_trip_per_token(magnitude):
    torch.manual_seed(0)
    x = torch.randn(256, COLS, device="cuda", dtype=torch.bfloat16) * magnitude
    q, sf = per_token_cast_to_fp8_triton(x, False)
    x_hat = (q.float().view(-1, COLS // 128, 128) * sf.unsqueeze(-1)).view(-1, COLS)
    return x, x_hat, q.float()


def _round_trip_grouped_per_token(magnitude):
    torch.manual_seed(0)
    _, padded_total_m, _, block_to_group, _, starts, actual_ms, block_starts = _grouped_layout()
    x = torch.randn(GROUP_ENDS[-1], COLS, device="cuda", dtype=torch.bfloat16) * magnitude
    q, sf = grouped_per_token_cast_to_fp8_triton(
        x, padded_total_m, block_to_group, starts, actual_ms, block_starts, False
    )
    rows = _occupied_rows(actual_ms, block_starts)
    q_occupied = q.float()[rows]
    x_hat = (q_occupied.view(-1, COLS // 128, 128) * sf[rows].unsqueeze(-1)).view(-1, COLS)
    return x, x_hat, q_occupied


def _round_trip_grouped_per_channel(magnitude):
    torch.manual_seed(0)
    _, padded_total_m, _, block_to_group, ks, starts, actual_ms, block_starts = _grouped_layout()
    x = torch.randn(GROUP_ENDS[-1], COLS, device="cuda", dtype=torch.bfloat16) * magnitude
    q, sf = grouped_per_channel_cast_to_fp8_rowmajor_triton(
        x, padded_total_m, block_to_group, starts, actual_ms, ks, block_starts, False
    )
    rows = _occupied_rows(actual_ms, block_starts)
    q_occupied = q.float()[rows]
    x_hat = q_occupied * sf[rows // 128]
    return x, x_hat, q_occupied


@pytest.mark.parametrize(
    "round_trip",
    [_round_trip_per_token, _round_trip_grouped_per_token, _round_trip_grouped_per_channel],
    ids=["per_token", "grouped_per_token", "grouped_per_channel"],
)
def test_cast_error_is_scale_invariant(round_trip):
    """Block-scaled e4m3 error is relative, so shrinking the input must not degrade it."""
    errors = []
    for magnitude in MAGNITUDES:
        x, x_hat, q = round_trip(magnitude)
        x = x.float()
        error = ((x_hat - x).norm() / x.norm()).item()
        flushed = (q == 0).float().mean().item()
        assert error < MAX_RELATIVE_ERROR, f"magnitude {magnitude:g}: relative error {error:.4f}"
        assert flushed < 1e-4, f"magnitude {magnitude:g}: {flushed:.3e} of elements flushed to zero"
        errors.append(error)
    assert max(errors) / min(errors) < 1.5, f"relative error tracks input magnitude: {errors}"
