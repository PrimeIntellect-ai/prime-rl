import pytest
import torch

from prime_rl.trainer.models.kernels.fp8_utils import (
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
