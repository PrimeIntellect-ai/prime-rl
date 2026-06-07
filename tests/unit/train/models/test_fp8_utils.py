import pytest
import torch

from prime_rl.trainer.models.kernels.fp8_utils import (
    per_block_cast_to_fp8_tp_triton,
    per_block_cast_to_fp8_triton,
)

pytestmark = [pytest.mark.gpu]


@pytest.mark.parametrize("rows,cols", [(256, 256), (256, 512), (512, 256), (1024, 768), (384, 128)])
def test_tp_cast_matches_materialized_transpose(rows, cols):
    """The fused transpose+cast is bit-identical to casting the materialized
    transpose (128x128 block quant is transpose-symmetric), so DeepGEMM receives
    an identical B tensor."""
    torch.manual_seed(rows + cols)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16) * 0.3

    ref_q, ref_s = per_block_cast_to_fp8_triton(x.transpose(0, 1).contiguous(), False)
    tp_q, tp_s = per_block_cast_to_fp8_tp_triton(x, False)

    assert tp_q.shape == ref_q.shape == (cols, rows)
    assert tp_s.shape == ref_s.shape
    assert tp_q.is_contiguous()
    assert torch.equal(tp_q.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(tp_s, ref_s)
