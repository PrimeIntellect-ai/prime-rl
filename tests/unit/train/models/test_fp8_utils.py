import pytest
import torch
import torch.nn.functional as F

from prime_rl.trainer.models.kernels.fp8_utils import (
    per_block_cast_to_fp8_tp_triton,
    per_block_cast_to_fp8_triton,
    per_token_cast_to_fp8_tp_triton,
    per_token_cast_to_fp8_triton,
)
from prime_rl.trainer.models.layers.fp8_linear import Float8BlockwiseLinear

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


@pytest.mark.parametrize("in_features,out_features", [(256, 256), (259, 196), (10944, 128)])
def test_fp8_linear_ragged_dimensions_and_bias(in_features, out_features):
    torch.manual_seed(1234)
    layer = Float8BlockwiseLinear(in_features, out_features, bias=True, device="cuda")
    x = torch.randn(29, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight_ref = layer.weight.detach().clone().requires_grad_(True)
    bias_ref = layer.bias.detach().clone().requires_grad_(True)
    x_ref = x.detach().clone().requires_grad_(True)
    actual = layer(x)
    reference = F.linear(x_ref, weight_ref, bias_ref)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    reference.backward(grad)

    for value, expected in ((actual, reference), (x.grad, x_ref.grad), (layer.weight.grad, weight_ref.grad)):
        assert value.shape == expected.shape
        assert torch.isfinite(value).all()
        relative_error = (value.float() - expected.float()).norm() / expected.float().norm()
        assert relative_error < 0.08
    torch.testing.assert_close(layer.bias.grad, bias_ref.grad, rtol=0, atol=0)
