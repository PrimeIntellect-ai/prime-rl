import prime_kernels
import pytest
import torch

from prime_rl.trainer.models.layers.mxfp8_qat import mxfp8_qat_grouped_mm, mxfp8_qat_linear

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
        reason="MXFP8 QAT requires Blackwell",
    ),
]


def _decoded(x):
    kernel = prime_kernels.load("mxfp8")
    return kernel.dequantize(*kernel.quantize(x))


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("groups", [1, 8])
def test_linear_forward_and_bf16_backward(groups, compiled):
    torch.manual_seed(47)
    shape = (2, 64, 256) if groups == 1 else (2, 64, groups, 256)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn((groups * 128, 256), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    op = torch.compile(mxfp8_qat_linear, fullgraph=True) if compiled else mxfp8_qat_linear
    out = op(x, w, groups)
    rounded = _decoded(w.detach())
    rounded_x = _decoded(x.detach().reshape(-1, 256)).reshape_as(x)
    if groups == 1:
        forward = torch.nn.functional.linear(rounded_x, rounded)
        surrogate = torch.nn.functional.linear(x, w + (rounded - w).detach())
    else:
        forward = (
            torch.bmm(
                rounded_x.reshape(-1, groups, 256).transpose(0, 1),
                rounded.view(groups, 128, 256).transpose(1, 2),
            )
            .transpose(0, 1)
            .reshape_as(out)
        )
        surrogate = (
            torch.bmm(
                x.reshape(-1, groups, 256).transpose(0, 1),
                (w + (rounded - w).detach()).view(groups, 128, 256).transpose(1, 2),
            )
            .transpose(0, 1)
            .reshape_as(out)
        )
    torch.testing.assert_close(out, forward, rtol=0, atol=0)
    grad = torch.randn_like(out)
    actual = torch.autograd.grad(out, (x, w), grad)
    expected = torch.autograd.grad(surrogate, (x, w), grad)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_expert_forward_and_bf16_backward(compiled, empty):
    torch.manual_seed(51)
    x = torch.randn((0 if empty else 512, 256), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn((4, 128, 256), device="cuda", dtype=torch.bfloat16).transpose(1, 2).requires_grad_()
    offs = torch.tensor([0, 0, 0, 0] if empty else [128, 128, 384, 512], device="cuda", dtype=torch.int32)
    op = torch.compile(mxfp8_qat_grouped_mm, fullgraph=True) if compiled else mxfp8_qat_grouped_mm
    out = op(x, w, offs)
    grad = torch.randn_like(out)
    actual = torch.autograd.grad(out, (x, w), grad)
    if empty:
        assert actual[0].shape == x.shape
        assert torch.count_nonzero(actual[1]) == 0
        return
    rounded = _decoded(w.detach().transpose(1, 2)).transpose(1, 2)
    forward = torch._grouped_mm(_decoded(x.detach()), rounded, offs=offs)
    torch.testing.assert_close(out, forward, rtol=0, atol=0)
    surrogate = torch._grouped_mm(x, w + (rounded - w).detach(), offs=offs)
    expected = torch.autograd.grad(surrogate, (x, w), grad)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
