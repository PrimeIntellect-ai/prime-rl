import prime_kernels
import pytest
import torch
import torch.nn.functional as F
from torchao.prototype.moe_training.mxfp8_linear import _to_mxfp8_then_scaled_mm
from torchao.prototype.mx_formats import ScaleCalculationMode
from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0
from torchao.quantization.quantize_.common import KernelPreference

from prime_rl.trainer.models.layers.mxfp8_qat import mxfp8_qat_grouped_mm, mxfp8_qat_linear

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not prime_kernels.is_available("mxfp8_moe"), reason="Requires MXFP8 on Blackwell"),
]


def _decoded(x):
    data, scales = triton_to_mxfp8_dim0(x.contiguous(), scaling_mode="rceil")
    return (data.float().unflatten(-1, (-1, 32)) * scales.float().unsqueeze(-1)).flatten(-2).bfloat16()


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("rows", [0, 33, 128])
def test_dense_qat_matches_native_forward_and_bf16_surrogate(rows, compiled):
    torch.manual_seed(19)
    x = torch.randn((rows, 256), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn((128, 256), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    op = torch.compile(mxfp8_qat_linear, fullgraph=True) if compiled else mxfp8_qat_linear
    actual = op(x, weight)
    if rows:
        native = _to_mxfp8_then_scaled_mm(x, weight, KernelPreference.AUTO, ScaleCalculationMode.RCEIL)
        torch.testing.assert_close(actual, native, atol=0, rtol=0)
    rounded = _decoded(weight.detach())
    surrogate = F.linear(x, weight + (rounded - weight).detach())
    grad = torch.randn((128, rows), device="cuda", dtype=torch.bfloat16).T
    actual_grad = torch.autograd.grad(actual, (x, weight), grad)
    reference_grad = torch.autograd.grad(surrogate, (x, weight), grad)
    for observed, expected in zip(actual_grad, reference_grad):
        torch.testing.assert_close(observed, expected, atol=0, rtol=0)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("groups", [4, 48])
@pytest.mark.parametrize("empty", [False, True])
def test_expert_qat_matches_native_forward_and_bf16_surrogate(compiled, groups, empty):
    torch.manual_seed(23)
    counts = [0 if empty or i % 3 == 1 else 64 for i in range(groups)]
    offsets = torch.tensor(counts, device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    x = torch.randn((sum(counts), 256), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn((groups, 128, 256), device="cuda", dtype=torch.bfloat16).transpose(1, 2).requires_grad_()
    op = torch.compile(mxfp8_qat_grouped_mm, fullgraph=True) if compiled else mxfp8_qat_grouped_mm
    actual = op(x, weight, offsets)
    if not empty:
        native = prime_kernels.load("mxfp8_moe").grouped_gemm(x, weight, offsets)
        torch.testing.assert_close(actual, native, atol=0, rtol=0)
    rounded = _decoded(weight.detach().transpose(1, 2)).transpose(1, 2)
    surrogate_weight = weight + (rounded - weight).detach()
    surrogate = torch.cat([part @ surrogate_weight[i] for i, part in enumerate(x.split(counts))])
    grad = torch.randn((128, sum(counts)), device="cuda", dtype=torch.bfloat16).T
    actual_grad = torch.autograd.grad(actual, (x, weight), grad)
    reference_grad = torch.autograd.grad(surrogate, (x, weight), grad)
    for observed, expected in zip(actual_grad, reference_grad):
        torch.testing.assert_close(observed, expected, atol=0, rtol=0)


@pytest.mark.parametrize("high_precision_wgrad", [False, True])
def test_native_expert_backward_remains_available(high_precision_wgrad):
    kernel = prime_kernels.load("mxfp8_moe")
    x = torch.randn((128, 256), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn((4, 128, 256), device="cuda", dtype=torch.bfloat16).transpose(1, 2).requires_grad_()
    offsets = torch.tensor([32, 64, 96, 128], device="cuda", dtype=torch.int32)
    output = kernel.grouped_gemm(x, weight, offsets, high_precision_wgrad=high_precision_wgrad)
    for grad in torch.autograd.grad(output, (x, weight), torch.randn_like(output)):
        assert torch.isfinite(grad).all()


def test_expert_qat_dispatch_padding():
    from prime_rl.trainer.distributed.token_dispatcher import permute_for_grouped_gemm, unpermute_from_grouped_gemm

    counts = [33, 0, 133, 7]
    x = torch.randn((sum(counts), 256), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn((4, 128, 256), device="cuda", dtype=torch.bfloat16).transpose(1, 2).requires_grad_()
    padded, aligned_counts, permutation = permute_for_grouped_gemm(
        x,
        torch.tensor(counts, device="cuda", dtype=torch.int32),
        experts_per_rank=4,
        num_ranks=1,
        alignment=prime_kernels.load("mxfp8_moe").TOKEN_GROUP_ALIGNMENT,
    )
    offsets = aligned_counts.cumsum(0, dtype=torch.int32)
    actual = unpermute_from_grouped_gemm(mxfp8_qat_grouped_mm(padded, weight, offsets), permutation)
    native = unpermute_from_grouped_gemm(
        prime_kernels.load("mxfp8_moe").grouped_gemm(padded, weight, offsets), permutation
    )
    torch.testing.assert_close(actual, native, atol=0, rtol=0)
    rounded = _decoded(weight.detach().transpose(1, 2)).transpose(1, 2)
    surrogate_weight = weight + (rounded - weight).detach()
    surrogate = torch.cat([part @ surrogate_weight[i] for i, part in enumerate(x.split(counts))])
    grad = torch.randn_like(actual)
    for observed, expected in zip(
        torch.autograd.grad(actual, (x, weight), grad), torch.autograd.grad(surrogate, (x, weight), grad)
    ):
        torch.testing.assert_close(observed, expected, atol=0, rtol=0)
