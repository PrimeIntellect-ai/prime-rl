"""Native nearest-even MXFP8 forward with a BF16 straight-through backward."""

import torch
from torchao.prototype.moe_training.mxfp8_linear import _to_mxfp8_then_scaled_mm
from torchao.prototype.mx_formats import ScaleCalculationMode
from torchao.prototype.mx_formats.kernels import triton_mxfp8_dequant_dim0, triton_to_mxfp8_dim0
from torchao.quantization.quantize_.common import KernelPreference


@torch.library.custom_op("prime_rl::mxfp8_qat_rounded_weight", mutates_args=())
def _rounded_weight(weight: torch.Tensor) -> torch.Tensor:
    rows = weight.contiguous().view(-1, weight.shape[-1])
    data, scales = triton_to_mxfp8_dim0(rows, scaling_mode="rceil")
    return triton_mxfp8_dequant_dim0(data, scales.view(torch.uint8), out_dtype=weight.dtype).view_as(weight)


@_rounded_weight.register_fake
def _rounded_weight_fake(weight):
    return torch.empty_like(weight, memory_format=torch.contiguous_format)


@torch.library.custom_op("prime_rl::mxfp8_qat_linear", mutates_args=())
def mxfp8_qat_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_empty((*x.shape[:-1], weight.shape[0]))
    return _to_mxfp8_then_scaled_mm(
        x.contiguous(), weight.contiguous(), KernelPreference.AUTO, ScaleCalculationMode.RCEIL
    )


@mxfp8_qat_linear.register_fake
def _linear_fake(x, weight):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def _linear_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _linear_backward(ctx, grad):
    x, weight = ctx.saved_tensors
    dy = grad.reshape(-1, grad.shape[-1])
    dx = (dy @ _rounded_weight(weight.detach())).view_as(x) if ctx.needs_input_grad[0] else None
    dw = dy.T @ x.reshape(-1, x.shape[-1]) if ctx.needs_input_grad[1] else None
    return dx, dw


mxfp8_qat_linear.register_autograd(_linear_backward, setup_context=_linear_context)


@torch.library.custom_op("prime_rl::mxfp8_qat_grouped_mm", mutates_args=())
def mxfp8_qat_grouped_mm(x: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    import prime_kernels

    if x.shape[0] == 0:
        return x.new_empty((0, weight.shape[-1]))
    # The native backend expects weights transposed from contiguous (expert, output, input) storage.
    weight = weight.transpose(-2, -1).contiguous().transpose(-2, -1)
    return prime_kernels.load("mxfp8_moe").grouped_gemm(x.contiguous(), weight, offsets)


@mxfp8_qat_grouped_mm.register_fake
def _grouped_fake(x, weight, offsets):
    return x.new_empty((x.shape[0], weight.shape[-1]))


def _grouped_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _grouped_backward(ctx, grad):
    x, weight, offsets = ctx.saved_tensors
    if x.shape[0] == 0:
        return (
            torch.zeros_like(x) if ctx.needs_input_grad[0] else None,
            torch.zeros_like(weight) if ctx.needs_input_grad[1] else None,
            None,
        )
    grad = grad.contiguous()
    dx = (
        torch._grouped_mm(grad, _rounded_weight(weight.detach().transpose(-2, -1)), offs=offsets)
        if ctx.needs_input_grad[0]
        else None
    )
    dw = torch._grouped_mm(x.T, grad, offs=offsets) if ctx.needs_input_grad[1] else None
    return dx, dw, None


mxfp8_qat_grouped_mm.register_autograd(_grouped_backward, setup_context=_grouped_context)
