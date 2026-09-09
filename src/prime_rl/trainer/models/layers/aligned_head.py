"""Fixed-K BF16-input projection that preserves the FP32 accumulator output."""

import torch
import triton
from torch import Tensor
from vllm.model_executor.layers.batch_invariant import matmul_kernel_persistent
from vllm.utils.platform_utils import num_compute_units


def fixed_k_fp32(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("The aligned FP32 head requires BF16 inputs and weights")
    a = x.reshape(-1, x.shape[-1])
    b = weight.T
    rows, inner = a.shape
    if b.shape[0] != inner:
        raise ValueError("Head input and weight dimensions disagree")
    columns = b.shape[1]
    output = torch.empty((rows, columns), device=x.device, dtype=torch.float32)
    sms = num_compute_units(x.device.index)
    grid = (min(sms, triton.cdiv(rows, 128) * triton.cdiv(columns, 128)),)
    matmul_kernel_persistent[grid](
        a,
        b,
        output,
        bias,
        rows,
        columns,
        inner,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=8,
        NUM_SMS=sms,
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=output.numel() > 2**31,
        HAS_BIAS=bias is not None,
        num_stages=3,
        num_warps=8,
    )
    return output.reshape(*x.shape[:-1], columns)


class FP32HeadLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor) -> Tensor:
        ctx.save_for_backward(x, weight)
        return fixed_k_fp32(x, weight)

    @staticmethod
    def backward(ctx, grad: Tensor):
        x, weight = ctx.saved_tensors
        flat_grad = grad.reshape(-1, grad.shape[-1])
        dx = (flat_grad @ weight.float()).reshape_as(x).to(x.dtype)
        dw = (flat_grad.T @ x.reshape(-1, x.shape[-1]).float()).to(weight.dtype)
        return dx, dw
