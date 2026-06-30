"""MXFP8-quantized expert-parallel all-to-all collectives.

These mirror torchao's ``moe_training.ep`` dispatch/combine primitives but keep bf16 at the
autograd boundary: each collective quantizes its payload to (1x32 e8m0-scaled) MXFP8 *only for
the wire transfer* and dequantizes back to bf16 immediately after, so no ``MXTensor`` leaks into
the surrounding autograd graph (which prime-rl's permute/unpermute slicing can't dispatch on).

The forward/backward asymmetry matches torchtitan:
- dispatch: forward transfers MXFP8, backward transfers bf16 (preserve gradient precision)
- combine:  forward transfers bf16, backward transfers MXFP8
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import all_to_all_single

try:
    from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0
    from torchao.prototype.mx_formats.mx_tensor import MXTensor
except ImportError:
    triton_to_mxfp8_dim0 = MXTensor = None

_BLOCK_SIZE = 32


def _mxfp8_all_to_all(
    x: torch.Tensor,
    output_splits: list[int],
    input_splits: list[int],
    group: dist.ProcessGroup,
    scaling_mode,
) -> torch.Tensor:
    """Quantize ``x`` to MXFP8, all-to-all the qdata and scales, then dequantize to ``x.dtype``."""
    data, scales = triton_to_mxfp8_dim0(
        x.contiguous(), inner_block_size=_BLOCK_SIZE, scaling_mode=str(scaling_mode.value).lower()
    )
    out_data = all_to_all_single(data, output_splits, input_splits, group)
    # NCCL has no float8_e8m0fnu type; transfer the scales as uint8 and reinterpret after.
    out_scales = all_to_all_single(scales.view(torch.uint8), output_splits, input_splits, group)
    out_data = torch.ops._c10d_functional.wait_tensor(out_data)
    out_scales = torch.ops._c10d_functional.wait_tensor(out_scales).view(torch.float8_e8m0fnu)
    mx = MXTensor(
        out_data,
        out_scales,
        elem_dtype=torch.float8_e4m3fn,
        block_size=_BLOCK_SIZE,
        orig_dtype=x.dtype,
        kernel_preference=None,
        act_quant_kwargs=None,
        is_swizzled_scales=False,
    )
    return mx.dequantize(x.dtype)


def _bf16_all_to_all(
    x: torch.Tensor,
    output_splits: list[int],
    input_splits: list[int],
    group: dist.ProcessGroup,
) -> torch.Tensor:
    out = all_to_all_single(x.contiguous(), output_splits, input_splits, group)
    return torch.ops._c10d_functional.wait_tensor(out)


class _MXFP8DispatchA2A(torch.autograd.Function):
    """Dispatch all-to-all: MXFP8 forward, bf16 backward."""

    @staticmethod
    def forward(ctx, x, output_splits, input_splits, group, scaling_mode):
        ctx.output_splits = output_splits
        ctx.input_splits = input_splits
        ctx.group = group
        return _mxfp8_all_to_all(x, output_splits, input_splits, group, scaling_mode)

    @staticmethod
    def backward(ctx, grad_output):
        # Inverse all-to-all in bf16 (swap splits).
        grad_input = _bf16_all_to_all(grad_output, ctx.input_splits, ctx.output_splits, ctx.group)
        return grad_input, None, None, None, None


class _MXFP8CombineA2A(torch.autograd.Function):
    """Combine all-to-all: bf16 forward, MXFP8 backward."""

    @staticmethod
    def forward(ctx, x, output_splits, input_splits, group, scaling_mode):
        ctx.output_splits = output_splits
        ctx.input_splits = input_splits
        ctx.group = group
        ctx.scaling_mode = scaling_mode
        return _bf16_all_to_all(x, output_splits, input_splits, group)

    @staticmethod
    def backward(ctx, grad_output):
        # Inverse all-to-all in MXFP8 (swap splits).
        grad_input = _mxfp8_all_to_all(grad_output, ctx.input_splits, ctx.output_splits, ctx.group, ctx.scaling_mode)
        return grad_input, None, None, None, None


def mxfp8_dispatch_all_to_all(x, output_splits, input_splits, group, scaling_mode):
    return _MXFP8DispatchA2A.apply(x, output_splits, input_splits, group, scaling_mode)


def mxfp8_combine_all_to_all(x, output_splits, input_splits, group, scaling_mode):
    return _MXFP8CombineA2A.apply(x, output_splits, input_splits, group, scaling_mode)
