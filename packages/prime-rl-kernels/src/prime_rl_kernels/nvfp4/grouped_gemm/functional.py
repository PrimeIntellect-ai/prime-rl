from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from prime_rl_kernels.nvfp4.grouped_gemm._extension import _grouped_mm as _grouped_mm_kernel
from prime_rl_kernels.nvfp4.quantize.functional import (
    _NVFP4Tensor,
    quantize_activations,
    quantize_weights,
)

NVFP4Backward = Literal["dequant_bf16", "bf16"]


class _GroupedNVFP4MM(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        matrix: torch.Tensor,
        weight: torch.Tensor,
        offsets: torch.Tensor,
        backward: NVFP4Backward,
    ) -> torch.Tensor:
        activations_nvfp4 = quantize_activations(matrix, offsets)
        weight_nvfp4 = quantize_weights(weight)
        output = _grouped_mm_kernel(
            activations_nvfp4.data,
            weight_nvfp4.data.transpose(-2, -1),
            activations_nvfp4.block_scales,
            weight_nvfp4.block_scales,
            offsets,
            activations_nvfp4.global_scales,
            weight_nvfp4.global_scales,
        )
        ctx.backward_mode = backward
        if backward == "bf16":
            ctx.save_for_backward(matrix, weight, offsets)
        else:
            ctx.save_for_backward(
                activations_nvfp4.data,
                activations_nvfp4.block_scales,
                activations_nvfp4.global_scales,
                weight_nvfp4.data,
                weight_nvfp4.block_scales,
                weight_nvfp4.global_scales,
                offsets,
            )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_output = grad_output.contiguous().bfloat16()
        if ctx.backward_mode == "bf16":
            matrix, weight, offsets = ctx.saved_tensors
        else:
            (
                matrix_data,
                matrix_block_scales,
                matrix_global_scales,
                weight_data,
                weight_block_scales,
                weight_global_scales,
                offsets,
            ) = ctx.saved_tensors

        grad_matrix = grad_weight = None
        if ctx.needs_input_grad[0]:
            if ctx.backward_mode == "dequant_bf16":
                weight = _NVFP4Tensor(
                    weight_data,
                    weight_block_scales,
                    weight_global_scales,
                    None,
                ).dequantize()
            grad_matrix = F.grouped_mm(
                grad_output,
                weight.transpose(-2, -1),
                offs=offsets,
                out_dtype=torch.bfloat16,
            )
        if ctx.needs_input_grad[1]:
            if ctx.backward_mode == "dequant_bf16":
                matrix = _NVFP4Tensor(
                    matrix_data,
                    matrix_block_scales,
                    matrix_global_scales,
                    offsets,
                ).dequantize()
            grad_weight = F.grouped_mm(
                matrix.transpose(0, 1),
                grad_output,
                offs=offsets,
                out_dtype=torch.bfloat16,
            )
        return grad_matrix, grad_weight, None, None


def grouped_gemm(
    matrix: torch.Tensor,
    weight: torch.Tensor,
    *,
    offs: torch.Tensor,
    backward: NVFP4Backward = "dequant_bf16",
) -> torch.Tensor:
    """Grouped NVFP4 forward with the selected BF16 backward operands.

    ``matrix`` has shape ``[M, K]``, ``weight`` has shape ``[G, K, N]``, and
    ``offs`` contains the cumulative row count for each of the ``G`` groups.
    ``dequant_bf16`` reconstructs both operands from the packed forward tensors;
    ``bf16`` retains the original BF16 operands.
    """

    if backward not in ("dequant_bf16", "bf16"):
        raise ValueError(f"unsupported NVFP4 backward mode: {backward}")
    if matrix.ndim != 2 or weight.ndim != 3:
        raise ValueError("matrix must be 2D and weight must be 3D")
    if weight.shape[0] != offs.numel():
        raise ValueError("weight and offsets must have the same number of groups")
    if matrix.shape[1] != weight.shape[1]:
        raise ValueError("matrix and weight contraction dimensions must match")
    if weight.device != matrix.device or weight.dtype != matrix.dtype:
        raise ValueError("weight must have the same CUDA device and dtype as matrix")
    return _GroupedNVFP4MM.apply(matrix, weight, offs, backward)
