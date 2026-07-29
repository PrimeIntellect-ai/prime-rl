from __future__ import annotations

from typing import Literal, TypeAlias

import torch
import torch.nn.functional as F

from prime_rl_kernels.nvfp4.grouped_gemm._extension import _grouped_mm as _grouped_mm_kernel
from prime_rl_kernels.nvfp4.quantize.functional import (
    _NVFP4Tensor,
    quantize_activations,
    quantize_weights,
)

_NVFP4_GEMM_ALIGNMENT = 32
NVFP4Backward: TypeAlias = Literal["bf16", "bf16_dequantized"]


def _check_blackwell() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("NVFP4 grouped GEMM requires CUDA")
    capability = torch.cuda.get_device_capability()
    if capability != (10, 0):
        raise RuntimeError(
            f"NVFP4 grouped GEMM currently requires SM100, but the current device is SM{capability[0]}{capability[1]}"
        )
    if not hasattr(torch, "float4_e2m1fn_x2"):
        raise RuntimeError("NVFP4 grouped GEMM requires PyTorch with float4_e2m1fn_x2 support")


def _check_matrix(matrix: torch.Tensor, name: str) -> None:
    if matrix.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    if matrix.dtype != torch.bfloat16:
        raise ValueError(f"{name} must have dtype torch.bfloat16")
    if matrix.shape[-1] % _NVFP4_GEMM_ALIGNMENT != 0:
        raise ValueError(f"{name}'s contraction dimension must be divisible by {_NVFP4_GEMM_ALIGNMENT}")


def _grouped_mm_quantized(
    activations: _NVFP4Tensor,
    weight: _NVFP4Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Dispatch quantized operands to the owned SM100 grouped GEMM."""

    if activations.global_scales.numel() != activations.data.shape[0]:
        raise ValueError("activation global scales must contain one value per token row")
    if weight.global_scales.numel() != offsets.numel():
        raise ValueError("weight global scales must contain one value per expert")
    return _grouped_mm_kernel(
        activations.data,
        weight.data.transpose(-2, -1),
        activations.block_scales,
        weight.block_scales,
        offsets,
        activations.global_scales,
        weight.global_scales,
    )


def _forward(
    matrix: torch.Tensor,
    weight: torch.Tensor,
    offsets: torch.Tensor,
) -> tuple[torch.Tensor, _NVFP4Tensor, _NVFP4Tensor]:
    activations_nvfp4 = quantize_activations(matrix, offsets)
    weight_nvfp4 = quantize_weights(weight)
    return (
        _grouped_mm_quantized(activations_nvfp4, weight_nvfp4, offsets),
        activations_nvfp4,
        weight_nvfp4,
    )


class _GroupedNVFP4MM(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        matrix: torch.Tensor,
        weight: torch.Tensor,
        offsets: torch.Tensor,
        backward: NVFP4Backward,
    ) -> torch.Tensor:
        output, activations_nvfp4, weight_nvfp4 = _forward(matrix, weight, offsets)
        ctx.backward_recipe = backward
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

        grad_matrix = grad_weight = None
        if ctx.backward_recipe == "bf16":
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
            matrix = weight = None

        if ctx.needs_input_grad[0]:
            if weight is None:
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
            if matrix is None:
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
    offsets: torch.Tensor,
    backward: NVFP4Backward = "bf16_dequantized",
) -> torch.Tensor:
    """Grouped NVFP4 forward with a selectable BF16 backward recipe.

    ``matrix`` has shape ``[M, K]``, ``weight`` has shape ``[G, K, N]``, and
    ``offsets`` contains the cumulative row count for each of the ``G`` groups.

    ``bf16`` saves the original BF16 operands for backward, while
    ``bf16_dequantized`` saves the forward's packed NVFP4 operands and
    dequantizes them to BF16 when backward runs.
    """

    if backward not in ("bf16", "bf16_dequantized"):
        raise ValueError("backward must be 'bf16' or 'bf16_dequantized'")
    if matrix.ndim != 2 or weight.ndim != 3:
        raise ValueError("matrix must be 2D and weight must be 3D")
    if weight.shape[0] != offsets.numel():
        raise ValueError("weight and offsets must have the same number of groups")
    if matrix.shape[1] != weight.shape[1]:
        raise ValueError("matrix and weight contraction dimensions must match")
    _check_matrix(matrix, "matrix")
    if weight.device != matrix.device or weight.dtype != matrix.dtype:
        raise ValueError("weight must have the same CUDA device and dtype as matrix")
    return _GroupedNVFP4MM.apply(matrix, weight, offsets, backward)
