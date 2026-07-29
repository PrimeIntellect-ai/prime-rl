from __future__ import annotations

from dataclasses import dataclass

import torch

from prime_rl_kernels.nvfp4.quantize._extension import (
    _dequantize_activations,
    _dequantize_weights,
)
from prime_rl_kernels.nvfp4.quantize._extension import (
    _quantize_activations as _quantize_activations_kernel,
)
from prime_rl_kernels.nvfp4.quantize._extension import (
    _quantize_weights as _quantize_weights_kernel,
)

_NVFP4_GEMM_ALIGNMENT = 32


@dataclass(frozen=True)
class _NVFP4Tensor:
    data: torch.Tensor
    block_scales: torch.Tensor
    global_scales: torch.Tensor
    offsets: torch.Tensor | None

    def dequantize(self) -> torch.Tensor:
        packed = self.data.view(torch.uint8)
        if self.offsets is not None:
            return _dequantize_activations(
                packed,
                self.block_scales,
                self.global_scales,
                self.offsets,
            )
        return _dequantize_weights(
            packed,
            self.block_scales,
            self.global_scales,
        ).transpose(-2, -1)


def _check_blackwell() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("NVFP4 quantization requires CUDA")
    capability = torch.cuda.get_device_capability()
    if capability != (10, 0):
        raise RuntimeError(
            f"NVFP4 quantization currently requires SM100, but the current device is SM{capability[0]}{capability[1]}"
        )
    if not hasattr(torch, "float4_e2m1fn_x2"):
        raise RuntimeError("NVFP4 quantization requires PyTorch with float4_e2m1fn_x2 support")


def _check_bf16_cuda(tensor: torch.Tensor, name: str) -> None:
    if tensor.device.type != "cuda" or tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be a CUDA bfloat16 tensor")


def quantize_activations(matrix: torch.Tensor, offsets: torch.Tensor) -> _NVFP4Tensor:
    """Quantize ``[M, K]`` activations with one FP32 decode scale per token."""

    _check_blackwell()
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")
    _check_bf16_cuda(matrix, "matrix")
    if matrix.shape[1] % _NVFP4_GEMM_ALIGNMENT != 0:
        raise ValueError(f"matrix's contraction dimension must be divisible by {_NVFP4_GEMM_ALIGNMENT}")
    if not matrix.is_contiguous():
        matrix = matrix.contiguous()
    if offsets.ndim != 1 or offsets.numel() == 0 or offsets.dtype != torch.int32 or offsets.device != matrix.device:
        raise ValueError("offsets must be a non-empty 1D int32 tensor on the same CUDA device as matrix")
    if not offsets.is_contiguous():
        offsets = offsets.contiguous()

    packed, block_scales, global_scales = _quantize_activations_kernel(
        matrix,
        offsets,
    )
    return _NVFP4Tensor(
        data=packed.view(torch.float4_e2m1fn_x2),
        block_scales=block_scales,
        global_scales=global_scales,
        offsets=offsets,
    )


def quantize_weights(weight: torch.Tensor) -> _NVFP4Tensor:
    """Quantize logical ``[G, K, N]`` weights with one FP32 decode scale per expert."""

    _check_blackwell()
    if weight.ndim != 3:
        raise ValueError("weight must be 3D")
    _check_bf16_cuda(weight, "weight")

    _, contraction_size, output_size = weight.shape
    if contraction_size % _NVFP4_GEMM_ALIGNMENT != 0:
        raise ValueError(f"weight's contraction dimension must be divisible by {_NVFP4_GEMM_ALIGNMENT}")
    if output_size % 8 != 0:
        raise ValueError("weight's output dimension must be divisible by 8")

    weight_rows = weight.transpose(-2, -1)
    if not weight_rows.is_contiguous():
        weight_rows = weight_rows.contiguous()
    packed, block_scales, global_scales = _quantize_weights_kernel(
        weight_rows,
    )
    return _NVFP4Tensor(
        data=packed.view(torch.float4_e2m1fn_x2),
        block_scales=block_scales,
        global_scales=global_scales,
        offsets=None,
    )
