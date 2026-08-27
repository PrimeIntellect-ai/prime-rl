from __future__ import annotations

import torch
from torch import nn

try:
    import deep_gemm
except ImportError:
    deep_gemm = None  # CPU-only environments don't ship deep_gemm; FP8 paths
    # are GPU-only at runtime, so leaving the symbol None is safe — only the
    # autograd Function bodies below actually call into it.

from prime_rl.trainer.models.kernels.fp8_utils import (
    per_block_cast_to_fp8_tp_triton,
    per_block_cast_to_fp8_triton,
    per_token_cast_to_fp8_tp_triton,
    per_token_cast_to_fp8_triton,
    ue8m0_for_device,
)


class _FP8BlockwiseMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, block_size, out_dtype=torch.bfloat16):
        x_shape = x.shape
        x_2d = x.reshape(-1, x_shape[-1]).contiguous()
        use_ue8m0 = ue8m0_for_device(x.device)
        x_fp8 = per_token_cast_to_fp8_triton(x_2d, use_ue8m0, block_size)
        weight_fp8 = per_block_cast_to_fp8_triton(weight, use_ue8m0, block_size)

        out = torch.empty((x_2d.size(0), weight.size(0)), device=x.device, dtype=out_dtype)
        deep_gemm.fp8_gemm_nt(x_fp8, weight_fp8, out)

        ctx.save_for_backward(x_2d, weight)
        ctx.x_shape = x_shape
        ctx.block_size = block_size
        return out.reshape(*x_shape[:-1], out.size(-1))

    @staticmethod
    def backward(ctx, grad_output):
        x_2d, weight = ctx.saved_tensors
        block_size = ctx.block_size
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
        use_ue8m0 = ue8m0_for_device(grad_output.device)

        grad_x = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_output_fp8 = per_token_cast_to_fp8_triton(grad_output_2d, use_ue8m0, block_size)
            weight_dx_fp8 = per_block_cast_to_fp8_tp_triton(weight, use_ue8m0, block_size)
            grad_x_2d = torch.empty_like(x_2d)
            deep_gemm.fp8_gemm_nt(grad_output_fp8, weight_dx_fp8, grad_x_2d)
            grad_x = grad_x_2d.reshape(ctx.x_shape)

        if ctx.needs_input_grad[1]:
            # deep_gemm.fp8_gemm_nt with recipe=(1, 1, 128) requires the K (token)
            # dim to be a multiple of 128. Zero-pad along the token axis so non-
            # aligned per-rank batches (from sequence packing) don't trip the kernel.
            M_tok = grad_output_2d.size(0)
            M_pad = (M_tok + block_size - 1) // block_size * block_size
            if M_pad != M_tok:
                pad_rows = M_pad - M_tok
                grad_output_2d_padded = torch.nn.functional.pad(grad_output_2d, (0, 0, 0, pad_rows))
                x_2d_padded = torch.nn.functional.pad(x_2d, (0, 0, 0, pad_rows))
            else:
                grad_output_2d_padded = grad_output_2d
                x_2d_padded = x_2d
            grad_output_t_fp8 = per_token_cast_to_fp8_tp_triton(grad_output_2d_padded, use_ue8m0, block_size)
            x_t_fp8 = per_token_cast_to_fp8_tp_triton(x_2d_padded, use_ue8m0, block_size)
            grad_weight_fp32 = torch.zeros_like(weight, dtype=torch.float32)
            deep_gemm.fp8_gemm_nt(
                grad_output_t_fp8,
                x_t_fp8,
                grad_weight_fp32,
                c=grad_weight_fp32,
                recipe=(1, 1, 128),
            )
            grad_weight = grad_weight_fp32.to(weight.dtype)

        return grad_x, grad_weight, None, None


class Float8BlockwiseLinear(nn.Linear):
    """nn.Linear replacement that uses FP8 blockwise matmul via DeepGEMM.

    Requires:
    - SM90 (Hopper) or SM100 (Blackwell) GPU
    - bfloat16 inputs/weights
    - No bias
    - in_features and out_features divisible by 128
    """

    def __init__(self, *args, block_size: int = 128, dtype=torch.bfloat16, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _FP8BlockwiseMM.apply(x, self.weight, self.block_size, torch.bfloat16)

    @classmethod
    def from_linear(cls, mod: nn.Linear) -> "Float8BlockwiseLinear":
        """Convert an existing nn.Linear to Float8BlockwiseLinear."""
        with torch.device("meta"):
            new_mod = cls(
                mod.in_features,
                mod.out_features,
                bias=mod.bias is not None,
            )
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        return new_mod
