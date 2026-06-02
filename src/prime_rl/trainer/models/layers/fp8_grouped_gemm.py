from __future__ import annotations

try:
    import deep_gemm
except ImportError:
    deep_gemm = None  # CPU-only environments don't ship deep_gemm; FP8 paths
    # are GPU-only at runtime, so leaving the symbol None is safe — only the
    # autograd Function bodies below actually call into it.
import torch

from prime_rl.trainer.models.kernels.fp8_utils import (
    GROUP_ALIGNMENT,
    build_grouped_layout,
    grouped_per_block_cast_to_fp8_triton,
    grouped_per_channel_cast_to_fp8_sm90_kmajor_triton,
    grouped_per_token_cast_to_fp8_triton,
    unpack_rows_triton,
)


def _compute_grad_weight(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    weight_shape: torch.Size,
    padded_total_m: int,
    block_to_group: torch.Tensor,
    ks_tensor: torch.Tensor,
    starts_tensor: torch.Tensor,
    actual_ms_tensor: torch.Tensor,
    block_starts_tensor: torch.Tensor,
    aligned_ms: list[int],
) -> torch.Tensor:
    x_k_major = grouped_per_channel_cast_to_fp8_sm90_kmajor_triton(
        x,
        padded_total_m,
        block_to_group,
        starts_tensor,
        actual_ms_tensor,
        ks_tensor,
        block_starts_tensor,
        False,
        GROUP_ALIGNMENT,
    )
    dy_k_major = grouped_per_channel_cast_to_fp8_sm90_kmajor_triton(
        grad_output,
        padded_total_m,
        block_to_group,
        starts_tensor,
        actual_ms_tensor,
        ks_tensor,
        block_starts_tensor,
        False,
        GROUP_ALIGNMENT,
    )
    grad_weight = torch.zeros(weight_shape, device=x.device, dtype=torch.float32)
    deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
        x_k_major,
        dy_k_major,
        grad_weight,
        aligned_ms,
        ks_tensor,
        grad_weight,
    )
    return grad_weight.to(torch.bfloat16)


class _GroupedFP8Gemm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        offs: torch.Tensor,
    ) -> torch.Tensor:
        (
            total_m,
            padded_total_m,
            grouped_layout,
            block_to_group,
            ks_tensor,
            starts_tensor,
            actual_ms_tensor,
            block_starts_tensor,
        ) = build_grouped_layout(offs, total_m=x.size(0))

        x_fp8 = grouped_per_token_cast_to_fp8_triton(
            x,
            padded_total_m,
            block_to_group,
            starts_tensor,
            actual_ms_tensor,
            block_starts_tensor,
            False,
            GROUP_ALIGNMENT,
        )
        weight_fp8 = grouped_per_block_cast_to_fp8_triton(
            weight.transpose(1, 2),
            False,
            GROUP_ALIGNMENT,
        )

        out_padded = torch.empty(
            (padded_total_m, weight.size(2)),
            device=x.device,
            dtype=x.dtype,
        )
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            x_fp8,
            weight_fp8,
            out_padded,
            grouped_layout,
            use_psum_layout=False,
        )
        out = unpack_rows_triton(
            out_padded,
            total_m,
            block_to_group,
            starts_tensor,
            actual_ms_tensor,
            block_starts_tensor,
        )

        ctx.padded_total_m = padded_total_m
        ctx.aligned_ms = ks_tensor.tolist()
        ctx.save_for_backward(
            x,
            weight,
            grouped_layout,
            block_to_group,
            ks_tensor,
            starts_tensor,
            actual_ms_tensor,
            block_starts_tensor,
        )
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            x,
            weight,
            grouped_layout,
            block_to_group,
            ks_tensor,
            starts_tensor,
            actual_ms_tensor,
            block_starts_tensor,
        ) = ctx.saved_tensors
        padded_total_m = ctx.padded_total_m
        aligned_ms = ctx.aligned_ms
        grad_output = grad_output.contiguous()

        grad_x = grad_weight = None

        if ctx.needs_input_grad[1]:
            grad_weight = _compute_grad_weight(
                x,
                grad_output,
                weight.shape,
                padded_total_m,
                block_to_group,
                ks_tensor,
                starts_tensor,
                actual_ms_tensor,
                block_starts_tensor,
                aligned_ms,
            )

        if ctx.needs_input_grad[0]:
            dy_fp8 = grouped_per_token_cast_to_fp8_triton(
                grad_output,
                padded_total_m,
                block_to_group,
                starts_tensor,
                actual_ms_tensor,
                block_starts_tensor,
                False,
                GROUP_ALIGNMENT,
            )
            weight_dx_fp8 = grouped_per_block_cast_to_fp8_triton(
                weight,
                False,
                GROUP_ALIGNMENT,
            )
            grad_x_padded = torch.empty(
                (padded_total_m, weight.size(1)),
                device=grad_output.device,
                dtype=grad_output.dtype,
            )
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                dy_fp8,
                weight_dx_fp8,
                grad_x_padded,
                grouped_layout,
                use_psum_layout=False,
            )
            grad_x = unpack_rows_triton(
                grad_x_padded,
                x.size(0),
                block_to_group,
                starts_tensor,
                actual_ms_tensor,
                block_starts_tensor,
            )

        return grad_x, grad_weight, None


def grouped_fp8_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    offs: torch.Tensor,
) -> torch.Tensor:
    """FP8 grouped GEMM, drop-in replacement for torch._grouped_mm.

    Args:
        x: (M, K) concatenated token activations in bfloat16.
        weight: (G, K, N) expert weights in bfloat16.
        offs: (G,) int32 cumulative token counts per expert.

    Returns:
        (M, N) output tensor in bfloat16.
    """
    return _GroupedFP8Gemm.apply(x, weight, offs)
