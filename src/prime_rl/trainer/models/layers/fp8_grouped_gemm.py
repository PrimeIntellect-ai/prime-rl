from __future__ import annotations

import torch

try:
    import deep_gemm
except ImportError:
    deep_gemm = None  # CPU-only environments don't ship deep_gemm; FP8 paths
    # are GPU-only at runtime, so leaving the symbol None is safe — only the
    # autograd Function bodies below actually call into it.

from prime_rl.trainer.models.kernels.fp8_utils import (
    GROUP_ALIGNMENT,
    build_grouped_layout,
    grouped_per_block_cast_to_fp8_triton,
    grouped_per_channel_cast_to_fp8_rowmajor_triton,
    grouped_per_channel_cast_to_fp8_sm90_kmajor_triton,
    grouped_per_token_cast_to_fp8_triton,
    ue8m0_for_device,
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
    is_sm100 = torch.cuda.get_device_capability(x.device)[0] >= 10
    if is_sm100:
        x_fp8 = grouped_per_channel_cast_to_fp8_rowmajor_triton(
            x,
            padded_total_m,
            block_to_group,
            starts_tensor,
            actual_ms_tensor,
            ks_tensor,
            block_starts_tensor,
            True,
            GROUP_ALIGNMENT,
        )
        dy_fp8 = grouped_per_channel_cast_to_fp8_rowmajor_triton(
            grad_output,
            padded_total_m,
            block_to_group,
            starts_tensor,
            actual_ms_tensor,
            ks_tensor,
            block_starts_tensor,
            True,
            GROUP_ALIGNMENT,
        )
        grouped_weight_grad = deep_gemm.k_grouped_fp8_gemm_tn_contiguous
    else:
        x_fp8 = grouped_per_channel_cast_to_fp8_sm90_kmajor_triton(
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
        dy_fp8 = grouped_per_channel_cast_to_fp8_sm90_kmajor_triton(
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
        grouped_weight_grad = deep_gemm.k_grouped_fp8_gemm_nt_contiguous

    grad_weight = torch.zeros(weight_shape, device=x.device, dtype=torch.float32)
    grouped_weight_grad(
        x_fp8,
        dy_fp8,
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
        layout: tuple | None = None,
        x_fp8_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if layout is None:
            layout = build_grouped_layout(offs, total_m=x.size(0))
        (
            total_m,
            padded_total_m,
            grouped_layout,
            block_to_group,
            ks_tensor,
            starts_tensor,
            actual_ms_tensor,
            block_starts_tensor,
        ) = layout

        use_ue8m0 = ue8m0_for_device(x.device)
        if x_fp8_cache is not None:
            x_fp8 = x_fp8_cache
        else:
            x_fp8 = grouped_per_token_cast_to_fp8_triton(
                x,
                padded_total_m,
                block_to_group,
                starts_tensor,
                actual_ms_tensor,
                block_starts_tensor,
                use_ue8m0,
                GROUP_ALIGNMENT,
            )
        weight_fp8 = grouped_per_block_cast_to_fp8_triton(
            weight.transpose(1, 2),
            use_ue8m0,
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
            use_ue8m0 = ue8m0_for_device(grad_output.device)
            dy_fp8 = grouped_per_token_cast_to_fp8_triton(
                grad_output,
                padded_total_m,
                block_to_group,
                starts_tensor,
                actual_ms_tensor,
                block_starts_tensor,
                use_ue8m0,
                GROUP_ALIGNMENT,
            )
            weight_dx_fp8 = grouped_per_block_cast_to_fp8_triton(
                weight,
                use_ue8m0,
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

        # grad_x, grad_weight, offs, layout, x_fp8_cache — the last three never need grad.
        return grad_x, grad_weight, None, None, None


def compute_grouped_layout(offs: torch.Tensor, total_m: int):
    return build_grouped_layout(offs, total_m=total_m)


def cast_grouped_input_to_fp8(x: torch.Tensor, layout: tuple) -> tuple[torch.Tensor, torch.Tensor]:
    (
        _total_m,
        padded_total_m,
        _grouped_layout,
        block_to_group,
        _ks_tensor,
        starts_tensor,
        actual_ms_tensor,
        block_starts_tensor,
    ) = layout
    use_ue8m0 = ue8m0_for_device(x.device)
    return grouped_per_token_cast_to_fp8_triton(
        x,
        padded_total_m,
        block_to_group,
        starts_tensor,
        actual_ms_tensor,
        block_starts_tensor,
        use_ue8m0,
        GROUP_ALIGNMENT,
    )


def grouped_fp8_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    offs: torch.Tensor,
    layout: tuple | None = None,
    x_fp8_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """FP8 grouped GEMM, drop-in replacement for torch._grouped_mm.

    Args:
        x: (M, K) concatenated token activations in bfloat16.
        weight: (G, K, N) expert weights in bfloat16.
        offs: (G,) int32 cumulative token counts per expert.
        layout: optional precomputed ``compute_grouped_layout(offs, total_m)`` result,
            to skip rebuilding the ragged-group layout when the caller already has one
            for this (offs, total_m) pair.
        x_fp8_cache: optional precomputed ``cast_grouped_input_to_fp8(x, layout)``
            result, to skip re-casting ``x`` when the caller already cast this exact
            tensor against this exact layout (e.g. for a sibling grouped_fp8_gemm call
            with a different weight but the same x).

    Returns:
        (M, N) output tensor in bfloat16.
    """
    return _GroupedFP8Gemm.apply(x, weight, offs, layout, x_fp8_cache)
