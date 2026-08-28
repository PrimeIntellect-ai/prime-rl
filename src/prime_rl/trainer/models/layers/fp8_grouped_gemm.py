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


@torch.library.custom_op("prime_rl::grouped_fp8_gemm", mutates_args=())
def _grouped_fp8_gemm(x: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    (
        total_m,
        padded_total_m,
        grouped_layout,
        block_to_group,
        _,
        starts_tensor,
        actual_ms_tensor,
        block_starts_tensor,
    ) = build_grouped_layout(offs, total_m=x.size(0))

    use_ue8m0 = ue8m0_for_device(x.device)
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

    out_padded = torch.empty((padded_total_m, weight.size(2)), device=x.device, dtype=x.dtype)
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        x_fp8,
        weight_fp8,
        out_padded,
        grouped_layout,
        use_psum_layout=False,
    )
    return unpack_rows_triton(
        out_padded,
        total_m,
        block_to_group,
        starts_tensor,
        actual_ms_tensor,
        block_starts_tensor,
    )


@_grouped_fp8_gemm.register_fake
def _grouped_fp8_gemm_fake(x: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[2]))


@torch.library.custom_op("prime_rl::grouped_fp8_gemm_backward", mutates_args=())
def _grouped_fp8_gemm_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    offs: torch.Tensor,
    needs_grad_x: bool,
    needs_grad_weight: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    (
        _,
        padded_total_m,
        grouped_layout,
        block_to_group,
        ks_tensor,
        starts_tensor,
        actual_ms_tensor,
        block_starts_tensor,
    ) = build_grouped_layout(offs, total_m=x.size(0))
    grad_output = grad_output.contiguous()
    grad_x = torch.empty_like(x)
    grad_weight = torch.empty_like(weight)

    if needs_grad_weight:
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
            ks_tensor.tolist(),
        )

    if needs_grad_x:
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
        weight_dx_fp8 = grouped_per_block_cast_to_fp8_triton(weight, use_ue8m0, GROUP_ALIGNMENT)
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

    return grad_x, grad_weight


@_grouped_fp8_gemm_backward.register_fake
def _grouped_fp8_gemm_backward_fake(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    offs: torch.Tensor,
    needs_grad_x: bool,
    needs_grad_weight: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty_like(weight)


def _grouped_fp8_gemm_setup_context(ctx, inputs, output) -> None:
    x, weight, offs = inputs
    ctx.save_for_backward(x, weight, offs)


def _grouped_fp8_gemm_autograd_backward(ctx, grad_output: torch.Tensor):
    x, weight, offs = ctx.saved_tensors
    needs_grad_x, needs_grad_weight, _ = ctx.needs_input_grad
    grad_x, grad_weight = _grouped_fp8_gemm_backward(
        grad_output,
        x.detach(),
        weight.detach(),
        offs,
        needs_grad_x,
        needs_grad_weight,
    )
    return grad_x if needs_grad_x else None, grad_weight if needs_grad_weight else None, None


_grouped_fp8_gemm.register_autograd(
    _grouped_fp8_gemm_autograd_backward,
    setup_context=_grouped_fp8_gemm_setup_context,
)


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
    return _grouped_fp8_gemm(x, weight, offs)
