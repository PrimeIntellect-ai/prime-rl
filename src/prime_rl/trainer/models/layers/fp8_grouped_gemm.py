from __future__ import annotations

from collections.abc import Callable

import torch

from prime_rl.trainer.models.kernels.fp8_utils import (
    GROUP_ALIGNMENT,
    build_grouped_layout,
    grouped_per_block_cast_to_fp8_triton,
    grouped_per_channel_cast_to_fp8_rowmajor_triton,
    grouped_per_channel_cast_to_fp8_sm90_kmajor_triton,
    grouped_per_token_cast_to_fp8_triton,
    ue8m0_for_device,
)


def _is_transposed_contiguous(weight: torch.Tensor) -> bool:
    return not weight.is_contiguous() and weight.transpose(1, 2).is_contiguous()


def _empty_grad_weight(weight: torch.Tensor, grad_weight_transposed: bool) -> torch.Tensor:
    if grad_weight_transposed:
        return weight.new_empty(weight.transpose(1, 2).shape).transpose(1, 2)
    return weight.new_empty(weight.shape)


def _grouped_layout(x: torch.Tensor, offs: torch.Tensor):
    assert x.size(0) % GROUP_ALIGNMENT == 0, f"expected rows padded to {GROUP_ALIGNMENT}, got {x.size(0)}"
    torch._assert_async(torch.all(offs % GROUP_ALIGNMENT == 0))
    return build_grouped_layout(offs, x.size(0))


def _compute_grad_weight(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    padded_total_m: int,
    block_to_group: torch.Tensor,
    ks_tensor: torch.Tensor,
    starts_tensor: torch.Tensor,
    actual_ms_tensor: torch.Tensor,
    block_starts_tensor: torch.Tensor,
    aligned_ms: list[int],
    grad_weight_transposed: bool,
) -> torch.Tensor:
    import deep_gemm

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

    return _grouped_wgrad_gemm(
        grouped_weight_grad, x_fp8, dy_fp8, weight, aligned_ms, ks_tensor, grad_weight_transposed
    )


def _grouped_wgrad_gemm(
    grouped_weight_grad: Callable,
    x_fp8: tuple[torch.Tensor, torch.Tensor],
    dy_fp8: tuple[torch.Tensor, torch.Tensor],
    weight: torch.Tensor,
    aligned_ms: list[int],
    ks_tensor: torch.Tensor,
    grad_weight_transposed: bool,
) -> torch.Tensor:
    if grad_weight_transposed:
        lhs, rhs, grad_shape = dy_fp8, x_fp8, weight.transpose(1, 2).shape
    else:
        lhs, rhs, grad_shape = x_fp8, dy_fp8, weight.shape
    grad_weight = torch.zeros(grad_shape, device=weight.device, dtype=torch.float32)
    grouped_weight_grad(
        lhs,
        rhs,
        grad_weight,
        aligned_ms,
        ks_tensor,
        grad_weight,
    )
    grad_weight = grad_weight.to(weight.dtype)
    return grad_weight.transpose(1, 2) if grad_weight_transposed else grad_weight


def _grouped_wgrad(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    block_to_group: torch.Tensor,
    ks_tensor: torch.Tensor,
    starts_tensor: torch.Tensor,
    actual_ms_tensor: torch.Tensor,
    block_starts_tensor: torch.Tensor,
    grad_weight_transposed: bool,
) -> torch.Tensor:
    aligned_ms = ks_tensor.tolist()
    grouped_total_m = sum(aligned_ms)
    return _compute_grad_weight(
        x,
        grad_output,
        weight,
        grouped_total_m,
        block_to_group[: grouped_total_m // GROUP_ALIGNMENT],
        ks_tensor,
        starts_tensor,
        actual_ms_tensor,
        block_starts_tensor,
        aligned_ms,
        grad_weight_transposed,
    )


@torch.library.custom_op("prime_rl::grouped_fp8_gemm", mutates_args=())
def _grouped_fp8_gemm(x: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    import deep_gemm

    (
        grouped_layout,
        block_to_group,
        _,
        starts_tensor,
        actual_ms_tensor,
        block_starts_tensor,
    ) = _grouped_layout(x, offs)

    use_ue8m0 = ue8m0_for_device(x.device)
    x_fp8 = grouped_per_token_cast_to_fp8_triton(
        x,
        x.size(0),
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

    out = torch.empty((x.size(0), weight.size(2)), device=x.device, dtype=x.dtype)
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        x_fp8,
        weight_fp8,
        out,
        grouped_layout,
        use_psum_layout=False,
    )
    return out


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
    grad_weight_transposed: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    (
        grouped_layout,
        block_to_group,
        ks_tensor,
        starts_tensor,
        actual_ms_tensor,
        block_starts_tensor,
    ) = _grouped_layout(x, offs)
    grad_output = grad_output.contiguous()

    if needs_grad_weight:
        grad_weight = _grouped_wgrad(
            x,
            grad_output,
            weight,
            block_to_group,
            ks_tensor,
            starts_tensor,
            actual_ms_tensor,
            block_starts_tensor,
            grad_weight_transposed,
        )
    else:
        grad_weight = _empty_grad_weight(weight, grad_weight_transposed)

    if needs_grad_x:
        import deep_gemm

        use_ue8m0 = ue8m0_for_device(grad_output.device)
        dy_fp8 = grouped_per_token_cast_to_fp8_triton(
            grad_output,
            x.size(0),
            block_to_group,
            starts_tensor,
            actual_ms_tensor,
            block_starts_tensor,
            use_ue8m0,
            GROUP_ALIGNMENT,
        )
        weight_dx_fp8 = grouped_per_block_cast_to_fp8_triton(weight, use_ue8m0, GROUP_ALIGNMENT)
        grad_x = torch.empty(x.shape, device=grad_output.device, dtype=grad_output.dtype)
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            dy_fp8,
            weight_dx_fp8,
            grad_x,
            grouped_layout,
            use_psum_layout=False,
        )
    else:
        grad_x = x.new_empty(x.shape)

    return grad_x, grad_weight


@_grouped_fp8_gemm_backward.register_fake
def _grouped_fp8_gemm_backward_fake(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    offs: torch.Tensor,
    needs_grad_x: bool,
    needs_grad_weight: bool,
    grad_weight_transposed: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return x.new_empty(x.shape), _empty_grad_weight(weight, grad_weight_transposed)


def _grouped_fp8_gemm_setup_context(ctx, inputs, output) -> None:
    x, weight, offs = inputs
    ctx.save_for_backward(x, weight, offs)
    ctx.grad_weight_transposed = _is_transposed_contiguous(weight)


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
        ctx.grad_weight_transposed,
    )
    return grad_x if needs_grad_x else None, grad_weight if needs_grad_weight else None, None


_grouped_fp8_gemm.register_autograd(
    _grouped_fp8_gemm_autograd_backward,
    setup_context=_grouped_fp8_gemm_setup_context,
)


PREQUANT_SM90_ONLY = (
    "the torch-op quantizers emit fp32 scales and the SM90 K-major wgrad layout, which DeepGEMM accepts on SM90 only"
)


@torch.library.custom_op("prime_rl::grouped_fp8_gemm_prequant", mutates_args=())
def _grouped_fp8_gemm_prequant(
    x_fp8: torch.Tensor, x_scale: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    """`prime_rl::grouped_fp8_gemm` of an input already cast to FP8 in 1x128 groups, shaped `[rows, in_features / 128, 128]`."""
    import deep_gemm

    assert not ue8m0_for_device(weight.device), PREQUANT_SM90_ONLY
    grouped_layout = _grouped_layout(x_fp8, offs)[0]
    weight_fp8 = grouped_per_block_cast_to_fp8_triton(weight.transpose(1, 2), False, GROUP_ALIGNMENT)
    out = torch.empty((x_fp8.size(0), weight.size(2)), device=x_fp8.device, dtype=torch.bfloat16)
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (x_fp8.flatten(1), x_scale), weight_fp8, out, grouped_layout, use_psum_layout=False
    )
    return out


@_grouped_fp8_gemm_prequant.register_fake
def _grouped_fp8_gemm_prequant_fake(
    x_fp8: torch.Tensor, x_scale: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    return x_fp8.new_empty((x_fp8.shape[0], weight.shape[2]), dtype=torch.bfloat16)


@torch.library.custom_op("prime_rl::grouped_fp8_gemm_dgrad_prequant", mutates_args=())
def _grouped_fp8_gemm_dgrad_prequant(
    grad_output_fp8: torch.Tensor, grad_output_scale: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    """Input gradient of `prime_rl::grouped_fp8_gemm` from an output gradient already cast to FP8 in 1x128 groups, shaped `[rows, out_features / 128, 128]`."""
    import deep_gemm

    assert not ue8m0_for_device(weight.device), PREQUANT_SM90_ONLY
    grouped_layout = _grouped_layout(grad_output_fp8, offs)[0]
    weight_dx_fp8 = grouped_per_block_cast_to_fp8_triton(weight, False, GROUP_ALIGNMENT)
    grad_x = torch.empty((grad_output_fp8.size(0), weight.size(1)), device=weight.device, dtype=torch.bfloat16)
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (grad_output_fp8.flatten(1), grad_output_scale), weight_dx_fp8, grad_x, grouped_layout, use_psum_layout=False
    )
    return grad_x


@_grouped_fp8_gemm_dgrad_prequant.register_fake
def _grouped_fp8_gemm_dgrad_prequant_fake(
    grad_output_fp8: torch.Tensor, grad_output_scale: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    return grad_output_fp8.new_empty((grad_output_fp8.shape[0], weight.shape[1]), dtype=torch.bfloat16)


@torch.library.custom_op("prime_rl::grouped_fp8_gemm_wgrad_prequant", mutates_args=())
def _grouped_fp8_gemm_wgrad_prequant(
    x_t_fp8: torch.Tensor,
    x_t_scale: torch.Tensor,
    grad_output_t_fp8: torch.Tensor,
    grad_output_t_scale: torch.Tensor,
    weight: torch.Tensor,
    offs: torch.Tensor,
    grad_weight_transposed: bool,
) -> torch.Tensor:
    """Weight gradient of `prime_rl::grouped_fp8_gemm` from input and output gradient cast by `quant_grouped_128x1_kmajor` over all rows."""
    import deep_gemm

    assert not ue8m0_for_device(weight.device), PREQUANT_SM90_ONLY
    torch._assert_async(torch.all(offs % GROUP_ALIGNMENT == 0))
    ks_tensor = build_grouped_layout(offs, x_t_scale.size(1) * GROUP_ALIGNMENT)[2]
    aligned_ms = ks_tensor.tolist()
    grouped_rows = sum(aligned_ms)
    grouped_blocks = grouped_rows // GROUP_ALIGNMENT
    x_t = (x_t_fp8[: grouped_rows * weight.size(1)], x_t_scale[:, :grouped_blocks])
    grad_output_t = (grad_output_t_fp8[: grouped_rows * weight.size(2)], grad_output_t_scale[:, :grouped_blocks])
    return _grouped_wgrad_gemm(
        deep_gemm.k_grouped_fp8_gemm_nt_contiguous,
        x_t,
        grad_output_t,
        weight,
        aligned_ms,
        ks_tensor,
        grad_weight_transposed,
    )


@_grouped_fp8_gemm_wgrad_prequant.register_fake
def _grouped_fp8_gemm_wgrad_prequant_fake(
    x_t_fp8: torch.Tensor,
    x_t_scale: torch.Tensor,
    grad_output_t_fp8: torch.Tensor,
    grad_output_t_scale: torch.Tensor,
    weight: torch.Tensor,
    offs: torch.Tensor,
    grad_weight_transposed: bool,
) -> torch.Tensor:
    return _empty_grad_weight(weight, grad_weight_transposed)


@torch.library.custom_op("prime_rl::grouped_fp8_gemm_wgrad", mutates_args=())
def _grouped_fp8_gemm_wgrad(
    grad_output: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor, grad_weight_transposed: bool
) -> torch.Tensor:
    """Weight gradient of `prime_rl::grouped_fp8_gemm` from bf16 tensors, with the Triton per-channel quantizers inside."""
    _, block_to_group, ks_tensor, starts_tensor, actual_ms_tensor, block_starts_tensor = _grouped_layout(x, offs)
    return _grouped_wgrad(
        x,
        grad_output.contiguous(),
        weight,
        block_to_group,
        ks_tensor,
        starts_tensor,
        actual_ms_tensor,
        block_starts_tensor,
        grad_weight_transposed,
    )


@_grouped_fp8_gemm_wgrad.register_fake
def _grouped_fp8_gemm_wgrad_fake(
    grad_output: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor, grad_weight_transposed: bool
) -> torch.Tensor:
    return _empty_grad_weight(weight, grad_weight_transposed)


def grouped_fp8_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    offs: torch.Tensor,
) -> torch.Tensor:
    """FP8 grouped GEMM, drop-in replacement for torch._grouped_mm.

    Args:
        x: (M, K) concatenated token activations in bfloat16, M a multiple of 128.
        weight: (G, K, N) expert weights in bfloat16.
        offs: (G,) int32 cumulative token counts per expert, each group a multiple of 128 rows.

    Returns:
        (M, N) output tensor in bfloat16. Rows past offs[-1] are left uninitialized.
    """
    return _grouped_fp8_gemm(x, weight, offs)
