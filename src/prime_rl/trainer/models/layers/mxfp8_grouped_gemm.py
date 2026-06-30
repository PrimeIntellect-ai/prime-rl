from __future__ import annotations

import torch

try:
    from torchao.prototype.moe_training import _to_mxfp8_then_scaled_grouped_mm
    from torchao.prototype.mx_formats.config import KernelPreference, ScaleCalculationMode
except ImportError:
    # torchao is an x86_64-only optional dependency; the MXFP8 grouped GEMM is
    # GPU/SM100-only at runtime, so leaving these None is safe.
    KernelPreference = ScaleCalculationMode = None
    _to_mxfp8_then_scaled_grouped_mm = None

_SCALING_MODES = {"rceil": "RCEIL", "floor": "FLOOR"}
_KERNELS = {"auto": "AUTO", "emulated": "EMULATED"}


def mxfp8_grouped_gemm(
    x: torch.Tensor,
    w_t: torch.Tensor,
    offsets: torch.Tensor,
    scaling_mode: str = "rceil",
    kernel: str = "auto",
) -> torch.Tensor:
    """Differentiable MXFP8 grouped GEMM, a drop-in replacement for ``torch._grouped_mm``.

    Args:
        x: 2D activations of shape ``(total_tokens, K)``.
        w_t: 3D expert weights already transposed to ``(num_experts, K, N)`` (per-group
            column-major), i.e. ``weight.transpose(-2, -1)``.
        offsets: int32 cumulative token counts per expert (end index of each group).
    """
    return _to_mxfp8_then_scaled_grouped_mm(
        x,
        w_t,
        offs=offsets,
        kernel_preference=getattr(KernelPreference, _KERNELS[kernel]),
        scale_calculation_mode=getattr(ScaleCalculationMode, _SCALING_MODES[scaling_mode]),
        pad_token_groups_for_grouped_mm=True,
    )
