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

# torchao's SM100 MXFP8 grouped-GEMM CUDA kernels (padding, scale rearrange) are hard-capped
# at 32 token groups per call. Models with more experts than this (e.g. 128-expert Qwen3-30B
# without expert parallelism) must be split into <=32-expert chunks.
_MAX_GROUPS = 32


def _mxfp8_grouped_gemm_call(x, w_t, offsets, kernel_preference, scale_calculation_mode):
    # pad_token_groups_for_grouped_mm=True lets the op pad each token group to a multiple of
    # 32 internally, so both the forward and the weight-gradient backward see aligned groups.
    return _to_mxfp8_then_scaled_grouped_mm(
        x,
        w_t,
        offs=offsets,
        kernel_preference=kernel_preference,
        scale_calculation_mode=scale_calculation_mode,
        pad_token_groups_for_grouped_mm=True,
    )


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

    torchao's SM100 grouped-GEMM kernels are capped at 32 token groups, so for models with
    more than 32 experts (e.g. 128-expert Qwen3-30B-A3B without expert parallelism) the GEMM
    is split into contiguous <=32-expert chunks and concatenated. This needs a single host
    sync to read the group offsets; chunk boundaries fall exactly on expert boundaries, so
    each chunk is an independent, differentiable grouped GEMM.
    """
    kp = getattr(KernelPreference, _KERNELS[kernel])
    scm = getattr(ScaleCalculationMode, _SCALING_MODES[scaling_mode])
    num_groups = w_t.shape[0]
    if num_groups <= _MAX_GROUPS:
        return _mxfp8_grouped_gemm_call(x, w_t, offsets, kp, scm)

    ends = offsets.tolist()
    outs = []
    tok_start = 0
    for cs in range(0, num_groups, _MAX_GROUPS):
        ce = min(cs + _MAX_GROUPS, num_groups)
        tok_end = ends[ce - 1]
        local_offsets = offsets[cs:ce] - tok_start
        outs.append(_mxfp8_grouped_gemm_call(x[tok_start:tok_end], w_t[cs:ce], local_offsets, kp, scm))
        tok_start = tok_end
    out = torch.cat(outs, dim=0)
    # The @expert_parallel decorator pads x with trailing rows beyond the last group offset;
    # torch._grouped_mm returns those rows as zeros. Match that so the unpermute step, which
    # indexes the full padded length, gets a correctly shaped output.
    if out.shape[0] < x.shape[0]:
        out = torch.cat([out, out.new_zeros(x.shape[0] - out.shape[0], out.shape[1])], dim=0)
    return out
