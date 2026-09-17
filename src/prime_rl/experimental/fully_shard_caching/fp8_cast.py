"""Blockwise fp8 cast launcher that writes into caller-supplied destinations."""

from __future__ import annotations

import torch

from prime_rl.trainer.models.kernels.fp8_utils import (
    GROUP_ALIGNMENT,
    _grouped_per_block_fp8_kernel,
    ceil_div,
)


def grouped_per_block_cast_to_fp8(
    x: torch.Tensor,
    use_ue8m0: bool,
    *,
    out: torch.Tensor | None = None,
    sf: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 3
    assert (out is None) == (sf is None)
    groups, rows, cols = x.shape
    scale_rows, scale_cols = ceil_div(rows, GROUP_ALIGNMENT), ceil_div(cols, GROUP_ALIGNMENT)
    if out is None:
        out = torch.empty((groups, rows, cols), device=x.device, dtype=torch.float8_e4m3fn)
        sf = torch.empty((groups, scale_rows, scale_cols), device=x.device, dtype=torch.float32)
    else:
        assert out.shape == (groups, rows, cols)
        assert out.dtype == torch.float8_e4m3fn
        assert out.device == x.device
        assert out.is_contiguous()
        assert sf.shape == (groups, scale_rows, scale_cols)
        assert sf.dtype == torch.float32
        assert sf.device == x.device
        assert sf.is_contiguous()
    grid = (groups, scale_rows, scale_cols)
    _grouped_per_block_fp8_kernel[grid](
        x,
        out,
        sf,
        groups,
        rows,
        cols,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        sf.stride(0),
        sf.stride(1),
        sf.stride(2),
        USE_UE8M0=use_ue8m0,
        BLOCK_M=GROUP_ALIGNMENT,
        BLOCK_N=GROUP_ALIGNMENT,
        num_warps=8,
    )
    return out, sf
