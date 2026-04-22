"""FP8 e4m3 blockwise quantization — PyTorch impl (identical to main's
``quantize_to_fp8_blockwise``). Diagnostic swap in iter18 to rule out any
subtle Triton-kernel difference vs the NCCL+FP8 path on main.
"""

import torch

BLOCK_SIZE = 128


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def _quantize_2d_pytorch(weight: torch.Tensor, block_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Bit-identical port of main's ``quantize_to_fp8_blockwise``."""
    if weight.ndim != 2:
        raise ValueError(f"FP8 quantization expects a 2D tensor, got shape={tuple(weight.shape)}")

    rows, cols = weight.shape
    pad_rows = (block_size - rows % block_size) % block_size
    pad_cols = (block_size - cols % block_size) % block_size

    if pad_rows or pad_cols:
        padded = torch.zeros(
            rows + pad_rows,
            cols + pad_cols,
            dtype=weight.dtype,
            device=weight.device,
        )
        padded[:rows, :cols] = weight
    else:
        padded = weight.contiguous()

    padded_rows, padded_cols = padded.shape
    blocks = padded.view(
        padded_rows // block_size,
        block_size,
        padded_cols // block_size,
        block_size,
    ).permute(0, 2, 1, 3)

    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    max_abs = blocks.float().abs().amax(dim=(2, 3))
    scales = (max_abs / fp8_max).clamp(min=1e-12)
    blocks_fp8 = (blocks.float() / scales[:, :, None, None]).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)

    quantized = blocks_fp8.permute(0, 2, 1, 3).reshape(padded_rows, padded_cols)[:rows, :cols].contiguous()
    return quantized, scales.float().contiguous()


def fp8_block_quantize(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    sf: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """2D FP8 blockwise quantize — matches main's PyTorch impl exactly.

    If ``out``/``sf`` are provided, results are copied into them.
    """
    q, s = _quantize_2d_pytorch(x, BLOCK_SIZE)
    if out is not None:
        out.copy_(q)
    if sf is not None:
        sf.copy_(s)
    return q, s


def grouped_fp8_block_quantize(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    sf: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """3D grouped FP8 block quantize — per-expert loop of ``_quantize_2d_pytorch``.

    Mirrors main's per-expert loop in ``convert_tt_layer_to_vllm_kernel``
    (loop over ``w13[expert_idx]`` then ``torch.stack``). If ``out``/``sf``
    are provided, results are copied in place.
    """
    if x.ndim != 3:
        raise ValueError(f"grouped_fp8_block_quantize expects 3D, got shape={tuple(x.shape)}")
    groups, rows, cols = x.shape
    q_accum = torch.empty(
        (groups, rows, cols), dtype=torch.float8_e4m3fn, device=x.device
    )
    s_accum = torch.empty(
        (groups, ceil_div(rows, BLOCK_SIZE), ceil_div(cols, BLOCK_SIZE)),
        dtype=torch.float32,
        device=x.device,
    )
    for g in range(groups):
        q_g, s_g = _quantize_2d_pytorch(x[g], BLOCK_SIZE)
        q_accum[g] = q_g
        s_accum[g] = s_g
    if out is not None:
        out.copy_(q_accum)
    if sf is not None:
        sf.copy_(s_accum)
    return q_accum, s_accum
