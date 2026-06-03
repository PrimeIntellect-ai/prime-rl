import torch
from torch import Tensor

BLOCK_SIZE = 128


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def quantize_to_fp8_blockwise(weight: Tensor, block_size: int = 128) -> tuple[Tensor, Tensor]:
    """Quantize a 2D tensor to FP8 e4m3 with per-block scales."""
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
    x: Tensor,
    out: Tensor | None = None,
    sf: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """2D FP8 blockwise quantize. Optionally writes into preallocated ``out``/``sf``."""
    q, s = quantize_to_fp8_blockwise(x, BLOCK_SIZE)
    if out is not None:
        out.copy_(q)
    if sf is not None:
        sf.copy_(s)
    return q, s


def grouped_fp8_block_quantize(
    x: Tensor,
    out: Tensor | None = None,
    sf: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """3D (expert-major) FP8 blockwise quantize via per-expert loop.

    Optionally writes into preallocated ``out``/``sf``.
    """
    if x.ndim != 3:
        raise ValueError(f"grouped_fp8_block_quantize expects 3D, got shape={tuple(x.shape)}")
    groups, rows, cols = x.shape
    q_accum = torch.empty((groups, rows, cols), dtype=torch.float8_e4m3fn, device=x.device)
    s_accum = torch.empty(
        (groups, ceil_div(rows, BLOCK_SIZE), ceil_div(cols, BLOCK_SIZE)),
        dtype=torch.float32,
        device=x.device,
    )
    for g in range(groups):
        q_g, s_g = quantize_to_fp8_blockwise(x[g], BLOCK_SIZE)
        q_accum[g] = q_g
        s_accum[g] = s_g
    if out is not None:
        out.copy_(q_accum)
    if sf is not None:
        sf.copy_(s_accum)
    return q_accum, s_accum


# ----------------------------------------------------------------------------
# Per-output-channel FP8 (cutlass-style): one scale per row of W. Used by
# cutlass scaled_mm + vLLM's native FP8 path. For a 2D weight of shape
# (out_features, in_features), reduction is over in_features (axis=-1) and
# the resulting scale has shape (out_features,). For a 3D stacked-expert
# weight of shape (num_local_experts, out_features, in_features) we run the
# same recipe per expert, producing a (num_local_experts, out_features)
# scale tensor. No padding / block reshuffling — the bytes go out in the
# same layout the trainer holds them in, which matches cutlass's
# RowMajor + per-channel scale convention.
# ----------------------------------------------------------------------------


def fp8_per_channel_quantize(
    weight: Tensor,
) -> tuple[Tensor, Tensor]:
    """Per-output-channel symmetric FP8 e4m3 quantization.

    Supports both 2D ``(out, in)`` linear weights and 3D
    ``(E, out, in)`` stacked-expert weights via the same code path.
    Returns ``(quantized, scale)`` where ``scale`` has shape
    ``weight.shape[:-1]`` (i.e. one scalar per output row, per expert).
    """
    if weight.ndim not in (2, 3):
        raise ValueError(
            f"fp8_per_channel_quantize expects 2D or 3D, got shape={tuple(weight.shape)}"
        )
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448 for e4m3
    # amax over the innermost (input-features) axis.
    amax = weight.detach().float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = (amax / fp8_max).clamp(min=1e-12)
    q = (weight.float() / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return q.contiguous(), scale.squeeze(-1).to(torch.float32).contiguous()


def fp8_per_channel_quantize_into(
    weight: Tensor,
    out: Tensor | None = None,
    sf: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Per-channel FP8 quantize, optionally writing into preallocated buffers.

    Shape contract:
      - ``out.shape == weight.shape``, dtype ``torch.float8_e4m3fn``
      - ``sf.shape == weight.shape[:-1]``, dtype ``torch.float32``
    """
    q, s = fp8_per_channel_quantize(weight)
    if out is not None:
        out.copy_(q)
    if sf is not None:
        sf.copy_(s)
    return q, s
