import torch
from torch import Tensor


def quantize_to_fp8_blockwise(
    weight: Tensor,
    block_size: int = 128,
    *,
    out_q: Tensor | None = None,
    out_scale: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Quantize a 2D tensor to FP8 e4m3 with per-block scales.

    When ``out_q`` and/or ``out_scale`` are supplied, the function writes its
    results in place into those buffers instead of allocating fresh tensors.
    This lets callers pre-register target memory with NIXL (or other RDMA
    backends) once and reuse it across weight-transfer iterations without any
    per-step allocation.

    Expected ``out_*`` shapes when provided:
      ``out_q``: same shape as ``weight``, dtype ``torch.float8_e4m3fn``.
      ``out_scale``: ``(ceil(rows/block_size), ceil(cols/block_size))``, dtype ``torch.float32``.
    """
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

    quantized = blocks_fp8.permute(0, 2, 1, 3).reshape(padded_rows, padded_cols)[:rows, :cols]

    if out_q is not None:
        if tuple(out_q.shape) != (rows, cols):
            raise ValueError(f"out_q shape {tuple(out_q.shape)} does not match weight {(rows, cols)}")
        if out_q.dtype != torch.float8_e4m3fn:
            raise ValueError(f"out_q dtype must be float8_e4m3fn, got {out_q.dtype}")
        out_q.copy_(quantized)
        q_ret = out_q
    else:
        q_ret = quantized.contiguous()

    scales_f32 = scales.float()
    if out_scale is not None:
        if tuple(out_scale.shape) != tuple(scales_f32.shape):
            raise ValueError(
                f"out_scale shape {tuple(out_scale.shape)} does not match scales {tuple(scales_f32.shape)}"
            )
        if out_scale.dtype != torch.float32:
            raise ValueError(f"out_scale dtype must be float32, got {out_scale.dtype}")
        out_scale.copy_(scales_f32)
        s_ret = out_scale
    else:
        s_ret = scales_f32.contiguous()

    return q_ret, s_ret


def fp8_blockwise_scale_shape(rows: int, cols: int, block_size: int = 128) -> tuple[int, int]:
    """Return the (scale_rows, scale_cols) shape that ``quantize_to_fp8_blockwise`` produces."""
    pad_rows = (block_size - rows % block_size) % block_size
    pad_cols = (block_size - cols % block_size) % block_size
    return ((rows + pad_rows) // block_size, (cols + pad_cols) // block_size)
