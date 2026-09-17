from typing import Callable

import torch
from torch import Tensor


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


def quantize_to_fp8_checkpoint(
    state_dict: dict[str, Tensor],
    block_size: int = 128,
    keep_unquantized: Callable[[str, Tensor], bool] | None = None,
) -> dict[str, Tensor]:
    """Quantize an HF-checkpoint-named state dict to the blockwise-fp8 checkpoint layout.

    This is the layout of the official FP8 checkpoints of MLA-MoE models
    (e.g. DeepSeek-V3-FP8 / GLM-*-FP8) and the only fp8 format vLLM loads
    natively: for every quantized 2D ``.weight`` tensor ``X.weight`` the dict
    carries the fp8 e4m3 weight of shape ``[out, in]`` plus an fp32
    ``X.weight_scale_inv`` of shape ``[ceil(out / block), ceil(in / block)]``.

    A tensor is quantized when it is a 2D ``.weight`` and ``keep_unquantized``
    does not exclude it. Everything else (norms, routers, biases, scales)
    passes through unchanged. Tensors not divisible by the block size are
    zero-padded internally; emitted weights and scales are exact slices.
    """
    out: dict[str, Tensor] = {}
    for name, tensor in state_dict.items():
        should_quantize = tensor.ndim == 2 and name.endswith(".weight")
        if keep_unquantized is not None:
            should_quantize = should_quantize and not keep_unquantized(name, tensor)
        if should_quantize:
            fp8_weight, scales = quantize_to_fp8_blockwise(tensor, block_size)
            out[name] = fp8_weight
            out[name.removesuffix(".weight") + ".weight_scale_inv"] = scales
        else:
            out[name] = tensor
    return out
