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


def quantize_to_vllm_kernel_format(weight: Tensor, block_size: int = 128) -> tuple[Tensor, Tensor]:
    """Quantize a weight into the FP8 layout used by vLLM kernels.

    Hopper kernels consume the regular blockwise scale grid. On SM100, vLLM's
    DeepGEMM kernels instead use UE8M0 scales in an architecture-specific packed
    layout. Reuse vLLM's own post-processing so tensors sent over the kernel
    weight-transfer path have the same representation as the loaded parameters.
    """
    quantized, scales = quantize_to_fp8_blockwise(weight, block_size)

    if not weight.is_cuda or torch.cuda.get_device_capability(weight.device) != (10, 0):
        return quantized, scales

    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
    )

    return deepgemm_post_process_fp8_weight_block(
        wq=quantized,
        ws=scales,
        quant_block_shape=(block_size, block_size),
        use_e8m0=True,
    )
