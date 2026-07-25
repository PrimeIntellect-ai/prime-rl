from __future__ import annotations

import functools
from pathlib import Path

import torch

from prime_rl_kernels.nvfp4._build import load_cuda_extension

_EXTENSION_NAME = "prime_rl_kernels_nvfp4_quantize_sm100"
_SOURCES = ("bindings.cpp", "quantize.cu")
_EXTENSION_READY = False


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _quantize_activations_fake(
    matrix: torch.Tensor,
    offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, contraction_size = matrix.shape
    groups = offsets.shape[0]
    scale_columns = _round_up(contraction_size // 16, 4)
    padded_scale_rows = _round_up(rows + groups * 127, 128)
    return (
        matrix.new_empty((rows, contraction_size // 2), dtype=torch.uint8),
        matrix.new_empty(
            (padded_scale_rows, scale_columns),
            dtype=torch.float8_e4m3fn,
        ),
        matrix.new_empty((rows,), dtype=torch.float32),
    )


def _quantize_weights_fake(
    weight_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    groups, output_size, contraction_size = weight_rows.shape
    padded_output_size = _round_up(output_size, 128)
    scale_columns = _round_up(contraction_size // 16, 4)
    return (
        weight_rows.new_empty(
            (groups, output_size, contraction_size // 2),
            dtype=torch.uint8,
        ),
        weight_rows.new_empty(
            (groups, padded_output_size * scale_columns),
            dtype=torch.float8_e4m3fn,
        ),
        weight_rows.new_empty((groups,), dtype=torch.float32),
    )


def _dequantize_activations_fake(
    packed: torch.Tensor,
    block_scales: torch.Tensor,
    global_scales: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    del block_scales, global_scales, offsets
    return packed.new_empty(
        (packed.shape[0], packed.shape[1] * 2),
        dtype=torch.bfloat16,
    )


def _dequantize_weights_fake(
    packed: torch.Tensor,
    block_scales: torch.Tensor,
    global_scales: torch.Tensor,
) -> torch.Tensor:
    del block_scales, global_scales
    return packed.new_empty(
        (packed.shape[0], packed.shape[1], packed.shape[2] * 2),
        dtype=torch.bfloat16,
    )


@functools.cache
def _load_extension() -> None:
    global _EXTENSION_READY

    source_dir = Path(__file__).with_name("csrc")
    extra_cflags = ["-O3", "-std=c++17"]
    extra_cuda_cflags = [
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--threads=4",
        "-gencode=arch=compute_100a,code=sm_100a",
    ]
    load_cuda_extension(
        base_name=_EXTENSION_NAME,
        sources=[source_dir / source for source in _SOURCES],
        fingerprint_files=source_dir.glob("*"),
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
    )
    _EXTENSION_READY = True


@functools.cache
def _prepare_extension_for_compile() -> None:
    _load_extension()
    torch.library.register_fake(
        "prime_rl_kernels_nvfp4::quantize_activations",
        _quantize_activations_fake,
    )
    torch.library.register_fake(
        "prime_rl_kernels_nvfp4::quantize_weights",
        _quantize_weights_fake,
    )
    torch.library.register_fake(
        "prime_rl_kernels_nvfp4::dequantize_activations",
        _dequantize_activations_fake,
    )
    torch.library.register_fake(
        "prime_rl_kernels_nvfp4::dequantize_weights",
        _dequantize_weights_fake,
    )


def _quantize_activations(
    matrix: torch.Tensor,
    offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not _EXTENSION_READY:
        _load_extension()
    return torch.ops.prime_rl_kernels_nvfp4.quantize_activations.default(matrix, offsets)


def _quantize_weights(
    weight_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not _EXTENSION_READY:
        _load_extension()
    return torch.ops.prime_rl_kernels_nvfp4.quantize_weights.default(weight_rows)


def _dequantize_activations(
    packed: torch.Tensor,
    block_scales: torch.Tensor,
    global_scales: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    if not _EXTENSION_READY:
        _load_extension()
    return torch.ops.prime_rl_kernels_nvfp4.dequantize_activations.default(
        packed,
        block_scales,
        global_scales,
        offsets,
    )


def _dequantize_weights(
    packed: torch.Tensor,
    block_scales: torch.Tensor,
    global_scales: torch.Tensor,
) -> torch.Tensor:
    if not _EXTENSION_READY:
        _load_extension()
    return torch.ops.prime_rl_kernels_nvfp4.dequantize_weights.default(
        packed,
        block_scales,
        global_scales,
    )
