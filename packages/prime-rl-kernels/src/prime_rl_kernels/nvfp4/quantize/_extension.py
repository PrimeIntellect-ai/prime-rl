from __future__ import annotations

import functools
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_EXTENSION_NAME = "prime_rl_kernels_nvfp4_quantize_sm100"
_SOURCES = ("bindings.cpp", "quantize.cu")


@functools.cache
def _load_extension() -> None:
    source_dir = Path(__file__).with_name("csrc")
    load(
        name=_EXTENSION_NAME,
        sources=[str(source_dir / source) for source in _SOURCES],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "--threads=4",
            "-gencode=arch=compute_100a,code=sm_100a",
        ],
        with_cuda=True,
        is_python_module=False,
        verbose=os.environ.get("PRIME_RL_KERNELS_BUILD_VERBOSE") == "1",
    )


def _quantize_activations(
    matrix: torch.Tensor,
    offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _load_extension()
    return torch.ops.prime_rl_kernels_nvfp4.quantize_activations.default(matrix, offsets)


def _quantize_weights(
    weight_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _load_extension()
    return torch.ops.prime_rl_kernels_nvfp4.quantize_weights.default(weight_rows)


def _dequantize_activations(
    packed: torch.Tensor,
    block_scales: torch.Tensor,
    global_scales: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
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
    _load_extension()
    return torch.ops.prime_rl_kernels_nvfp4.dequantize_weights.default(
        packed,
        block_scales,
        global_scales,
    )
