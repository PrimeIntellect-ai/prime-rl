from __future__ import annotations

import functools
import importlib.util
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_EXTENSION_NAME = "prime_rl_kernels_nvfp4_sm100"
_SOURCES = (
    "bindings.cpp",
    "f4f4bf16_ultra_grouped.cu",
    "f4f4bf16_ultra_grouped_256_128_256_2_1_1.cu",
    "f4f4bf16_ultra_grouped_256_256_256_2_1_1.cu",
)


def _cutlass_source_root() -> Path:
    spec = importlib.util.find_spec("cutlass_library")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("prime-rl-kernels NVFP4 grouped GEMM requires the nvidia-cutlass package")
    source_root = Path(next(iter(spec.submodule_search_locations))) / "source"
    if not (source_root / "include" / "cutlass" / "cutlass.h").is_file():
        raise RuntimeError(f"nvidia-cutlass headers were not found under {source_root}")
    return source_root


@functools.cache
def _load_extension() -> None:
    source_dir = Path(__file__).with_name("csrc")
    cutlass_root = _cutlass_source_root()
    load(
        name=_EXTENSION_NAME,
        sources=[str(source_dir / source) for source in _SOURCES],
        extra_include_paths=[
            str(source_dir),
            str(cutlass_root / "include"),
            str(cutlass_root / "tools" / "util" / "include"),
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--threads=4",
            "-gencode=arch=compute_100a,code=sm_100a",
        ],
        with_cuda=True,
        is_python_module=False,
        verbose=os.environ.get("PRIME_RL_KERNELS_BUILD_VERBOSE") == "1",
    )


def _grouped_mm(
    activations: torch.Tensor,
    weight: torch.Tensor,
    activation_block_scales: torch.Tensor,
    weight_block_scales: torch.Tensor,
    offsets: torch.Tensor,
    activation_token_scales: torch.Tensor,
    weight_expert_scales: torch.Tensor,
) -> torch.Tensor:
    _load_extension()
    return torch.ops.prime_rl_kernels_nvfp4.grouped_mm.default(
        activations,
        weight,
        activation_block_scales,
        weight_block_scales,
        offsets,
        activation_token_scales,
        weight_expert_scales,
    )
