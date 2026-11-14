"""Device memory that NIXL can register with the NIC."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import torch

# PyTorch reads the first of these that is set, in this order, for every accelerator.
_ALLOC_CONF_VARS = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_ALLOC_CONF")

_pool: torch.cuda.MemPool | None = None
_allocator: torch.cuda.memory.CUDAPluggableAllocator | None = None


def _get_pool() -> torch.cuda.MemPool:
    """Build (once) a cudaMalloc-backed pool, bypassing the caching allocator."""
    global _pool, _allocator
    if _pool is not None:
        return _pool

    import ctypes
    from pathlib import Path

    from torch.utils.cpp_extension import load_inline

    # The inline extension links against cudart, which is not loaded globally
    # by default in every environment.
    try:
        ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass

    module = load_inline(
        name="cuda_malloc_allocator",
        cpp_sources=[
            r"""
#include <cuda_runtime.h>
#include <cstddef>
extern "C" {
void* cuda_malloc(ptrdiff_t size, int device, void* stream) {
    (void) stream;
    int previous = -1;
    cudaGetDevice(&previous);
    cudaSetDevice(device);
    void* pointer = nullptr;
    cudaError_t error = cudaMalloc(&pointer, (size_t) size);
    if (previous >= 0) cudaSetDevice(previous);
    if (error != cudaSuccess) return nullptr;
    return pointer;
}
void cuda_free(void* pointer, ptrdiff_t size, int device, void* stream) {
    (void) size; (void) stream;
    int previous = -1;
    cudaGetDevice(&previous);
    cudaSetDevice(device);
    cudaFree(pointer);
    if (previous >= 0) cudaSetDevice(previous);
}
}
"""
        ],
        functions=[],
        extra_cflags=["-O2"],
        with_cuda=True,
    )
    _allocator = torch.cuda.memory.CUDAPluggableAllocator(str(Path(module.__file__)), "cuda_malloc", "cuda_free")
    _pool = torch.cuda.MemPool(_allocator.allocator())
    return _pool


def _expandable_segments_enabled(config: str) -> bool:
    """Parse as PyTorch does: whitespace is ignored and the last setting wins."""
    enabled = False
    for item in "".join(config.split()).split(","):
        key, _, value = item.partition(":")
        if key == "expandable_segments":
            enabled = value == "True"
    return enabled


def _check_caching_allocator_registerable() -> None:
    """Reject expandable segments, which hand out unregisterable addresses.

    An expandable segment spans several driver-level physical handles, so the NIC
    cannot pin the virtual range and NIXL registration fails. PyTorch configures
    every accelerator's allocator from the first of ``_ALLOC_CONF_VARS`` that is
    set, so this checks the same variable. A trainer inherits
    ``PYTORCH_CUDA_ALLOC_CONF`` from ``DEFAULT_TRAINER_ENV_VARS`` regardless of its
    accelerator, which takes precedence over ``PYTORCH_ALLOC_CONF``. Settings
    changed at runtime, after PyTorch reads the environment, are not seen.
    """
    name = next((var for var in _ALLOC_CONF_VARS if var in os.environ), None)
    if name is not None and _expandable_segments_enabled(os.environ[name]):
        raise RuntimeError(
            "NIXL weight transfer cannot register memory allocated with "
            f"expandable_segments:True. Set {name}=expandable_segments:False "
            "in this component's env_vars."
        )


@contextmanager
def use_registerable_pool(device: torch.device) -> Iterator[None]:
    """Allocate arenas the NIC can pin, which differs by allocator."""
    if device.type == "cuda":
        # The caching allocator runs with expandable segments here, so arenas come
        # from a dedicated cudaMalloc pool instead.
        with torch.cuda.use_mem_pool(_get_pool()):
            yield
    elif device.type == "xpu":
        # Registration needs one allocation per region, which the caching allocator
        # already satisfies once expandable segments are off.
        _check_caching_allocator_registerable()
        yield
    else:
        raise NotImplementedError(f"NIXL weight transfer does not support {device.type!r} devices")
