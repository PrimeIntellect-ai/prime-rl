"""Classic cudaMalloc pool for GPU memory registered with NIXL."""

from __future__ import annotations

import ctypes
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

try:
    ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)
except OSError:
    pass

import torch  # noqa: E402
from torch.utils.cpp_extension import load_inline  # noqa: E402

_SOURCE = r"""
#include <cuda_runtime.h>
#include <cstddef>
extern "C" {
void* prime_rl_classic_alloc(ptrdiff_t size, int device, void* stream) {
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
void prime_rl_classic_free(void* pointer, ptrdiff_t size, int device, void* stream) {
    (void) size; (void) stream;
    int previous = -1;
    cudaGetDevice(&previous);
    cudaSetDevice(device);
    cudaFree(pointer);
    if (previous >= 0) cudaSetDevice(previous);
}
}
"""

_pool: torch.cuda.MemPool | None = None
_allocator: torch.cuda.memory.CUDAPluggableAllocator | None = None


def _get_pool() -> torch.cuda.MemPool:
    global _pool, _allocator
    if _pool is not None:
        return _pool
    module = load_inline(
        name="prime_rl_classic_cuda_alloc",
        cpp_sources=[_SOURCE],
        functions=[],
        extra_cflags=["-O2"],
        with_cuda=True,
    )
    _allocator = torch.cuda.memory.CUDAPluggableAllocator(
        str(Path(module.__file__)), "prime_rl_classic_alloc", "prime_rl_classic_free"
    )
    _pool = torch.cuda.MemPool(_allocator.allocator())
    return _pool


@contextmanager
def classic_cuda_alloc() -> Iterator[None]:
    with torch.cuda.use_mem_pool(_get_pool()):
        yield
