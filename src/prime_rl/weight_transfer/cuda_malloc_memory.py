"""CUDA memory sizing and a cudaMalloc-backed pool for NIXL arenas."""

from __future__ import annotations

import ctypes
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

try:
    ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)
except OSError:
    pass

import torch  # noqa: E402
from torch.utils.cpp_extension import load_inline  # noqa: E402

_pool: torch.cuda.MemPool | None = None
_allocator: torch.cuda.memory.CUDAPluggableAllocator | None = None


@dataclass(frozen=True)
class CudaBufferSizing:
    buffer_count: int
    free_bytes: int
    total_bytes: int
    headroom_bytes: int


def size_cuda_buffers(
    buffer_bytes: int,
    max_buffers: int,
    device: torch.device,
    extra_headroom_bytes: int,
) -> CudaBufferSizing:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    headroom_bytes = max(4 * 1024**3, int(total_bytes * 0.02)) + extra_headroom_bytes
    buffer_count = max(1, min(max_buffers, (free_bytes - headroom_bytes) // buffer_bytes))
    return CudaBufferSizing(
        buffer_count=buffer_count,
        free_bytes=free_bytes,
        total_bytes=total_bytes,
        headroom_bytes=headroom_bytes,
    )


def _get_pool() -> torch.cuda.MemPool:
    global _pool, _allocator
    if _pool is not None:
        return _pool
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
    _allocator = torch.cuda.memory.CUDAPluggableAllocator(
        str(Path(module.__file__)), "cuda_malloc", "cuda_free"
    )
    _pool = torch.cuda.MemPool(_allocator.allocator())
    return _pool


@contextmanager
def use_cuda_malloc_pool() -> Iterator[None]:
    with torch.cuda.use_mem_pool(_get_pool()):
        yield
