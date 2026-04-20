"""Classic ``cudaMalloc``-backed CUDA MemPool for NIXL-registered buffers.

With ``PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`` the default CUDA
caching allocator hands out VMM-backed (``cuMemCreate`` + ``cuMemMap``)
virtual ranges. ``ibv_reg_mr`` on such ranges succeeds, but the mlx5 HCA's
MMU walk at WRITE time completes with ``syndrome 0x4`` ("Local protection"),
because ``nvidia_peermem``'s ``get_pages`` cannot pin a VA that spans
multiple ``cuMemCreate`` handles. UCX tears the endpoint down and NIXL
surfaces it to the app as ``REMOTE_DISCONNECT``.

Tensors we hand to ``nixl_agent.register_memory`` must therefore come from
a classic, contiguous ``cudaMalloc`` block. We expose a ``MemPool`` backed
by a ``CUDAPluggableAllocator`` that calls ``cudaMalloc`` / ``cudaFree``
directly, plus a ``classic_cuda_alloc()`` context manager for scoping
specific allocations into the pool. Everything else in the process keeps
using expandable segments.
"""

from __future__ import annotations

import ctypes as _ctypes
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# TileLang ships a libcudart stub that proxies to the real CUDA runtime via
# dlsym(RTLD_DEFAULT, ...). If the stub's own symbols are the first ones
# found (because nothing loaded the real libcudart globally yet) its
# self-check fails and the stub calls abort() — which is what we hit the
# moment we enter the classic-cudaMalloc MemPool. Preloading the real
# library with RTLD_GLOBAL makes dlsym find it first. Wrapped in try/except
# because CDLL can fail on machines without a real CUDA runtime (e.g. CI).
try:
    _ctypes.CDLL("libcudart.so", mode=_ctypes.RTLD_GLOBAL)
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
    int prev = -1;
    cudaGetDevice(&prev);
    cudaSetDevice(device);
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, (size_t) size);
    if (prev >= 0) cudaSetDevice(prev);
    if (err != cudaSuccess) return nullptr;
    return ptr;
}
void prime_rl_classic_free(void* ptr, ptrdiff_t size, int device, void* stream) {
    (void) size; (void) stream;
    int prev = -1;
    cudaGetDevice(&prev);
    cudaSetDevice(device);
    cudaFree(ptr);
    if (prev >= 0) cudaSetDevice(prev);
}
}
"""

_pool: torch.cuda.MemPool | None = None
_allocator_wrapper: torch.cuda.memory.CUDAPluggableAllocator | None = None


def _get_pool() -> torch.cuda.MemPool:
    global _pool, _allocator_wrapper
    if _pool is not None:
        return _pool
    module = load_inline(
        name="prime_rl_classic_cuda_alloc_v2",
        cpp_sources=[_SOURCE],
        functions=[],
        extra_cflags=["-O2"],
        with_cuda=True,
    )
    so_path = Path(module.__file__)
    _allocator_wrapper = torch.cuda.memory.CUDAPluggableAllocator(
        str(so_path), "prime_rl_classic_alloc", "prime_rl_classic_free"
    )
    _pool = torch.cuda.MemPool(_allocator_wrapper.allocator())
    return _pool


@contextmanager
def classic_cuda_alloc() -> Iterator[None]:
    """Scope tensor allocations into a classic-``cudaMalloc`` MemPool.

    Use when the resulting tensor's address must be a contiguous
    ``cudaMalloc`` block — currently only the NIXL-registered slot buffers.
    """
    pool = _get_pool()
    with torch.cuda.use_mem_pool(pool):
        yield
