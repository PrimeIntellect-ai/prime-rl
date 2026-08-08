"""Device memory sizing and registerable allocation for NIXL arenas."""

from __future__ import annotations

import ctypes
import os
from contextlib import contextmanager
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


def size_device_buffers(
    buffer_bytes: int,
    max_buffers: int,
    device: torch.device,
    extra_headroom_bytes: int,
) -> int:
    free_bytes, total_bytes = torch.get_device_module(device).mem_get_info(device)
    headroom_bytes = max(4 * 1024**3, int(total_bytes * 0.02)) + extra_headroom_bytes
    return max(1, min(max_buffers, (free_bytes - headroom_bytes) // buffer_bytes))


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
    _allocator = torch.cuda.memory.CUDAPluggableAllocator(str(Path(module.__file__)), "cuda_malloc", "cuda_free")
    _pool = torch.cuda.MemPool(_allocator.allocator())
    return _pool


def _expandable_segments_enabled(config: str) -> bool:
    for item in config.split(","):
        key, separator, value = item.partition(":")
        if separator and key.strip() == "expandable_segments" and value.strip().lower() == "true":
            return True
    return False


def _check_caching_allocator_registerable() -> None:
    """Reject expandable segments, which hand out unregisterable addresses.

    An expandable segment spans several driver-level physical handles, so the
    NIC cannot pin the virtual range and NIXL registration fails. Both variables
    are supported because a trainer inherits ``PYTORCH_CUDA_ALLOC_CONF`` regardless
    of its accelerator (see ``DEFAULT_TRAINER_ENV_VARS``).
    """
    config = os.environ.get("PYTORCH_ALLOC_CONF")
    if config is None:
        config = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if _expandable_segments_enabled(config):
        raise RuntimeError(
            "NIXL weight transfer cannot register memory allocated with "
            "expandable_segments:True. Set expandable_segments:False for this component."
        )


@contextmanager
def use_registerable_pool(device: torch.device) -> Iterator[None]:
    """Allocate memory that NIXL can register."""
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
