"""Site-customize hook — auto-imported by every Python process via Python's
site initialization. Used here to preload the real CUDA runtime with
RTLD_GLOBAL so TileLang's libcudart_stub.so doesn't win dlsym() lookups.

Without this, vLLM's WorkerProc subprocesses (which spawn from
multiproc_executor before any user module loads) hit:
    AttributeError: /app/.venv/lib/python3.12/site-packages/tilelang/lib/
    libcudart_stub.so: undefined symbol: cudaDeviceReset
when FlashInfer's CudaRTLibrary tries to dlsym `cudaDeviceReset`.

LD_PRELOAD set in the entrypoint shell does NOT reliably propagate to the
spawned worker subprocesses (uv/multiprocessing.spawn interaction). This
sitecustomize runs at every Python interpreter startup, before vLLM's
worker_main is reached, before any vllm/flashinfer/tilelang module loads.
"""

import ctypes
import os

_REAL_LIBCUDART_CANDIDATES = (
    "/usr/local/cuda/lib64/libcudart.so.12",
    "/usr/local/cuda/lib64/libcudart.so",
    "/usr/lib/aarch64-linux-gnu/libcudart.so.12",
    "/usr/lib/x86_64-linux-gnu/libcudart.so.12",
)
for _candidate in _REAL_LIBCUDART_CANDIDATES:
    if os.path.exists(_candidate):
        try:
            ctypes.CDLL(_candidate, mode=ctypes.RTLD_GLOBAL)
            break
        except OSError:
            continue
