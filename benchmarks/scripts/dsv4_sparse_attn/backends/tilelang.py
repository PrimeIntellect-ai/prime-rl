"""The TileLang forward and backward that `main` ships."""

import contextlib
import importlib
import time

import torch

LABEL = "tilelang"
FORWARD_ONLY = False
EXECUTED_SLOT_TILE = {"fwd": 64, "bwd": 32}
IMPORTS = ["tilelang", "prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn"]


def unavailable_reason() -> str | None:
    try:
        import tilelang  # noqa: F401
    except ImportError as error:
        return f"tilelang does not import: {error}"
    return None


def fwd(q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, sinks: torch.Tensor, scale: float):
    from prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn import dsv4_sparse_attn

    return dsv4_sparse_attn(q, kv, indices, sinks, scale, backend="tilelang")


def install_compile_counter():
    import tilelang.cache.kernel_cache as kernel_cache

    tilelang_jit = importlib.import_module("tilelang.jit")

    counts = {"compiles": 0, "disk_loads": 0, "jit_misses": 0, "compile_s": 0.0, "disk_load_s": 0.0, "jit_miss_s": 0.0}
    real_phase = kernel_cache.jit_phase
    real_load = kernel_cache.KernelCache._load_kernel_from_disk
    real_jit_compile = tilelang_jit.JITImpl.compile

    @contextlib.contextmanager
    def counting_phase(name, *args, **kwargs):
        start = time.perf_counter()
        with real_phase(name, *args, **kwargs):
            yield
        if name == "cache.compile":
            counts["compiles"] += 1
            counts["compile_s"] += time.perf_counter() - start

    def counting_load(self, *args, **kwargs):
        start = time.perf_counter()
        kernel = real_load(self, *args, **kwargs)
        if kernel is not None:
            counts["disk_loads"] += 1
            counts["disk_load_s"] += time.perf_counter() - start
        return kernel

    def counting_jit_compile(self, *args, **kwargs):
        start = time.perf_counter()
        kernel = real_jit_compile(self, *args, **kwargs)
        counts["jit_misses"] += 1
        counts["jit_miss_s"] += time.perf_counter() - start
        return kernel

    kernel_cache.jit_phase = counting_phase
    kernel_cache.KernelCache._load_kernel_from_disk = counting_load
    tilelang_jit.JITImpl.compile = counting_jit_compile
    return lambda: dict(counts)
