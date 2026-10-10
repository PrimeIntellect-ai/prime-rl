"""The TileLang forward and backward that `main` ships."""

import contextlib

import torch

LABEL = "tilelang"
FORWARD_ONLY = False
EXECUTED_SLOT_TILE = {"fwd": 64, "bwd": 32}


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

    counts = {"compiles": 0, "disk_loads": 0}
    real_phase = kernel_cache.jit_phase
    real_load = kernel_cache.KernelCache._load_kernel_from_disk

    @contextlib.contextmanager
    def counting_phase(name, *args, **kwargs):
        if name == "cache.compile":
            counts["compiles"] += 1
        with real_phase(name, *args, **kwargs):
            yield

    def counting_load(self, *args, **kwargs):
        kernel = real_load(self, *args, **kwargs)
        if kernel is not None:
            counts["disk_loads"] += 1
        return kernel

    kernel_cache.jit_phase = counting_phase
    kernel_cache.KernelCache._load_kernel_from_disk = counting_load
    return lambda: dict(counts)
