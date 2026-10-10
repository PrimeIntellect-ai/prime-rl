"""The warp-specialized CuTe DSL forward (sm90) with the TileLang backward."""

import time

import torch

from backends import tilelang

LABEL = "CuTe warp-specialized fwd + TileLang bwd"
FORWARD_ONLY = False
EXECUTED_SLOT_TILE = {"fwd": 128, "bwd": 32}
IMPORTS = [
    "tilelang",
    "cutlass.cute",
    "prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn",
    "prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn_fwd_cute_ws",
]


def unavailable_reason() -> str | None:
    try:
        import cutlass.cute  # noqa: F401
    except ImportError as error:
        return f"cutlass.cute does not import: {error}"
    if torch.cuda.get_device_capability() != (9, 0):
        return f"the CuTe forward is built for sm_90a, this GPU is sm_{''.join(map(str, torch.cuda.get_device_capability()))}"
    return tilelang.unavailable_reason()


def fwd(q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, sinks: torch.Tensor, scale: float):
    from prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn import dsv4_sparse_attn

    return dsv4_sparse_attn(q, kv, indices, sinks, scale, backend="cute_ws")


def install_compile_counter():
    import cutlass.cute as cute

    read_tilelang = tilelang.install_compile_counter()
    counts = {"cute_compiles": 0, "cute_compile_s": 0.0}
    real_compile = cute.compile

    def counting_compile(*args, **kwargs):
        start = time.perf_counter()
        compiled = real_compile(*args, **kwargs)
        counts["cute_compiles"] += 1
        counts["cute_compile_s"] += time.perf_counter() - start
        return compiled

    cute.compile = counting_compile

    def read():
        merged = read_tilelang()
        merged["compiles"] = merged.get("compiles", 0) + counts["cute_compiles"]
        return {**merged, **counts}

    return read
