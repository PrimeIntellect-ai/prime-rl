"""FlashMLA's sparse prefill forward with the cuDNN frontend's SM90 DSA backward, as `dsa_backend` ships it."""

import time

import torch

LABEL = "cudnn_flashmla"
FORWARD_ONLY = False
EXECUTED_SLOT_TILE = {"fwd": 64, "bwd": 64}


def unavailable_reason() -> str | None:
    try:
        import flash_mla  # noqa: F401
        from cudnn.deepseek_sparse_attention.sparse_attention_backward import _interface_sm90  # noqa: F401
    except ImportError as error:
        return f"flash_mla or the cuDNN DSA backward does not import: {error}"
    if torch.cuda.get_device_capability()[0] != 9:
        return "the cudnn_flashmla backend runs on SM90 only"
    return None


def fwd(q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, sinks: torch.Tensor, scale: float):
    from prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn import dsv4_sparse_attn

    return dsv4_sparse_attn(q, kv, indices, sinks, scale, backend="cudnn_flashmla")


def install_compile_counter():
    import cutlass.cute as cute
    from cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm90 import flash_attn_bwd_sm90

    compile_ms = [0.0]
    real_compile = cute.compile

    def timed_compile(*args, **kwargs):
        start = time.perf_counter()
        compiled = real_compile(*args, **kwargs)
        compile_ms[0] += (time.perf_counter() - start) * 1e3
        return compiled

    cute.compile = timed_compile

    def read_counts() -> dict[str, int]:
        main = len(flash_attn_bwd_sm90.compile_cache)
        auxiliary = len(flash_attn_bwd_sm90.compile_cache_pre) + len(flash_attn_bwd_sm90.compile_cache_post)
        return {"compiles": main + auxiliary, "bwd_main_compiles": main, "compile_ms": round(compile_ms[0])}

    return read_counts
