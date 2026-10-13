"""The cuDNN frontend's SM90 DSA backward, with a main kernel that compiles once for every slot width.

The main kernel reads `max_topk` only to clamp each query's `topk_length`: it sizes no shared memory and
unrolls nothing. The frontend still keys the compile on `max_topk = topk_idxs.shape[-1]`, so every new slot
width would compile a new kernel. When the caller passes `topk_length`, this builds the kernel with `max_topk`
at a fixed clamp and keys it on the clamp. Without `topk_length`, `max_topk` stands in for the length, so that
path (GLM's) keeps the frontend's behavior. Every compile of the three backward kernels (preprocess, main,
postprocess) is logged at INFO. It leans on private frontend internals (the `compile_cache*` attributes, the
main key ending in `(have_topk_length, max_topk)`, the `max_topk` kwarg) and raises if they move.
"""

import time
from collections.abc import Callable

import cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm90 as interface_sm90
import cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm90 as kernel_sm90
import torch

from prime_rl.utils.logger import get_logger

_topk_clamp: int | None = None
_stock_init = kernel_sm90.FlashAttentionDSABackwardSm90.__init__


def topk_clamp(max_slots: int) -> int:
    """The `max_topk` the main kernel is built with to serve every width up to `max_slots`.

    A power of two: at 16384 the SASS equals the stock 640-wide kernel's apart from four immediates, while
    `2**31 - 1` adds 120 instructions and costs 1% of forward+backward.
    """
    return 1 << (max_slots - 1).bit_length()


def _main_kernel_key(key: tuple) -> tuple:
    *static_key, have_topk_length, max_topk = key
    if not isinstance(have_topk_length, bool) or not isinstance(max_topk, int):
        raise RuntimeError(f"the cuDNN DSA main-kernel key no longer ends in (have_topk_length, max_topk): {key}")
    if not have_topk_length:
        return key
    if _topk_clamp is None:
        raise RuntimeError("a cuDNN DSA backward call with topk_length must go through cudnn_dsa_compile")
    if max_topk > _topk_clamp:
        raise ValueError(f"slot width {max_topk} exceeds the cuDNN DSA backward's compiled clamp {_topk_clamp}")
    return (*static_key, have_topk_length, _topk_clamp)


def _init_with_clamped_max_topk(self, *args, have_topk_length: bool, max_topk: int, **kwargs) -> None:
    _stock_init(
        self, *args, have_topk_length=have_topk_length, max_topk=_topk_clamp if have_topk_length else max_topk, **kwargs
    )


class _ExecutorCache(dict):
    """An executor cache that stores entries under `normalize_key(key)` and logs each compile.

    The frontend checks `key in cache`, compiles on a miss, then assigns the executor, so the time between
    the missed lookup and the assignment is the compile.
    """

    def __init__(self, kernel: str, normalize_key: Callable[[tuple], tuple] = lambda key: key) -> None:
        super().__init__()
        self.kernel = kernel
        self.normalize_key = normalize_key
        self.miss_time = 0.0

    def __contains__(self, key) -> bool:
        found = dict.__contains__(self, self.normalize_key(key))
        if not found:
            self.miss_time = time.perf_counter()
        return found

    def __getitem__(self, key):
        return dict.__getitem__(self, self.normalize_key(key))

    def __setitem__(self, key, compiled) -> None:
        normalized_key = self.normalize_key(key)
        dict.__setitem__(self, normalized_key, compiled)
        seconds = time.perf_counter() - self.miss_time
        max_topk = f" with max_topk {normalized_key[-1]}" if self.kernel == "main" else ""
        get_logger().info(f"Compiled the cuDNN DSA backward {self.kernel} kernel{max_topk} in {seconds:.2f}s")


def _install() -> None:
    if kernel_sm90.FlashAttentionDSABackwardSm90.__init__ is _init_with_clamped_max_topk:
        return
    kernel_sm90.FlashAttentionDSABackwardSm90.__init__ = _init_with_clamped_max_topk
    bwd = interface_sm90.flash_attn_bwd_sm90
    bwd.compile_cache_pre = _ExecutorCache("preprocess")
    bwd.compile_cache = _ExecutorCache("main", _main_kernel_key)
    bwd.compile_cache_post = _ExecutorCache("postprocess")


def flash_attn_bwd_sm90(*args, max_slots: int | None = None, **kwargs) -> tuple[torch.Tensor, ...]:
    """The frontend's `flash_attn_bwd_sm90`; a call with `topk_length` serves widths up to `max_slots`."""
    global _topk_clamp
    _install()
    if kwargs.get("topk_length") is not None:
        if max_slots is None:
            raise ValueError("a cuDNN DSA backward call with topk_length needs max_slots, its widest slot width")
        _topk_clamp = topk_clamp(max_slots)
    return interface_sm90.flash_attn_bwd_sm90(*args, **kwargs)
