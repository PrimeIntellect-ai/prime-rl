"""Compile the cuDNN DSA backward kernel once, instead of again whenever sequence length or packing changes.

Out of the box, cuDNN compiles a new kernel for every number of key slots per query (`topk_idxs.shape[-1]`), and
that number changes with sequence length and document packing. The kernel only uses it as an upper limit, so
when a call passes `topk_length` we compile once with a fixed limit that covers every call. Calls without
`topk_length` are left alone. On SM100 this applies to the `generic_m64` kernel only, and the number of key slots
must be a multiple of 64. This patches private cuDNN frontend internals and raises if they change.
"""

import math
import time
from collections.abc import Callable

import cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm90 as interface_sm90
import cudnn.deepseek_sparse_attention.sparse_attention_backward._interface_sm100 as interface_sm100
import cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm90 as kernel_sm90
import cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100 as kernel_sm100
import cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100_deterministic as kernel_sm100_deterministic
import torch

from prime_rl.utils.logger import get_logger

_topk_clamp: int | None = None
_stock_init = kernel_sm90.FlashAttentionDSABackwardSm90.__init__
_sm100_topk_clamp: int | None = None
_sm100_stock_init = kernel_sm100.FlashAttentionDSABackwardSm100.__init__
_SM100_GENERIC_KERNELS = (
    kernel_sm100.FlashAttentionDSABackwardSm100,
    kernel_sm100_deterministic.FlashAttentionDSABackwardSm100Deterministic,
)


def topk_clamp(max_slots: int) -> int:
    """The `max_topk` the main kernel is built with: the smallest power of two that is at least `max_slots`.

    A power of two compiles to nearly the stock kernel's SASS, where `2**31 - 1` costs 1% of forward+backward.
    """
    exponent = math.ceil(math.log2(max_slots))
    return 2**exponent


def _main_kernel_key(key: tuple) -> tuple:
    *static_key, have_topk_length, max_topk = key
    if not isinstance(have_topk_length, bool) or not isinstance(max_topk, int):
        raise RuntimeError(f"the cuDNN DSA main-kernel key no longer ends in (have_topk_length, max_topk): {key}")
    if not have_topk_length:
        return key
    if _topk_clamp is None:
        raise RuntimeError("a cuDNN DSA backward call with topk_length must go through cudnn_dsa_patch")
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

    def __init__(
        self,
        kernel: str,
        normalize_key: Callable[[tuple], tuple] = lambda key: key,
        max_topk_position: int | None = None,
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.normalize_key = normalize_key
        self.max_topk_position = max_topk_position
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
        max_topk = "" if self.max_topk_position is None else f" with max_topk {normalized_key[self.max_topk_position]}"
        get_logger().info(f"Compiled the cuDNN DSA backward {self.kernel} kernel{max_topk} in {seconds:.2f}s")


def _patch_sm90_compile_cache() -> None:
    if kernel_sm90.FlashAttentionDSABackwardSm90.__init__ is _init_with_clamped_max_topk:
        return
    kernel_sm90.FlashAttentionDSABackwardSm90.__init__ = _init_with_clamped_max_topk
    bwd = interface_sm90.flash_attn_bwd_sm90
    bwd.compile_cache_pre = _ExecutorCache("preprocess")
    bwd.compile_cache = _ExecutorCache("main", _main_kernel_key, max_topk_position=-1)
    bwd.compile_cache_post = _ExecutorCache("postprocess")


def flash_attn_bwd_sm90(*args, max_slots: int | None = None, **kwargs) -> tuple[torch.Tensor, ...]:
    """The frontend's `flash_attn_bwd_sm90`; a call with `topk_length` serves widths up to `max_slots`."""
    global _topk_clamp
    _patch_sm90_compile_cache()
    if kwargs.get("topk_length") is not None:
        if max_slots is None:
            raise ValueError("a cuDNN DSA backward call with topk_length needs max_slots, its widest slot width")
        _topk_clamp = topk_clamp(max_slots)
    return interface_sm90.flash_attn_bwd_sm90(*args, **kwargs)


def _sm100_main_kernel_key(key: tuple) -> tuple:
    backend, *static_key, max_topk, has_topk_length = key
    if not isinstance(has_topk_length, bool) or not isinstance(max_topk, int):
        raise RuntimeError(f"the cuDNN SM100 DSA key no longer ends in (max_topk, has_topk_length): {key}")
    if backend != "generic_m64" or not has_topk_length:
        return key
    if _sm100_topk_clamp is None:
        raise RuntimeError("a cuDNN SM100 DSA backward call with topk_length must go through cudnn_dsa_patch")
    if max_topk > _sm100_topk_clamp:
        raise ValueError(f"slot width {max_topk} exceeds the cuDNN DSA backward's compiled clamp {_sm100_topk_clamp}")
    if max_topk % 64 != 0:
        raise ValueError(
            f"the clamped cuDNN SM100 DSA backward needs a slot width that is a multiple of 64, got {max_topk}"
        )
    return (backend, *static_key, _sm100_topk_clamp, has_topk_length)


def _sm100_init_with_clamped_max_topk(self, element_dtype, head_dim, head_dim_v, block_tile, max_topk=0) -> None:
    if _sm100_topk_clamp is not None and type(self) in _SM100_GENERIC_KERNELS:
        max_topk = _sm100_topk_clamp
    _sm100_stock_init(self, element_dtype, head_dim, head_dim_v, block_tile, max_topk)


def _patch_sm100_compile_cache() -> None:
    if kernel_sm100.FlashAttentionDSABackwardSm100.__init__ is _sm100_init_with_clamped_max_topk:
        return
    kernel_sm100.FlashAttentionDSABackwardSm100.__init__ = _sm100_init_with_clamped_max_topk
    interface_sm100.flash_attn_bwd_sm100.compile_cache = _ExecutorCache(
        "main", _sm100_main_kernel_key, max_topk_position=-2
    )


def flash_attn_bwd_sm100(*args, max_slots: int | None = None, **kwargs) -> tuple[torch.Tensor, ...]:
    """The frontend's `flash_attn_bwd_sm100`; a call with `topk_length` serves widths up to `max_slots`."""
    global _sm100_topk_clamp
    _patch_sm100_compile_cache()
    if kwargs.get("topk_length") is None:
        return interface_sm100.flash_attn_bwd_sm100(*args, **kwargs)
    if max_slots is None:
        raise ValueError("a cuDNN DSA backward call with topk_length needs max_slots, its widest slot width")
    _sm100_topk_clamp = topk_clamp(max_slots)
    try:
        return interface_sm100.flash_attn_bwd_sm100(*args, **kwargs)
    finally:
        _sm100_topk_clamp = None
