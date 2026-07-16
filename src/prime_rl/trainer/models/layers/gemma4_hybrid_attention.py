"""Hybrid Hopper attention for Gemma 4's mixed head dimensions.

Gemma 4 uses 256-wide heads in sliding-attention layers and 512-wide heads in
global-attention layers.  CuTe FlashAttention 4 on SM90 supports the former but
not the latter, so the global layers use exact blockwise FlexAttention instead
of falling back to SDPA.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
from transformers import AttentionInterface
from transformers.integrations.flash_attention import flash_attention_forward

_GLOBAL_HEAD_DIM = 512
_FLEX_KERNEL_OPTIONS = {
    "BLOCK_M": 32,
    "BLOCK_N": 32,
    "num_warps": 4,
    "num_stages": 1,
}
_compiled_flex_attention = torch.compile(flex_attention, dynamic=False)


def _causal_mask(_batch: Tensor, _head: Tensor, query: Tensor, key: Tensor) -> Tensor:
    return query >= key


@lru_cache(maxsize=16)
def _get_causal_block_mask(query_length: int, key_length: int, device_index: int) -> BlockMask:
    return create_block_mask(
        _causal_mask,
        B=None,
        H=None,
        Q_LEN=query_length,
        KV_LEN=key_length,
        device=torch.device("cuda", device_index),
    )


def gemma4_hybrid_attention_forward(
    module: nn.Module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[Tensor, None]:
    """Use FA4 for supported heads and exact FlexAttention for Gemma 4 D=512."""
    if query.shape[-1] <= 256:
        return flash_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            sliding_window=sliding_window,
            is_causal=is_causal,
            **kwargs,
        )

    if query.shape[-1] != _GLOBAL_HEAD_DIM or key.shape[-1] != _GLOBAL_HEAD_DIM:
        raise ValueError(
            f"Gemma 4 hybrid attention expected head_dim <= 256 or {_GLOBAL_HEAD_DIM}, "
            f"got query={query.shape[-1]}, key={key.shape[-1]}"
        )
    if value.shape[-1] != _GLOBAL_HEAD_DIM:
        raise ValueError(f"Gemma 4 global value head_dim must be {_GLOBAL_HEAD_DIM}, got {value.shape[-1]}")
    if attention_mask is not None:
        raise ValueError("Gemma 4 D=512 FlexAttention path requires padding-free inputs")
    if dropout:
        raise ValueError("Gemma 4 D=512 FlexAttention path does not support attention dropout")
    if sliding_window is not None:
        raise ValueError("Gemma 4 D=512 heads are expected only in global-attention layers")

    causal = module.is_causal if is_causal is None else is_causal
    block_mask = None
    if causal:
        device_index = query.device.index
        if device_index is None:
            raise ValueError("Gemma 4 hybrid attention requires CUDA tensors")
        block_mask = _get_causal_block_mask(query.shape[-2], key.shape[-2], device_index)

    output = _compiled_flex_attention(
        query,
        key,
        value,
        block_mask=block_mask,
        enable_gqa=query.shape[1] != key.shape[1],
        scale=scaling,
        kernel_options=_FLEX_KERNEL_OPTIONS,
    )
    return output.transpose(1, 2).contiguous(), None


def register_gemma4_hybrid_attention() -> None:
    """Replace Transformers' FA4 interface with the Gemma 4 hybrid dispatcher."""
    AttentionInterface.register("flash_attention_4", gemma4_hybrid_attention_forward)
