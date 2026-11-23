# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 attention: relative-bias GQA with short-conv K/V, L2-clipped
q/k/r, gated output and per-token attention sinks.

The paged relative attention kernels are pure Triton (`ops/attention.py`), so
this module runs on Hopper as well as Blackwell.
"""

from typing import cast

import torch
from torch import nn
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
)

from .configs import Inkling2ModelConfig
from .ops.attention import (
    gated_rel_attention,
    plan_decode_attention,
    rel_decode_attention,
)
from .ops.sconv import attention_prologue, shift_rel_proj
from .sconv import K_STREAM, V_STREAM, Inkling2ShortConv, conv_metadata


class Inkling2AttentionMetadataBuilder(FlashAttentionMetadataBuilder):
    """Orders decode/verify requests first."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)


class Inkling2AttentionBackend(FlashAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[FlashAttentionMetadataBuilder]:
        return Inkling2AttentionMetadataBuilder


def _kv_cache_spec(
    vllm_config: VllmConfig, num_kv_heads: int, head_dim: int, sliding_window: int | None
) -> KVCacheSpec:
    block_size = vllm_config.cache_config.block_size
    dtype = vllm_config.model_config.dtype
    if sliding_window is None:
        return FullAttentionSpec(block_size=block_size, num_kv_heads=num_kv_heads, head_size=head_dim, dtype=dtype)
    return SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_dim,
        dtype=dtype,
        sliding_window=sliding_window,
    )


class Inkling2AttentionPassCache(nn.Module, AttentionLayerBase):
    """The KV cache of one extra loop pass of an `Inkling2Attention`: each
    pass of the loop core attends over its own keys and values."""

    def __init__(
        self, prefix: str, num_heads: int, num_kv_heads: int, head_dim: int, sliding_window: int | None
    ) -> None:
        super().__init__()
        self.prefix = prefix
        # vLLM groups attention layers by their per-rank head count.
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.kv_cache = torch.tensor([])
        forward_context = get_current_vllm_config().compilation_config.static_forward_context
        if prefix in forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        forward_context[prefix] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return Inkling2AttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return _kv_cache_spec(vllm_config, self.num_kv_heads, self.head_dim, self.sliding_window)


class Inkling2Attention(nn.Module, AttentionLayerBase):
    """Inkling2 attention for one decoder layer.

    The query side (q, r, gate, sink) is projected from `x_q`, k from `x_k`
    and v from `x_v`. Local layers take one input for all three, except in the
    loop passes after the first, whose messages differ per input. Full layers
    use dynamic K/V convolutions and log-scaled q/r.

    A loop core layer has one KV cache and one conv state per pass:
    `conv_states[p]`, and this layer's KV cache for pass 0 or
    `pass_caches[p - 1]` after.
    """

    def __init__(
        self,
        config: Inkling2ModelConfig,
        is_local: bool,
        conv_state,
        prefix: str,
        pass_conv_states: tuple = (),
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.is_local = is_local
        num_heads = config.swa_num_attention_heads if is_local else config.num_attention_heads
        num_kv_heads = config.swa_num_key_value_heads if is_local else config.num_key_value_heads
        self.head_dim = config.swa_head_dim if is_local else config.head_dim
        self.d_rel = config.d_rel
        tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        assert num_heads % tp_size == 0 and num_kv_heads % tp_size == 0
        self.num_heads = num_heads // tp_size
        self.num_kv_heads = num_kv_heads // tp_size
        self.scaling = self.head_dim**-0.5
        self.sliding_window = config.sliding_window_size
        self.rel_extent = config.sliding_window_size if is_local else config.rel_extent
        self.window = config.sliding_window_size - 1 if is_local else -1
        self.qk_clip = config.qk_l2_clip_threshold
        self.rel_clip = config.rel_l2_clip_threshold
        self.eps = config.rms_norm_eps
        assert config.use_value_norm_no_gamma

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        # Query-side outputs first so full layers can project them in one GEMM.
        self.split_sizes = [
            q_size,
            self.num_heads * self.d_rel,
            q_size,
            self.num_heads,
            kv_size,
            kv_size,
        ]
        # Full layers project the query side alone; rounding its N up to 8
        # keeps the output rows 16-byte aligned for cuBLAS (the extra columns,
        # from the K rows, are discarded).
        q_side = sum(self.split_sizes[:4])
        self.q_side_splits = [*self.split_sizes[:4], -q_side % 8]
        pad_size = -sum(self.split_sizes) % 8
        if pad_size:
            self.split_sizes.append(pad_size)
        # Attention linears stay in bf16 (`quant_config=None`): the release
        # keeps everything but the routed experts in bf16.
        self.qkvr = MergedColumnParallelLinear(
            config.hidden_size,
            [size * tp_size for size in self.split_sizes],
            bias=False,
            prefix=f"{prefix}.qkvr",
            quant_config=None,
        )
        self.wo_ud = RowParallelLinear(
            q_size * tp_size,
            config.hidden_size,
            bias=False,
            reduce_results=False,
            prefix=f"{prefix}.wo_ud",
            quant_config=None,
        )
        self.rel_logits_proj = nn.Parameter(torch.empty(self.d_rel, self.rel_extent), requires_grad=False)
        # Full-head (replicated) clip scales: slicing per TP rank at use keeps
        # the checkpoint's tensor layout, so reloads stay in place.
        self.q_scale = nn.Parameter(torch.empty(num_heads, self.head_dim), requires_grad=False)
        self.r_scale = nn.Parameter(torch.empty(num_heads, self.d_rel), requires_grad=False)
        # Derived from `rel_logits_proj` after loading; written in place so
        # CUDA graphs keep reading the same storage across weight reloads.
        self.register_buffer(
            "rel_proj_shifted",
            torch.empty(8, self.d_rel + 1, self.rel_extent + 512, dtype=torch.bfloat16),
            persistent=False,
        )

        self.conv_state = conv_state
        self.conv_states = (conv_state, *pass_conv_states)
        self.pass_caches = nn.ModuleList(
            Inkling2AttentionPassCache(
                f"{prefix}.pass{pass_idx}",
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.sliding_window if is_local else None,
            )
            for pass_idx in range(1, len(self.conv_states))
        )
        self.k_sconv, self.v_sconv = (
            Inkling2ShortConv(
                kv_size,
                config.sconv_kernel_size,
                conv_state,
                stream,
                dynamic_input_size=None if is_local else config.hidden_size,
                num_heads=self.num_kv_heads,
            )
            for stream in (K_STREAM, V_STREAM)
        )

        vllm_config = get_current_vllm_config()
        assert vllm_config.cache_config.cache_dtype in ("auto", "bfloat16")
        self.kv_cache = torch.tensor([])
        forward_context = vllm_config.compilation_config.static_forward_context
        if prefix in forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        forward_context[prefix] = self

    def process_weights_after_loading(self, dtype: torch.dtype | None = None) -> None:
        self.rel_proj_shifted.copy_(shift_rel_proj(self.rel_logits_proj, self.is_local))

    def get_attn_backend(self) -> type[AttentionBackend]:
        return Inkling2AttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return _kv_cache_spec(
            vllm_config, self.num_kv_heads, self.head_dim, self.sliding_window if self.is_local else None
        )

    def _pass_cache(self, pass_idx: int) -> tuple[str, torch.Tensor]:
        """The attention metadata key and KV cache of loop pass `pass_idx`."""
        if pass_idx == 0:
            return self.prefix, self.kv_cache
        cache = self.pass_caches[pass_idx - 1]
        return cache.prefix, cache.kv_cache

    def _project(self, x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor) -> list[torch.Tensor]:
        weight = self.qkvr.weight
        if x_k is x_q and x_v is x_q:
            projected = nn.functional.linear(x_q, weight)
            return list(projected.split(self.split_sizes, dim=-1)[:6])
        q_side, kv_size = sum(self.split_sizes[:4]), self.split_sizes[4]
        k_end = q_side + kv_size
        projected = nn.functional.linear(x_q, weight[: sum(self.q_side_splits)])
        k = nn.functional.linear(x_k, weight[q_side:k_end])
        v = nn.functional.linear(x_v, weight[k_end : k_end + kv_size])
        return [*projected.split(self.q_side_splits, dim=-1)[:4], k, v]

    def _split_kv_cache(self, kv_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_dim, dim=-1)
        return (
            canonicalize_singleton_dim_strides(key_cache),
            canonicalize_singleton_dim_strides(value_cache),
        )

    def forward(
        self,
        positions: torch.Tensor,
        x_q: torch.Tensor,
        x_k: torch.Tensor | None = None,
        x_v: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        pass_idx: int = 0,
    ) -> torch.Tensor:
        """Return this rank's bf16 `(T, hidden)` partial output (to be
        all-reduced over TP).

        Without `x_k` and `x_v`, q/k/v are read from `x_q`. Full layers take
        the fp32 log-scaling `tau` for q/r. `pass_idx` picks the loop pass's
        caches.
        """
        if x_k is None:
            assert x_v is None and self.is_local
            x_k = x_v = x_q
        assert x_v is not None
        q, r, gate, sinks, k, v = self._project(x_q, x_k, x_v)
        output = torch.empty(gate.shape, dtype=q.dtype, device=q.device)
        self._attention(
            positions,
            q,
            r,
            gate,
            sinks,
            k,
            v,
            self.k_sconv.dynamic_taps(x_k),
            self.v_sconv.dynamic_taps(x_v),
            tau,
            output,
            pass_idx,
        )
        return nn.functional.linear(output, self.wo_ud.weight)

    @eager_break_during_capture
    def _attention(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        r: torch.Tensor,
        gate: torch.Tensor,
        sinks: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_dynamic: torch.Tensor | None,
        v_dynamic: torch.Tensor | None,
        tau: torch.Tensor | None,
        output: torch.Tensor,
        pass_idx: int = 0,
    ) -> None:
        """Write the gated attention output to `output`.

        The prologue's layout and the attention plan follow the batch's
        requests, so this runs eagerly in piecewise CUDA graphs, which are
        captured on a dummy batch and replayed for any batch of the same
        token count.
        """
        attn_metadata = get_forward_context().attn_metadata
        layer_name, kv_cache = self._pass_cache(pass_idx)
        conv_state = self.conv_states[pass_idx]
        if not isinstance(attn_metadata, dict) or kv_cache.numel() == 0:
            output.zero_()
            return
        md = cast(FlashAttentionMetadata, attn_metadata[layer_name])
        conv_md = conv_metadata(conv_state)
        assert conv_md is not None
        num_tokens = q.shape[0]
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        key_cache, value_cache = self._split_kv_cache(kv_cache)
        clipped_q = q.new_empty(q.shape)
        # Sheared as `rel_attention` reads it (see attention_prologue).
        rel_logits = q.new_empty((num_tokens + 128, self.num_heads, self.rel_extent + 256))
        h0 = self.tp_rank * self.num_heads
        attention_prologue(
            q,
            r.view(num_tokens, self.num_heads, self.d_rel),
            k,
            v,
            self.q_scale.narrow(0, h0, self.num_heads),
            self.r_scale.narrow(0, h0, self.num_heads),
            tau,
            self.rel_proj_shifted,
            self.k_sconv.conv_weight,
            self.v_sconv.conv_weight,
            k_dynamic,
            v_dynamic,
            conv_state.kv_cache,
            positions,
            conv_md.block_table,
            conv_md.seq_idx,
            conv_md.slot_mapping,
            conv_md.query_start,
            key_cache,
            value_cache,
            md.slot_mapping,
            clipped_q,
            rel_logits,
            md.query_start_loc,
            md.seq_lens,
            clip_threshold=self.qk_clip,
            rel_threshold=self.rel_clip,
            norm_eps=self.eps,
        )
        # Decode and verify batches split the keys; the split count follows
        # the batch alone, so it holds for every replay of a CUDA graph.
        num_splits = plan_decode_attention(
            self.is_local,
            md.seq_lens.shape[0],
            md.max_query_len,
            self.num_heads,
            self.num_kv_heads,
            self.rel_extent,
        )
        if num_splits:
            rel_decode_attention(
                clipped_q,
                key_cache,
                value_cache,
                rel_logits,
                gate,
                sinks,
                output,
                block_table=md.block_table,
                cache_seqlens=md.seq_lens,
                cu_seqlens_q=md.query_start_loc,
                max_query_len=md.max_query_len,
                num_actual_tokens=md.num_actual_tokens,
                softmax_scale=self.scaling,
                is_local=self.is_local,
                num_splits=num_splits,
            )
            return
        gated_rel_attention(
            clipped_q,
            key_cache,
            value_cache,
            rel_logits,
            gate,
            sinks,
            output,
            block_table=md.block_table,
            cache_seqlens=md.seq_lens,
            cu_seqlens_q=md.query_start_loc,
            max_seqlen_q=md.max_query_len,
            num_actual_tokens=md.num_actual_tokens,
            softmax_scale=self.scaling,
            window=self.window,
        )
