"""Inkling 2 attention: grouped-query attention with a per-query relative bias, a per-token sink and an output gate.

The score of query ``i`` on key ``j`` is ``scale * q_i . k_j + rel[i, h, i - j]``, where the relative bias reaches
back ``rel_extent`` tokens and is zero beyond. No flash-attention kernel carries that bias and its gradient, so
the core is computed densely in FP32, in query chunks, with a backward that recomputes each chunk instead of
keeping the score matrices.
"""

import bisect
import math

import torch
from torch import nn

from prime_rl.trainer.models.inkling2_loop.configuration_inkling2_loop import (
    SLIDING_ATTENTION,
    Inkling2LoopTextConfig,
)
from prime_rl.trainer.models.inkling2_loop.layers import (
    DynamicShortConv,
    StaticShortConv,
    l2_clip,
    log_length_scale,
    unweighted_rms_norm,
)
from prime_rl.trainer.models.layers.attn import quadratic_attention_flops

# Scores (heads x queries x keys) held at once. Forward and backward keep a handful of tensors of this size.
SCORE_ELEMENTS_PER_CHUNK = 1 << 26


def _query_chunks(first_key: torch.Tensor, num_heads: int) -> list[tuple[int, int, int]]:
    """Split the queries into ``(start, stop, key_start)`` chunks whose score tensors stay under the budget.

    ``first_key`` is non-decreasing, so a chunk's keys are ``key_start = first_key[start]`` up to its last query,
    and every query of the chunk is scored against all of them. A chunk therefore ends at the first query that
    sees none of the keys before the chunk: the next document's first token, or on a sliding layer the query one
    window after the chunk's first. From that query on, every score against the keys before the chunk would be
    masked.

    The plan follows the row's document layout, read from ``first_key`` on the host: a plan from the shapes alone
    would score the earlier documents of a packed row too. A CUDA graph cannot record that read
    (``cuda_graph_blocker``).
    """
    total = first_key.shape[0]
    budget = SCORE_ELEMENTS_PER_CHUNK / num_heads
    first_keys = first_key.tolist()
    chunks = []
    start = 0
    while start < total:
        key_start = first_keys[start]
        before = start - key_start
        # Largest `size` with size * (before + size) <= budget.
        size = max(1, int((math.sqrt(before * before + 4 * budget) - before) / 2))
        stop = min(total, start + size)
        stop = bisect.bisect_left(first_keys, start, start + 1, stop)
        chunks.append((start, stop, key_start))
        start = stop
    return chunks


def _chunk_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rel: torch.Tensor,
    first_key: torch.Tensor,
    scale: float,
    chunk: tuple[int, int, int],
    dtype: torch.dtype,
):
    """One query chunk's operands, grouped by KV head, and its attention probabilities.

    Returns ``(q_g [Hkv, G, C, D], k_g [Hkv, 1, Lk, D], v_g [Hkv, 1, Lk, D], probs [Hkv, G, C, Lk],
    lse [Hkv, G, C], rel_index [C, Lk], in_band [C, Lk])``.
    """
    start, stop, key_start = chunk
    num_heads, num_kv_heads = q.shape[1], k.shape[1]
    group = num_heads // num_kv_heads
    extent = rel.shape[-1]

    q_g = q[start:stop].to(dtype).unflatten(1, (num_kv_heads, group)).permute(1, 2, 0, 3)
    k_g = k[key_start:stop].to(dtype).transpose(0, 1).unsqueeze(1)
    v_g = v[key_start:stop].to(dtype).transpose(0, 1).unsqueeze(1)
    rel_g = rel[start:stop].to(dtype).unflatten(1, (num_kv_heads, group)).permute(1, 2, 0, 3)

    query_positions = torch.arange(start, stop, device=q.device)
    key_positions = torch.arange(key_start, stop, device=q.device)
    distance = query_positions[:, None] - key_positions[None, :]
    in_band = (distance >= 0) & (distance < extent)
    rel_index = distance.clamp(0, extent - 1)
    visible = (distance >= 0) & (key_positions[None, :] >= first_key[start:stop, None])

    scores = torch.matmul(q_g, k_g.transpose(-1, -2)) * scale
    bias = rel_g.gather(-1, rel_index.expand(num_kv_heads, group, -1, -1))
    scores = scores + bias.masked_fill(~in_band, 0.0)
    scores = scores.masked_fill(~visible, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.exp(scores - lse.unsqueeze(-1))
    return q_g, k_g, v_g, probs, lse, rel_index, in_band


@torch.library.custom_op("prime_rl_attn::inkling2_rel_attention", mutates_args=())
def _rel_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rel: torch.Tensor,
    first_key: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = torch.promote_types(q.dtype, torch.float32)
    num_tokens, num_heads, head_dim = q.shape
    out = torch.empty((num_tokens, num_heads, head_dim), dtype=dtype, device=q.device)
    lse = torch.empty((num_tokens, num_heads), dtype=dtype, device=q.device)
    for chunk in _query_chunks(first_key, num_heads):
        start, stop, _ = chunk
        _, _, v_g, probs, chunk_lse, _, _ = _chunk_scores(q, k, v, rel, first_key, scale, chunk, dtype)
        out[start:stop] = torch.matmul(probs, v_g).permute(2, 0, 1, 3).flatten(1, 2)
        lse[start:stop] = chunk_lse.permute(2, 0, 1).flatten(1, 2)
    return out, lse


@_rel_attention.register_fake
def _(q, k, v, rel, first_key, scale):
    dtype = torch.promote_types(q.dtype, torch.float32)
    return q.new_empty(q.shape, dtype=dtype), q.new_empty(q.shape[:2], dtype=dtype)


@torch.library.custom_op("prime_rl_attn::inkling2_rel_attention_backward", mutates_args=())
def _rel_attention_backward(
    grad_out: torch.Tensor,
    grad_lse: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rel: torch.Tensor,
    first_key: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = torch.promote_types(q.dtype, torch.float32)
    num_heads, num_kv_heads = q.shape[1], k.shape[1]
    group = num_heads // num_kv_heads
    grad_q = torch.zeros(q.shape, dtype=dtype, device=q.device)
    grad_k = torch.zeros(k.shape, dtype=dtype, device=q.device)
    grad_v = torch.zeros(v.shape, dtype=dtype, device=q.device)
    grad_rel = torch.zeros(rel.shape, dtype=dtype, device=q.device)
    for chunk in _query_chunks(first_key, num_heads):
        start, stop, key_start = chunk
        q_g, k_g, v_g, probs, _, rel_index, in_band = _chunk_scores(q, k, v, rel, first_key, scale, chunk, dtype)
        grad_out_g = grad_out[start:stop].to(dtype).unflatten(1, (num_kv_heads, group)).permute(1, 2, 0, 3)
        grad_lse_g = grad_lse[start:stop].to(dtype).unflatten(1, (num_kv_heads, group)).permute(1, 2, 0)
        out_g = torch.matmul(probs, v_g)
        # d loss / d score: softmax backward of the output, plus the log-sum-exp's own gradient.
        grad_scores = torch.matmul(grad_out_g, v_g.transpose(-1, -2))
        grad_scores = grad_scores - (grad_out_g * out_g).sum(-1, keepdim=True) + grad_lse_g.unsqueeze(-1)
        grad_scores = probs * grad_scores

        grad_q[start:stop] = (torch.matmul(grad_scores, k_g) * scale).permute(2, 0, 1, 3).flatten(1, 2)
        grad_k[key_start:stop] += (torch.matmul(grad_scores.transpose(-1, -2), q_g) * scale).sum(1).transpose(0, 1)
        grad_v[key_start:stop] += torch.matmul(probs.transpose(-1, -2), grad_out_g).sum(1).transpose(0, 1)
        grad_rel_g = torch.zeros((num_kv_heads, group, stop - start, rel.shape[-1]), dtype=dtype, device=q.device)
        grad_rel_g.scatter_add_(
            -1, rel_index.expand(num_kv_heads, group, -1, -1), grad_scores.masked_fill(~in_band, 0.0)
        )
        grad_rel[start:stop] = grad_rel_g.permute(2, 0, 1, 3).flatten(1, 2)
    return grad_q.to(q.dtype), grad_k.to(k.dtype), grad_v.to(v.dtype), grad_rel.to(rel.dtype)


@_rel_attention_backward.register_fake
def _(grad_out, grad_lse, q, k, v, rel, first_key, scale):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v), torch.empty_like(rel)


def _rel_attention_setup_context(ctx, inputs, output) -> None:
    q, k, v, rel, first_key, scale = inputs
    ctx.save_for_backward(q, k, v, rel, first_key)
    ctx.scale = scale
    ctx.output_shapes = (output[0].shape, output[1].shape)
    ctx.output_dtype = output[0].dtype


def _rel_attention_autograd(ctx, grad_out: torch.Tensor | None, grad_lse: torch.Tensor | None):
    q, k, v, rel, first_key = ctx.saved_tensors
    out_shape, lse_shape = ctx.output_shapes
    if grad_out is None:
        grad_out = q.new_zeros(out_shape, dtype=ctx.output_dtype)
    if grad_lse is None:
        grad_lse = q.new_zeros(lse_shape, dtype=ctx.output_dtype)
    grad_q, grad_k, grad_v, grad_rel = _rel_attention_backward(grad_out, grad_lse, q, k, v, rel, first_key, ctx.scale)
    return grad_q, grad_k, grad_v, grad_rel, None, None


_rel_attention.register_autograd(_rel_attention_autograd, setup_context=_rel_attention_setup_context)


def rel_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rel: torch.Tensor, first_key: torch.Tensor, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal grouped-query attention with a relative bias, in FP32.

    Args:
        q: ``[T, Hq, D]`` queries.
        k: ``[T, Hkv, D]`` keys. ``Hq`` is a multiple of ``Hkv``; query head ``h`` reads KV head ``h // (Hq // Hkv)``.
        v: ``[T, Hkv, D]`` values.
        rel: ``[T, Hq, E]``. ``rel[i, h, d]`` is added to query ``i``'s score on the key ``d`` tokens back.
        first_key: ``[T]`` int64, non-decreasing: the first key each query sees (its document's start, or its
            sliding window's). Query ``i`` attends keys ``first_key[i]..i``.
        scale: Multiplies ``q . k``.

    Returns:
        ``out [T, Hq, D]`` and the log-sum-exp of each query's scores ``lse [T, Hq]``, both FP32.
    """
    return _rel_attention(q, k, v, rel, first_key, scale)


class Inkling2Attention(nn.Module):
    """One attention layer. Sliding layers see ``sliding_window_size`` keys and run static K/V convolutions; full
    layers see their whole document, scale queries and relative states with the log of the position, and run K/V
    convolutions whose taps depend on the token."""

    def __init__(self, config: Inkling2LoopTextConfig, layer_idx: int, num_visits: int) -> None:
        super().__init__()
        self.is_sliding = config.layer_types[layer_idx] == SLIDING_ATTENTION
        self.num_heads = config.swa_num_attention_heads if self.is_sliding else config.num_attention_heads
        self.num_kv_heads = config.swa_num_key_value_heads if self.is_sliding else config.num_key_value_heads
        self.head_dim = config.swa_head_dim if self.is_sliding else config.head_dim
        self.d_rel = config.d_rel
        self.sliding_window = config.sliding_window_size if self.is_sliding else None
        self.rel_extent = config.sliding_window_size if self.is_sliding else config.rel_extent
        self.qk_l2_clip_threshold = config.qk_l2_clip_threshold
        self.rel_l2_clip_threshold = config.rel_l2_clip_threshold
        self.value_norm_eps = config.rms_norm_eps
        self.log_scaling = (config.log_scaling_n_floor, config.log_scaling_alpha) if not self.is_sliding else None
        self.scaling = self.head_dim**-0.5
        self.num_visits = num_visits

        hidden_size, kv_dim = config.hidden_size, self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_dim, bias=False)
        self.rel_proj = nn.Linear(hidden_size, self.num_heads * self.d_rel, bias=False)
        self.gate_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        self.sink_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)
        self.q_scale = nn.Parameter(torch.ones(self.num_heads, self.head_dim))
        self.rel_scale = nn.Parameter(torch.ones(self.num_heads, self.d_rel))
        # Bias-vs-distance profiles that each token's relative states mix. Column `d` is `d` tokens back.
        self.rel_table = nn.Parameter(torch.empty(self.d_rel, self.rel_extent))
        if self.is_sliding:
            self.k_conv = StaticShortConv(kv_dim, config.sconv_kernel_size)
            self.v_conv = StaticShortConv(kv_dim, config.sconv_kernel_size)
        else:
            self.k_conv = DynamicShortConv(kv_dim, hidden_size, self.num_kv_heads, config.sconv_kernel_size)
            self.v_conv = DynamicShortConv(kv_dim, hidden_size, self.num_kv_heads, config.sconv_kernel_size)

    def attention_flops(self, seq_len: int) -> int:
        return self.num_visits * quadratic_attention_flops(
            num_heads=self.num_heads,
            qk_head_dim=self.head_dim,
            v_head_dim=self.head_dim,
            seq_len=seq_len,
            attended_tokens=min(seq_len, self.sliding_window or seq_len),
        )

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        positions: torch.Tensor,
        first_key: torch.Tensor,
    ) -> torch.Tensor:
        """``*_states [1, T, hidden]`` are the normed inputs of the query, key and value projections (one tensor
        outside the loop's later passes). ``positions [T]`` counts tokens inside each document."""
        num_tokens = query_states.shape[1]
        dtype = query_states.dtype

        q = self.q_proj(query_states)
        rel_states = self.rel_proj(query_states)
        gate = self.gate_proj(query_states)
        sink = self.sink_proj(query_states)
        k = self.k_conv(self.k_proj(key_states), key_states, positions)
        v = self.v_conv(self.v_proj(value_states), value_states, positions)

        length_scale = None
        if self.log_scaling is not None:
            length_scale = log_length_scale(positions, *self.log_scaling)
        q = l2_clip(q.view(num_tokens, self.num_heads, self.head_dim), self.qk_l2_clip_threshold, self.q_scale)
        k = l2_clip(k.view(num_tokens, self.num_kv_heads, self.head_dim), self.qk_l2_clip_threshold)
        v = unweighted_rms_norm(v.view(num_tokens, self.num_kv_heads, self.head_dim), self.value_norm_eps).to(dtype)
        rel_states = l2_clip(
            rel_states.view(num_tokens, self.num_heads, self.d_rel), self.rel_l2_clip_threshold, self.rel_scale
        )
        if length_scale is not None:
            q = (q.float() * length_scale).to(dtype)
            rel_states = (rel_states.float() * length_scale).to(dtype)
        rel = torch.matmul(rel_states, self.rel_table.to(dtype))

        out, lse = rel_attention(q, k, v, rel, first_key, self.scaling)
        # The sink is one more logit per query with a zero value: it scales the output by the mass the keys keep.
        out = out * torch.sigmoid(gate.float().view(num_tokens, self.num_heads, self.head_dim))
        out = out * torch.sigmoid(lse - sink.float().view(num_tokens, self.num_heads)).unsqueeze(-1)
        return self.o_proj(out.reshape(1, num_tokens, -1).to(dtype))


__all__ = ["Inkling2Attention", "rel_attention"]
