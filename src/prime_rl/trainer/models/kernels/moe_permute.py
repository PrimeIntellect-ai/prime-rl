"""Expert-parallel local layout: received tokens <-> padded expert-grouped rows.

An expert-parallel dispatch hands each rank its received tokens once, with the `(rows, top_k)`
local expert ids each must visit (-1 for experts on other ranks) and the matching routing scores.
The grouped GEMM wants one row per (token, expert) pair, grouped by expert with every group padded
to the GEMM's alignment. A `PairLayout` maps between the two: `positions[r, k]` is the padded row of
pair `(r, k)` or -1, and `sources[p]` is the flat pair `r * top_k + k` padded row `p` holds, or -1
for padding.

Both directions are single-pass gathers, in backward too, so nothing needs atomics and the results
are deterministic:
- `gather_pairs`: padded rows from the received tokens (padding rows are zero), optionally scaled by
  the pair's score; backward sums each token's pairs.
- `reduce_pairs`: received tokens from padded rows, summing each token's pairs weighted by their
  scores in fp32; backward gathers `score * grad` back into the padded rows and computes the score
  gradients.
"""

from dataclasses import dataclass
from itertools import accumulate

import torch
import triton
import triton.language as tl

BLOCK_H = 1024


@dataclass(frozen=True)
class PairLayout:
    positions: torch.Tensor
    sources: torch.Tensor
    top_k: int


@torch.compiler.disable()
def build_pair_layout(
    expert_ids: torch.Tensor, pairs_per_expert: list[int], alignment: int
) -> tuple[PairLayout, torch.Tensor]:
    """Layout of `(rows, top_k)` local `expert_ids` (-1 elsewhere) given each local expert's pair count.

    Returns the layout and the padded `(num_local_experts,)` group sizes for the grouped GEMM.
    """
    rows, top_k = expert_ids.shape
    device = expert_ids.device
    num_experts = len(pairs_per_expert)
    padded = [(count + alignment - 1) // alignment * alignment for count in pairs_per_expert]
    padded_starts = torch.tensor([0, *accumulate(padded)], dtype=torch.int64, device=device)
    pair_starts = torch.tensor([0, *accumulate(pairs_per_expert)], dtype=torch.int64, device=device)

    flat = expert_ids.reshape(-1).long()
    # Pairs for other ranks sort after every local expert.
    key = torch.where(flat >= 0, flat, num_experts)
    order = torch.argsort(key, stable=True)
    sorted_key = key[order]
    rank_in_expert = torch.arange(flat.numel(), device=device) - pair_starts[sorted_key]
    valid = sorted_key < num_experts
    padded_row = torch.where(valid, padded_starts[sorted_key.clamp(max=num_experts - 1)] + rank_in_expert, -1)

    positions = torch.empty(flat.numel(), dtype=torch.int64, device=device)
    positions[order] = padded_row
    sources = torch.full((sum(padded),), -1, dtype=torch.int64, device=device)
    sources[padded_row[valid]] = order[valid]
    group_sizes = torch.tensor(padded, dtype=torch.int64, device=device)
    return PairLayout(positions.view(rows, top_k), sources, top_k), group_sizes


@triton.jit
def _gather_rows_kernel(src, sources, scale, out, H, TOP_K: tl.constexpr, HAS_SCALE: tl.constexpr, BLOCK: tl.constexpr):
    """`out[p] = scale[s] * src[s // TOP_K]` for `s = sources[p]`, zero where `s < 0`."""
    p = tl.program_id(0).to(tl.int64)
    s = tl.load(sources + p)
    valid = s >= 0
    row = tl.where(valid, s // TOP_K, 0)
    w = tl.load(scale + s, mask=valid, other=0.0) if HAS_SCALE else 1.0
    offs = tl.arange(0, BLOCK)
    for h0 in range(0, H, BLOCK):
        mask = h0 + offs < H
        v = tl.load(src + row * H + h0 + offs, mask=mask & valid, other=0.0)
        if HAS_SCALE:
            v = (v.to(tl.float32) * w).to(v.dtype)
        tl.store(out + p * H + h0 + offs, v, mask=mask)


@triton.jit
def _gather_sum_kernel(src, positions, w, out, H, TOP_K: tl.constexpr, HAS_W: tl.constexpr, BLOCK: tl.constexpr):
    """`out[r] = sum_k w[r, k] * src[positions[r, k]]` over `positions >= 0`, accumulated in fp32."""
    r = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    for h0 in range(0, H, BLOCK):
        mask = h0 + offs < H
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for k in tl.static_range(TOP_K):
            pos = tl.load(positions + r * TOP_K + k)
            v = tl.load(src + pos * H + h0 + offs, mask=mask & (pos >= 0), other=0.0).to(tl.float32)
            if HAS_W:
                v *= tl.load(w + r * TOP_K + k)
            acc += v
        tl.store(out + r * H + h0 + offs, acc.to(out.dtype.element_ty), mask=mask)


@triton.jit
def _pair_grad_kernel(
    by_row, sources, w, by_pair, d_pair, d_w, H, TOP_K: tl.constexpr, WRITE_D_PAIR: tl.constexpr, BLOCK: tl.constexpr
):
    """For padded row `p` holding pair `s = sources[p]` of row `r = s // TOP_K`:
    `d_pair[p] = w[s] * by_row[r]` (zero for padding) and `d_w[s] = by_row[r] . by_pair[p]`."""
    p = tl.program_id(0).to(tl.int64)
    s = tl.load(sources + p)
    valid = s >= 0
    row = tl.where(valid, s // TOP_K, 0)
    scale = tl.load(w + s, mask=valid, other=0.0) if WRITE_D_PAIR else 0.0
    offs = tl.arange(0, BLOCK)
    dot = tl.zeros([BLOCK], dtype=tl.float32)
    for h0 in range(0, H, BLOCK):
        mask = h0 + offs < H
        g = tl.load(by_row + row * H + h0 + offs, mask=mask & valid, other=0.0).to(tl.float32)
        if WRITE_D_PAIR:
            tl.store(d_pair + p * H + h0 + offs, (g * scale).to(d_pair.dtype.element_ty), mask=mask)
        o = tl.load(by_pair + p * H + h0 + offs, mask=mask & valid, other=0.0).to(tl.float32)
        dot += g * o
    tl.store(d_w + s, tl.sum(dot), mask=valid)


def _gather_rows(src: torch.Tensor, sources: torch.Tensor, scale: torch.Tensor | None, top_k: int) -> torch.Tensor:
    out = src.new_empty(sources.shape[0], src.shape[1])
    if out.shape[0]:
        _gather_rows_kernel[(out.shape[0],)](
            src, sources, scale if scale is not None else src, out, src.shape[1], top_k, scale is not None, BLOCK_H
        )
    return out


def _gather_sum(src: torch.Tensor, positions: torch.Tensor, w: torch.Tensor | None) -> torch.Tensor:
    out = src.new_empty(positions.shape[0], src.shape[1])
    if out.shape[0]:
        _gather_sum_kernel[(out.shape[0],)](
            src, positions, w if w is not None else src, out, src.shape[1], positions.shape[1], w is not None, BLOCK_H
        )
    return out


def _pair_grads(
    by_row: torch.Tensor,
    sources: torch.Tensor,
    w: torch.Tensor | None,
    by_pair: torch.Tensor,
    num_pairs: int,
    top_k: int,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    d_pair = by_pair.new_empty(by_pair.shape) if w is not None else None
    d_w = torch.zeros(num_pairs, dtype=torch.float32, device=by_row.device)
    if by_pair.shape[0]:
        _pair_grad_kernel[(by_pair.shape[0],)](
            by_row,
            sources,
            w if w is not None else d_w,
            by_pair,
            d_pair if d_pair is not None else by_pair,
            d_w,
            by_row.shape[1],
            top_k,
            w is not None,
            BLOCK_H,
        )
    return d_pair, d_w


class _GatherPairs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scores, positions, sources, top_k):
        scale = scores.float().contiguous().view(-1) if scores is not None else None
        ctx.save_for_backward(x, scale, positions, sources)
        ctx.top_k = top_k
        ctx.scores_shape = scores.shape if scores is not None else None
        return _gather_rows(x.contiguous(), sources, scale, top_k)

    @staticmethod
    def backward(ctx, grad):
        x, scale, positions, sources = ctx.saved_tensors
        grad = grad.contiguous()
        w = scale.view(ctx.scores_shape) if scale is not None else None
        d_x = _gather_sum(grad, positions, w)
        d_scores = None
        if scale is not None and ctx.needs_input_grad[1]:
            _, d_w = _pair_grads(x, sources, None, grad, scale.numel(), ctx.top_k)
            d_scores = d_w.view(ctx.scores_shape)
        return d_x, d_scores, None, None, None


class _ReducePairs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pairs, scores, positions, sources, top_k):
        pairs = pairs.contiguous()
        w = scores.float().contiguous() if scores is not None else None
        ctx.save_for_backward(pairs, w, positions, sources)
        ctx.top_k = top_k
        return _gather_sum(pairs, positions, w)

    @staticmethod
    def backward(ctx, grad):
        pairs, w, positions, sources = ctx.saved_tensors
        grad = grad.contiguous()
        if w is None:
            return _gather_rows(grad, sources, None, ctx.top_k), None, None, None, None
        d_pairs, d_w = _pair_grads(grad, sources, w.view(-1), pairs, w.numel(), ctx.top_k)
        return d_pairs, d_w.view(w.shape), None, None, None


@torch.compiler.disable()
def gather_pairs(x: torch.Tensor, layout: PairLayout, scores: torch.Tensor | None = None) -> torch.Tensor:
    """`(rows, H)` received tokens -> `(padded, H)` expert-grouped pairs, scaled by `scores` when given."""
    return _GatherPairs.apply(x, scores, layout.positions, layout.sources, layout.top_k)


@torch.compiler.disable()
def reduce_pairs(pairs: torch.Tensor, layout: PairLayout, scores: torch.Tensor | None = None) -> torch.Tensor:
    """`(padded, H)` expert outputs -> `(rows, H)`, each row the `scores`-weighted sum of its pairs."""
    return _ReducePairs.apply(pairs, scores, layout.positions, layout.sources, layout.top_k)


__all__ = ["PairLayout", "build_pair_layout", "gather_pairs", "reduce_pairs"]
