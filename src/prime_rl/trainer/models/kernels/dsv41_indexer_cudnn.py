"""DeepSeek-V4.1 Lightning Indexer on cuDNN's fused Blackwell scoring + top-k.

cuDNN scores `sum_h relu(q_h . k) * w_h` and selects the top-k without materializing more than
the causal triangle of scores, for any compress ratio. It reads its documents as varlen segments:
segment `b` holds queries `[cu_q[b], cu_q[b + 1])` and entries `[cu_k[b], cu_k[b + 1])`, and its
query `i` reads the first `(q_causal_offsets[b] + i + 1) // ratio` of those entries. A segment is
the part of one document whose queries this rank holds, so the same layout serves a whole packed
row and a context-parallel shard of it.

It has no counterpart of the two-level candidate filter, so the candidate source layer and the
layers consuming its candidates stay on `dsv41_index_topk`.

The kernel compiles once per segment count and maximum segment lengths, which change with every
packing. `IndexerSegments.build` rounds all three up to powers of two, padding with empty
segments, so a run compiles a handful of variants instead of one per step.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

try:
    from cudnn.deepseek_sparse_attention import indexer_forward_top_k_wrapper
except ImportError:
    indexer_forward_top_k_wrapper = None  # type: ignore


def cudnn_indexer_available(num_heads: int, head_dim: int) -> bool:
    """cuDNN's fused indexer serves 32 or 64 heads of width 128 on SM100 / SM103."""
    return (
        indexer_forward_top_k_wrapper is not None
        and num_heads in (32, 64)
        and head_dim == 128
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() in ((10, 0), (10, 3))
    )


def _next_power_of_2(n: int) -> int:
    return 1 << max(0, n - 1).bit_length()


@dataclass(frozen=True)
class IndexerSegments:
    """This rank's queries cut into per-document segments, padded to bucketed sizes."""

    cu_seqlens_q: Tensor  # (n_segments + 1,) int32, query boundaries counted from this rank's first query
    cu_seqlens_k: Tensor  # (n_segments + 1,) int32, entry boundaries in the packed row
    q_causal_offsets: Tensor  # (n_segments,) int32, document-local position of each segment's first query
    max_seqlen_q: int
    max_seqlen_k: int

    @classmethod
    def build(cls, *, cu_seqlens: Tensor, q_start: int, n_queries: int, compress_rate: int) -> "IndexerSegments":
        """`cu_seqlens` covers the whole packed row; this rank holds queries `[q_start, q_start + n_queries)`."""
        cu = cu_seqlens.to(torch.int64)
        starts, ends = cu[:-1], cu[1:]
        q_end = q_start + n_queries
        docs = ((ends > q_start) & (starts < q_end)).nonzero().squeeze(1)
        seg_start = starts[docs].clamp_min(q_start)
        seg_end = ends[docs].clamp_max(q_end)
        n_entries = (ends - starts) // compress_rate
        first_entry = n_entries.cumsum(0) - n_entries

        seg_q = seg_end - seg_start
        seg_k = n_entries[docs]
        max_q, max_k, n_segments = (
            int(x) for x in torch.stack([seg_q.max(), seg_k.max(), docs.new_tensor(docs.numel())]).tolist()
        )
        max_q = _next_power_of_2(max_q)
        max_k = max(_next_power_of_2(max_k), max_q // compress_rate)
        pad = _next_power_of_2(n_segments) - n_segments

        zero = seg_q.new_zeros(1)
        cu_q = torch.cat([zero, seg_q.cumsum(0)])
        cu_k = torch.cat([first_entry[docs[:1]], first_entry[docs] + seg_k])
        offsets = seg_start - starts[docs]
        # Empty segments past the last real one read nothing and add no queries.
        cu_q = torch.cat([cu_q, cu_q[-1:].expand(pad)])
        cu_k = torch.cat([cu_k, cu_k[-1:].expand(pad)])
        offsets = torch.nn.functional.pad(offsets, (0, pad))
        return cls(
            cu_seqlens_q=cu_q.int(),
            cu_seqlens_k=cu_k.int(),
            q_causal_offsets=offsets.int(),
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
        )


@torch.library.custom_op("prime_rl::dsv41_index_topk_cudnn", mutates_args=())
def dsv41_index_topk_cudnn(
    q: Tensor,
    k: Tensor,
    w: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    q_causal_offsets: Tensor,
    topk: int,
    compress_rate: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> Tensor:
    """`q` `(S_q, H, D)`, `k` `(S_k, D)`, `w` `(S_q, H)` -> `(S_q, topk)` int64 entry indices into the
    packed row, -1 where a query has fewer readable entries."""
    out = indexer_forward_top_k_wrapper(
        q.contiguous(),
        k.unsqueeze(1).contiguous(),
        w.to(torch.bfloat16).contiguous(),
        top_k=topk,
        ratio=compress_rate,
        sm_scale=1.0,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        q_causal_offsets=q_causal_offsets,
        return_softmax=False,
        topk_indices_global=True,
        # Ties at the k-th score otherwise resolve differently from run to run; this costs nothing measurable.
        deterministic=True,
    )
    return out["indices"].long()


@dsv41_index_topk_cudnn.register_fake
def _dsv41_index_topk_cudnn_fake(
    q, k, w, cu_seqlens_q, cu_seqlens_k, q_causal_offsets, topk, compress_rate, max_seqlen_q, max_seqlen_k
):
    return q.new_empty((q.shape[0], topk), dtype=torch.int64)


__all__ = ["IndexerSegments", "cudnn_indexer_available", "dsv41_index_topk_cudnn"]
