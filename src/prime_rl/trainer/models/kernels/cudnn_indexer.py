"""cuDNN frontend FP8 indexer for DSA sparse token selection (SM90).

Same contract and scoring formula as `fp8_indexer`, I_{t,s} = Σ_j w_{t,j} · ReLU(q_{t,j} · k_s),
with the same UE8M0 per-token FP8 quantization, but scored by the cuDNN SM90 indexer forward and
selected by the cuDNN radix top-k. Packed documents become THD segments, so each document is scored
only against its own keys.
"""

import torch

from prime_rl.trainer.models.kernels.fp8_indexer import per_token_group_quant_fp8

# Upper bound on fp32 logits per top-k launch: the kernel allocates a scratch buffer of twice the
# logits' size, which would be 16 GiB for a 16k x 131k CP shard in one launch.
TOPK_CHUNK_ELEMS = 1 << 27


def _document_segments(ks: torch.Tensor, ke: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """THD segments of the queries, one per document (run of equal `ks`).

    Within a document `ke` advances by one per query, so the segment's keys are `[ks, ke[-1])` and
    its first query sits at doc-local position `ke[0] - ks - 1`. Consecutive documents' key ranges
    abut, so the key segments tile the gathered K from the first document's start.
    """
    S_q = ks.shape[0]
    is_start = torch.ones(S_q, dtype=torch.bool, device=ks.device)
    is_start[1:] = ks[1:] != ks[:-1]
    starts = is_start.nonzero().squeeze(1).to(torch.int32)
    cu_seqlens_q = torch.cat([starts, starts.new_tensor([S_q])])
    cu_seqlens_k = torch.cat([ks[starts], ke[-1:]])
    q_causal_offsets = ke[starts] - ks[starts] - 1
    max_seqlen_q, max_seqlen_k = torch.stack([cu_seqlens_q.diff().max(), (ke - ks).max()]).tolist()
    return cu_seqlens_q, cu_seqlens_k, q_causal_offsets, max_seqlen_q, max_seqlen_k


def _chunked_top_k(logits: torch.Tensor, lengths: torch.Tensor, topk: int) -> torch.Tensor:
    from cudnn.deepseek_sparse_attention import indexer_top_k_wrapper

    num_rows, num_cols = logits.shape
    chunk_rows = max(256, TOPK_CHUNK_ELEMS // num_cols)
    return torch.cat(
        [
            indexer_top_k_wrapper(logits[lo : lo + chunk_rows], lengths[lo : lo + chunk_rows], topk, return_val=False)[
                "indices"
            ]
            for lo in range(0, num_rows, chunk_rows)
        ]
    )


@torch.library.custom_op("prime_rl::cudnn_fp8_indexer", mutates_args=())
def cudnn_fp8_indexer(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """cuDNN FP8 indexer: UE8M0 quantization + cuDNN SM90 scoring + cuDNN radix top-k.

    Args:
        q: [S_q, H, D] bf16 query vectors per head (H in {32, 64}, D = 128)
        k: [S_k, D] bf16 key vectors (shared across heads)
        w: [S_q, H] bf16 per-head weights
        ks: [S_q] int32 document start per query (in K's coordinate system)
        ke: [S_q] int32 causal end per query (= global position + 1); must advance by one per
            query within a document
        topk: number of top indices to return (<= 2048)

    Returns:
        [S_q, topk] int32 selected token indices per query (sentinel = S_k), in no particular order
    """
    from cudnn.deepseek_sparse_attention.indexer_forward._interface_sm90 import indexer_fwd

    S_q, H, D = q.shape
    S_k = k.shape[0]

    q_fp8, q_scales = per_token_group_quant_fp8(q.reshape(S_q * H, D).contiguous(), group_size=D, use_ue8m0=True)
    k_fp8, k_scales = per_token_group_quant_fp8(k.contiguous(), group_size=D, use_ue8m0=True)
    q_scales = q_scales.view(S_q, H)
    # Fold q's descale into the weights (fp32), as fp8_indexer does; K's descale goes to the kernel.
    w = w * q_scales

    cu_seqlens_q, cu_seqlens_k, q_causal_offsets, max_seqlen_q, max_seqlen_k = _document_segments(ks, ke)
    # Rows 32-byte aligned for the top-k kernel's vector loads.
    num_cols = (max_seqlen_k + 7) // 8 * 8

    # ratio=1 is the plain token-level causal mask; scores are doc-local: column j is key ks + j.
    logits = indexer_fwd(
        q_fp8.view(S_q, H, D),
        k_fp8.view(S_k, 1, D),
        w,
        ratio=1,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=num_cols,
        q_causal_offsets=q_causal_offsets,
        precision="fp8",
        q_scale=q_scales,
        k_scale=k_scales,
    )

    lengths = ke - ks
    local = _chunked_top_k(logits, lengths, topk)
    # Slots past a short row's length come back as -1.
    valid = (local >= 0) & (local < lengths[:, None])
    return torch.where(valid, local + ks[:, None], S_k).to(torch.int32)


@cudnn_fp8_indexer.register_fake
def _cudnn_fp8_indexer_fake(q, k, w, ks, ke, topk):
    return q.new_empty((q.shape[0], topk), dtype=torch.int32)
