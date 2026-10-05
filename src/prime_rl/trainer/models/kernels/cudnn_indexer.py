"""cuDNN frontend FP8 indexer for DSA sparse token selection (SM90 / SM100 / SM103).

Same contract and scoring formula as `fp8_indexer`, I_{t,s} = Σ_j w_{t,j} · ReLU(q_{t,j} · k_s),
with the same UE8M0 per-token FP8 quantization. On SM90 the cuDNN indexer forward writes dense
scores and the cuDNN radix top-k selects from them; on SM100 / SM103 the fused cuDNN kernel scores
and selects without materializing the dense scores. Packed documents become THD segments, so each
document is scored only against its own keys.
"""

import math

import torch

from prime_rl.trainer.models.kernels.fp8_indexer import per_token_group_quant_fp8

# Upper bound on fp32 logits per top-k launch: the kernel allocates a scratch buffer of twice the
# logits' size, which would be 16 GiB for a 16k x 131k CP shard in one launch.
TOPK_CHUNK_ELEMS = 1 << 27

_SM100_CAPABILITIES = ((10, 0), (10, 3))


def cudnn_indexer_arch(device: torch.device) -> str | None:
    """The cuDNN indexer variant for this device ("sm90" / "sm100"), or None if unsupported."""
    capability = torch.cuda.get_device_capability(device)
    if capability[0] == 9:
        return "sm90"
    if capability in _SM100_CAPABILITIES:
        return "sm100"
    return None


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

    # Radix select is ~5x faster than torch.topk on real indexer scores, but ~1.4x slower when the
    # k-th value is tied across most of a row (e.g. the all-zero scores of a degenerate random init).
    num_rows, num_cols = logits.shape
    # When every document is shorter than topk there are fewer columns than topk; pad with -1.
    select = min(topk, num_cols)
    chunk_rows = max(256, TOPK_CHUNK_ELEMS // num_cols)
    indices = torch.cat(
        [
            indexer_top_k_wrapper(
                logits[lo : lo + chunk_rows], lengths[lo : lo + chunk_rows], select, return_val=False
            )["indices"]
            for lo in range(0, num_rows, chunk_rows)
        ]
    )
    if select < topk:
        indices = torch.nn.functional.pad(indices, (0, topk - select), value=-1)
    return indices


@torch.library.custom_op("prime_rl::cudnn_fp8_indexer", mutates_args=())
def cudnn_fp8_indexer(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """cuDNN FP8 indexer: UE8M0 quantization + cuDNN scoring and top-k (see module docstring).

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
    S_q, H, D = q.shape
    S_k = k.shape[0]

    q_fp8, q_scales = per_token_group_quant_fp8(q.reshape(S_q * H, D).contiguous(), group_size=D, use_ue8m0=True)
    k_fp8, k_scales = per_token_group_quant_fp8(k.contiguous(), group_size=D, use_ue8m0=True)
    q_fp8, q_scales = q_fp8.view(S_q, H, D), q_scales.view(S_q, H)
    segments = _document_segments(ks, ke)

    if cudnn_indexer_arch(q.device) == "sm90":
        local = _sm90_top_k(q_fp8, q_scales, k_fp8, k_scales, w, ks, ke, segments, topk)
    else:
        local = _sm100_fused_top_k(q_fp8, q_scales, k_fp8, k_scales, w, segments, topk)
    lengths = ke - ks
    # Slots past a short row's length come back as -1.
    valid = (local >= 0) & (local < lengths[:, None])
    return torch.where(valid, local + ks[:, None], S_k).to(torch.int32)


def _sm90_top_k(q_fp8, q_scales, k_fp8, k_scales, w, ks, ke, segments, topk) -> torch.Tensor:
    """Dense doc-local scores from the SM90 indexer forward, then the radix top-k."""
    from cudnn.deepseek_sparse_attention.indexer_forward._interface_sm90 import indexer_fwd

    cu_seqlens_q, cu_seqlens_k, q_causal_offsets, max_seqlen_q, max_seqlen_k = segments
    # Fold q's descale into the weights, as fp8_indexer does. An fp32 W selects the kernel's
    # pre-scaled path, which then skips q_scale (indexer_fwd_sm90 `use_fp8_prescaled_w`).
    w = w.float() * q_scales
    # Rows 32-byte aligned for the top-k kernel's vector loads.
    num_cols = (max_seqlen_k + 7) // 8 * 8

    # ratio=1 is the plain token-level causal mask; scores are doc-local: column j is key ks + j.
    logits = indexer_fwd(
        q_fp8,
        k_fp8.unsqueeze(1),
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
    return _chunked_top_k(logits, ke - ks, topk)


def _sm100_fused_top_k(q_fp8, q_scales, k_fp8, k_scales, w, segments, topk) -> torch.Tensor:
    """Doc-local top-k from the fused SM100 MXFP8 indexer (scores never materialized densely).

    Untested: no SM100 / SM103 hardware has run this path.

    The kernel takes MXFP8 (one E8M0 scale per 32 elements). Repeating each token's single UE8M0
    scale over its four 32-element blocks keeps the exact per-token FP8 values and scales of the
    SM90 / Triton (and vLLM) indexer, rather than requantizing at a finer granularity.
    """
    from cudnn.deepseek_sparse_attention import indexer_forward_top_k_wrapper
    from cudnn.deepseek_sparse_attention.utils.sm100.mxfp8_scale_utils import (
        make_scale_cu_seqlens_padded,
        pack_k_scale_thd,
        pack_q_scale_thd,
    )

    cu_seqlens_q, cu_seqlens_k, q_causal_offsets, max_seqlen_q, max_seqlen_k = segments
    S_q, H, D = q_fp8.shape
    blocks = D // 32

    def mx_scales(scales: torch.Tensor) -> torch.Tensor:
        return scales.unsqueeze(-1).expand(*scales.shape, blocks).to(torch.float8_e8m0fnu)

    # Scale storage is addressed from zero; the segments' keys start at the first document's `ks`.
    k_lo, k_hi = cu_seqlens_k[0].item(), cu_seqlens_k[-1].item()
    cu_seqlens_k_rel = cu_seqlens_k - k_lo
    # Each Q span times H, and each K span, must fill whole 128-row scale tiles.
    cu_seqlens_q_scale = make_scale_cu_seqlens_padded(cu_seqlens_q, 128 // math.gcd(128, H))
    cu_seqlens_k_scale = make_scale_cu_seqlens_padded(cu_seqlens_k_rel, 128)
    q_scale_packed = pack_q_scale_thd(mx_scales(q_scales), cu_seqlens_q, cu_seqlens_q_scale, qhead_per_kv_head=H)
    k_scale_packed = pack_k_scale_thd(mx_scales(k_scales[k_lo:k_hi].view(-1, 1)), cu_seqlens_k_rel, cu_seqlens_k_scale)

    return indexer_forward_top_k_wrapper(
        q_fp8,
        k_fp8.unsqueeze(1),
        w,
        topk,
        ratio=1,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        q_causal_offsets=q_causal_offsets,
        precision="mxfp8",
        q_scale=q_scale_packed,
        k_scale=k_scale_packed,
        cu_seqlens_q_scale_padded=cu_seqlens_q_scale,
        cu_seqlens_k_scale_padded=cu_seqlens_k_scale,
        return_softmax=False,
        topk_indices_global=False,
    )["indices"]


@cudnn_fp8_indexer.register_fake
def _cudnn_fp8_indexer_fake(q, k, w, ks, ke, topk):
    return q.new_empty((q.shape[0], topk), dtype=torch.int32)
