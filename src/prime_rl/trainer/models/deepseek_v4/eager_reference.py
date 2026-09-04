"""Naive DeepSeek V4 attention reference.

Nothing in the production path calls these. They exist for the tests and for the benchmarks in
`notes/ds-v4-kernels/bench/`, where a dense, obviously correct implementation is the standard the
fused kernel is measured against. They take plain tensors, so this module imports nothing from
`attention.py` and the dependency runs one way only.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def eager_attention_with_sinks(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks: torch.Tensor,
    attention_mask: torch.Tensor,
    scaling: float,
    dropout: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    attn_weights = attn_weights + attention_mask

    sink_logits = sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined_logits = torch.cat([attn_weights, sink_logits], dim=-1)
    # Row-max subtraction is not free here: without it the exponentials overflow in bf16.
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)

    scores = F.dropout(probs[..., :-1], p=dropout, training=training).to(value.dtype)
    attn_output = torch.matmul(scores, value)
    return attn_output.transpose(1, 2).contiguous()


def build_sliding_window_mask(*, tok_doc_idx: Tensor, sliding_window: int, dtype: torch.dtype) -> Tensor:
    """Additive `(1, 1, seq_len, seq_len)` mask over query rows and key columns.

    A key is readable when it lies in the query's own document and within the `sliding_window`
    tokens up to and including the query.

    A padded micro-batch folds its padding into the last document, so the padding is masked as a
    continuation of the last document. Causality already keeps it away from every real token, and it
    is loss-masked.
    """
    seq_len = tok_doc_idx.shape[0]
    device = tok_doc_idx.device
    tok_idx = torch.arange(seq_len, device=device)

    distance = tok_idx[:, None] - tok_idx[None, :]
    in_causal_window = (distance >= 0) & (distance < sliding_window)
    same_document = tok_doc_idx[:, None] == tok_doc_idx[None, :]
    readable = in_causal_window & same_document

    mask = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    return mask.masked_fill_(~readable, torch.finfo(dtype).min)[None, None]


def block_bias_from_indices(top_k_indices: Tensor, n_entries: int, dtype: torch.dtype) -> Tensor:
    """Render the indexer's picks as the dense additive `(batch, 1, seq_len, n_entries)` bias.

    `0` on the selected entries, `-inf` everywhere else. The dense and sparse attention paths
    both start from the same index tensor, so they cannot disagree about which entries a query
    reads.
    """
    batch, seq_len, _ = top_k_indices.shape
    # The `-1` sentinels are scattered into one throwaway column that is sliced back off.
    safe_indices = torch.where(top_k_indices >= 0, top_k_indices, torch.full_like(top_k_indices, n_entries))
    block_bias = torch.full((batch, 1, seq_len, n_entries + 1), float("-inf"), dtype=dtype, device=top_k_indices.device)
    block_bias.scatter_(-1, safe_indices.unsqueeze(1), 0.0)
    return block_bias[..., :n_entries]


def dense_mask_from_indices(indices: Tensor, n_positions: int, dtype: torch.dtype) -> Tensor:
    """Render a gather-index tensor as the dense additive `(batch, 1, seq_len, n_positions)` mask.

    `indices` is the `(batch, seq_len, 1, n_slots)` int32 tensor addressing the position axis of a
    `kv_buf` with `n_positions` positions. The mask is `0` on every position at least one of a
    query's slots names and `-inf` everywhere else, with the sentinel position `n_positions - 1`
    always `-inf`: it is `kv_buf`'s zero pad, not a real key.

    This is the fused kernel's oracle. Rendering the index tensor dense and running naive eager
    attention over the whole `kv_buf` exercises the index contract and the attention math together.
    """
    batch, seq_len, _, _ = indices.shape
    mask = torch.full((batch, 1, seq_len, n_positions), float("-inf"), dtype=dtype, device=indices.device)
    mask.scatter_(-1, indices[:, :, 0, :].to(torch.int64).unsqueeze(1), 0.0)
    mask[..., n_positions - 1] = float("-inf")
    return mask
