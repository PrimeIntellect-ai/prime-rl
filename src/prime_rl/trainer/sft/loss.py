"""SFT loss weighting.

Square-averaging (InternVL 2.5; used by Nemotron Nano V2 VL): each supervised
token of document i gets weight 1/sqrt(len_i), where len_i is the document's
count of tokens under the loss mask — NOT its sequence length, so image tokens
and masked prompt tokens do not inflate a document's weight. A document's total
contribution then scales as sqrt(len_i): the compromise between token-sum
(gradients biased toward long responses) and per-document mean (biased toward
short ones).

The weighted per-microbatch losses are raw sums; normalization happens once per
optimizer step against the GLOBAL weight sum (all grad-accum steps and DP ranks),
mirroring the existing global-token-count rescale. A per-microbatch normalizer
would reintroduce the composition bias this scheme removes: a document's
effective weight would depend on what else landed in its microbatch.
"""

import torch
from torch import Tensor


def compute_document_loss_weights(loss_mask: Tensor, seq_lens: Tensor) -> Tensor:
    """Per-token weights for square-averaged loss over a packed row.

    Args:
        loss_mask: Bool tensor of any shape with ``loss_mask.numel() == seq_lens.sum()``;
            True marks supervised tokens. Must be the full (pre-CP-shard) mask.
        seq_lens: Per-document token counts of the packed row(s), in order. The
            trailing pad region is part of the last document and carries a False mask.

    Returns:
        Float32 tensor shaped like ``loss_mask``: 1/sqrt(supervised_count) on each
        document's supervised tokens, 0 elsewhere (including documents with no
        supervised tokens).
    """
    flat_mask = loss_mask.reshape(-1)
    num_docs = seq_lens.numel()
    device = flat_mask.device
    doc_ids = torch.repeat_interleave(torch.arange(num_docs, device=device), seq_lens.to(device))
    if doc_ids.numel() != flat_mask.numel():
        raise ValueError(f"seq_lens sum ({doc_ids.numel()}) does not match loss_mask size ({flat_mask.numel()})")
    supervised_counts = torch.zeros(num_docs, dtype=torch.float32, device=device)
    supervised_counts.scatter_add_(0, doc_ids, flat_mask.float())
    inv_sqrt = torch.where(supervised_counts > 0, supervised_counts.rsqrt(), torch.zeros_like(supervised_counts))
    return (inv_sqrt[doc_ids] * flat_mask.float()).reshape(loss_mask.shape)


__all__ = ["compute_document_loss_weights"]
