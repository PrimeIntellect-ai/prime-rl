"""Paired support evidence carried through vLLM's numeric mask transport."""

import numpy as np
import pybase64
import torch

NEG_INF_BITS = -8388608  # IEEE float32 -inf; plain IDs decode to missing evidence.


def pack_support_logprobs(tensors, logits, cu_num_logits, paired):
    ids = tensors.token_ids.long().clamp(0, logits.shape[-1] - 1)
    rows = cu_num_logits[:-1].long().clamp(max=logits.shape[0] - 1)
    # Capture is non-speculative: each request has one output slot.
    valid = torch.arange(ids.shape[-1], device=logits.device)[None, :] < tensors.counts[:, None]
    values = logits[rows.unsqueeze(-1), ids].float().masked_fill(~valid, -torch.inf)
    # Complete bounded supports let us normalize K gathered logits, not V.
    values = values.log_softmax(-1)
    # The capture kernel emits ascending IDs; stable sorting preserves ID ties.
    # Sort after normalization: distinct logits can round to equal logprobs.
    order = values.argsort(dim=-1, descending=True, stable=True)
    ids = ids.gather(-1, order)
    values = values.gather(-1, order)
    bits = values.contiguous().view(torch.int32) ^ NEG_INF_BITS
    enabled = torch.as_tensor(paired, device=logits.device).unsqueeze(-1)
    packed_ids = torch.where(enabled, ids, tensors.token_ids.long())
    packed = (torch.where(enabled, bits, 0).long() << 32) | packed_ids
    return tensors._replace(token_ids=packed)


def serialize_paired_support(packed, counts):
    """Return aligned mask IDs and existing compact-logprob wire data."""
    bits = np.asarray(packed, dtype=np.uint64)
    ids = (bits & np.uint64(0xFFFFFFFF)).astype("<u4")
    values = ((bits >> np.uint64(32)).astype(np.uint32).view(np.int32) ^ NEG_INF_BITS).view("<f4")
    counts = np.asarray(counts, dtype=np.int32)
    if int(counts.sum()) != len(ids) or (counts <= 0).any() or not np.isfinite(values).all():
        raise ValueError("Incomplete paired sampling-support evidence")
    return ids.astype(np.int32, copy=False), {
        "data": pybase64.b64encode(ids.tobytes() + values.tobytes()).decode("ascii"),
        "format": "u32-f32-le-v1",
        "offsets": [0, *np.cumsum(counts).tolist()],
        "num_top_logprobs": int(counts.max(initial=0)),
    }
