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
    bits = values.contiguous().view(torch.int32) ^ NEG_INF_BITS
    enabled = torch.as_tensor(paired, device=logits.device).unsqueeze(-1)
    packed_ids = torch.where(enabled, ids, tensors.token_ids.long())
    packed = (torch.where(enabled, bits, 0).long() << 32) | packed_ids
    return tensors._replace(token_ids=packed)


def pack_sampler_rows(tensors, logits, cu_num_logits, num_sampled, sampled, num_rows, draws):
    """Per selected request: the sampled token, then ``num_rows`` ids from the processed distribution q (i.i.d.
    draws with replacement, or the top ids), each packed with its full-vocabulary processed logprob. The rows
    copy vLLM's ``[sampled, top...]`` logprob layout. Draws use the global generator, so they are independent
    of the sampled token."""
    from vllm.v1.worker.gpu.sample.output import SamplingMaskTensors

    selected = np.flatnonzero(num_rows)
    width = int(num_rows.max())
    x = logits[cu_num_logits[:-1].long()[selected]].float()
    if draws[selected].all():
        alternatives = torch.multinomial(x.softmax(-1), width, replacement=True)
    elif not draws[selected].any():
        alternatives = x.topk(width).indices
    else:
        use_draws = torch.as_tensor(draws[selected], device=x.device).unsqueeze(-1)
        alternatives = torch.where(
            use_draws, torch.multinomial(x.softmax(-1), width, replacement=True), x.topk(width).indices
        )
    ids = torch.cat([sampled[selected].view(-1, 1).long(), alternatives], dim=1)
    values = (x.gather(1, ids) - x.logsumexp(-1, keepdim=True)).clamp_min(-9999.0)
    packed = ((values.contiguous().view(torch.int32) ^ NEG_INF_BITS).long() << 32) | ids

    num_reqs = num_sampled.shape[0]
    if tensors is None:
        token_ids = packed.new_zeros((num_reqs, width + 1))
        counts = torch.zeros(num_reqs, dtype=torch.int32, device=x.device)
        tensors = SamplingMaskTensors(
            token_ids, token_ids.new_zeros((num_reqs, 1), dtype=torch.uint8), counts, x.shape[1]
        )
    token_ids = torch.nn.functional.pad(tensors.token_ids.long(), (0, max(0, width + 1 - tensors.token_ids.shape[1])))
    index = torch.as_tensor(selected, device=x.device)
    token_ids[index, : width + 1] = packed
    row_counts = torch.as_tensor(num_rows[selected] + 1, device=x.device, dtype=torch.int32)
    counts = tensors.counts.index_put((index,), torch.where(num_sampled[index] > 0, row_counts, 0))
    return tensors._replace(token_ids=token_ids, counts=counts)


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
