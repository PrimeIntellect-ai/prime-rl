"""FlashMLA's sparse prefill forward called directly: a reference, not a backend, showing the forward target."""

import torch
import torch.nn.functional as F

LABEL = "flashmla_fwd_ref (reference, not a backend)"
FORWARD_ONLY = True
EXECUTED_SLOT_TILE = None
SLOT_MULTIPLE = 128


def unavailable_reason() -> str | None:
    try:
        import flash_mla  # noqa: F401
    except ImportError as error:
        return f"flash_mla does not import: {error}"
    return None


def fwd(q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, sinks: torch.Tensor, scale: float):
    from flash_mla import flash_mla_sparse_fwd

    batch, n_queries, heads, dim = q.shape
    n_positions = kv.shape[1]
    n_slots = indices.shape[-1]
    padded_slots = -(-n_slots // SLOT_MULTIPLE) * SLOT_MULTIPLE
    padded = F.pad(indices, (0, padded_slots - n_slots), value=-1)
    batch_offset = torch.arange(batch, device=indices.device, dtype=indices.dtype).view(batch, 1, 1, 1) * n_positions
    flat_indices = torch.where(padded >= 0, padded + batch_offset, -1).view(batch * n_queries, 1, padded_slots)
    slot_numbers = torch.arange(1, padded_slots + 1, device=indices.device, dtype=torch.int32)
    topk_length = torch.where(padded >= 0, slot_numbers, 0).amax(-1).view(batch * n_queries).to(torch.int32)
    out, _max_logits, _lse = flash_mla_sparse_fwd(
        q.view(batch * n_queries, heads, dim),
        kv.view(batch * n_positions, 1, dim),
        flat_indices,
        scale,
        dim,
        attn_sink=sinks,
        topk_length=topk_length,
    )
    return out.view(batch, n_queries, heads, dim), None


def install_compile_counter():
    return lambda: {"compiles": 0}
