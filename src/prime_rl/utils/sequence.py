from __future__ import annotations

import torch


def get_cu_seqlens_from_position_ids(position_ids: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Build local-relative cumulative sequence lengths from packed position ids.

    ``position_ids`` may be a full packed sequence or a context-parallel local shard.
    In the shard case, the first local token can be a continuation with a non-zero
    position id, so cumulative offsets must be anchored to the local tensor rather
    than to the absolute position id value.
    """
    flat_position_ids = position_ids.view(-1)
    total_tokens = flat_position_ids.numel()
    assert total_tokens > 0, "Cannot build cu_seqlens for an empty position_ids tensor"

    zero_starts = (flat_position_ids == 0).nonzero(as_tuple=True)[0]
    starts = torch.cat(
        [torch.zeros(1, dtype=zero_starts.dtype, device=zero_starts.device), zero_starts]
    ).unique_consecutive()

    ends = torch.cat(
        [
            starts[1:],
            torch.tensor([total_tokens], dtype=starts.dtype, device=starts.device),
        ]
    )
    seqlens = ends - starts

    cu_seqlens = torch.empty(seqlens.numel() + 1, dtype=torch.int32, device=position_ids.device)
    cu_seqlens[0] = 0
    cu_seqlens[1:] = seqlens.cumsum(dim=0, dtype=torch.int32)

    return cu_seqlens, seqlens.max().item()
