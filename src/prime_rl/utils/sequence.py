from __future__ import annotations

import torch


def get_cu_seqlens_from_position_ids(position_ids: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Build local-relative cumulative sequence lengths from packed position ids.

    ``position_ids`` may be a full packed sequence or a context-parallel local shard.
    In the shard case, the first local token can be a continuation with a non-zero
    position id, so cumulative offsets must be anchored to the local tensor rather
    than to the absolute position id value.
    """
    flat_position_ids = position_ids.reshape(-1)
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


def get_cu_seqlens_from_seq_lens(seq_lens: torch.Tensor, total_tokens: int | None = None) -> tuple[torch.Tensor, int]:
    if seq_lens.ndim != 1:
        raise ValueError(f"seq_lens must be 1D, got shape={tuple(seq_lens.shape)}")
    if seq_lens.numel() == 0:
        raise ValueError("seq_lens must not be empty")
    if bool((seq_lens <= 0).any().item()):
        raise ValueError(f"seq_lens must be positive, got {seq_lens.tolist()}")
    if total_tokens is not None and int(seq_lens.sum().item()) != total_tokens:
        raise ValueError(f"seq_lens sum must equal sequence length: {seq_lens.tolist()} vs {total_tokens}")

    seq_lens = seq_lens.to(dtype=torch.int32)
    cu_seqlens = torch.empty(seq_lens.numel() + 1, dtype=torch.int32, device=seq_lens.device)
    cu_seqlens[0] = 0
    cu_seqlens[1:] = seq_lens.cumsum(dim=0, dtype=torch.int32)
    return cu_seqlens, int(seq_lens.max().item())


def get_cp_local_seq_lens(
    seq_lens: torch.Tensor,
    total_tokens: int,
    cp_rank: int,
    cp_world_size: int,
) -> torch.Tensor:
    """Intersect global sequence boundaries with one contiguous CP shard."""
    shard_size = total_tokens // cp_world_size
    shard_start = seq_lens.new_tensor(cp_rank * shard_size)
    shard_end = shard_start + shard_size
    seq_ends = seq_lens.cumsum(dim=0)
    seq_starts = torch.cat([seq_lens.new_zeros(1), seq_ends[:-1]])
    overlaps = torch.minimum(seq_ends, shard_end) - torch.maximum(seq_starts, shard_start)
    return overlaps[overlaps > 0]
