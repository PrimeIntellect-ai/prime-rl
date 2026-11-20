from __future__ import annotations

import torch


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
