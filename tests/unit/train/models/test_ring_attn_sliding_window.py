# ruff: noqa: I001 — `prime_rl._compat` must run before `ring_flash_attn` imports below.
import prime_rl._compat  # noqa: F401

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from ring_flash_attn.llama3_flash_attn_varlen import llama3_flash_attn_prepare_cu_seqlens

from prime_rl.trainer.models.layers.ring_attn import sliding_window_kv

WORLD_SIZE = 4
# 48 tokens, 12 per rank: the second document spans all four shards, so rank 3's 19-token halo
# takes 12 tokens from rank 2 and 7 from rank 1.
CU_SEQLENS = torch.tensor([0, 5, 40, 48], dtype=torch.int32)


def _varlen_window_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, window):
    """Eager stand-in for flash-attn varlen: causal, bottom-right aligned, `window` keys per query."""
    outputs = []
    for i in range(len(cu_seqlens_q) - 1):
        qi = q[cu_seqlens_q[i] : cu_seqlens_q[i + 1]]
        ki = k[cu_seqlens_k[i] : cu_seqlens_k[i + 1]]
        vi = v[cu_seqlens_k[i] : cu_seqlens_k[i + 1]]
        distance = torch.arange(len(qi))[:, None] + len(ki) - len(qi) - torch.arange(len(ki))[None, :]
        scores = torch.einsum("qhd,khd->hqk", qi, ki).masked_fill((distance < 0) | (distance >= window), float("-inf"))
        outputs.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), vi))
    return torch.cat(outputs)


def _run(rank: int, init_file: str, window: int) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD_SIZE)
    torch.manual_seed(0)
    q, k, v = (torch.randn(48, 2, 8, dtype=torch.float64, requires_grad=True) for _ in range(3))
    reference = _varlen_window_attention(q, k, v, CU_SEQLENS, CU_SEQLENS, window)
    reference.square().sum().backward()

    cu_seqlens_q, cu_seqlens_k, _, _, local_k_slice = llama3_flash_attn_prepare_cu_seqlens(
        CU_SEQLENS, True, rank, WORLD_SIZE
    )
    shard = slice(rank * 12, (rank + 1) * 12)
    local_q, local_k, local_v = (t.detach()[shard].clone().requires_grad_() for t in (q, k, v))
    halo_k, halo_v, halo_cu_seqlens_k = sliding_window_kv(
        local_k, local_v, cu_seqlens_k, local_k_slice, window - 1, dist.group.WORLD
    )
    out = _varlen_window_attention(local_q, halo_k, halo_v, cu_seqlens_q, halo_cu_seqlens_k, window)
    out.square().sum().backward()

    torch.testing.assert_close(out, reference[shard])
    for local, full in ((local_q, q), (local_k, k), (local_v, v)):
        torch.testing.assert_close(local.grad, full.grad[shard])
    dist.destroy_process_group()


# window 5 reaches only the previous rank; window 20 reaches two ranks back (shard length 12).
@pytest.mark.parametrize("window", [5, 20])
def test_sliding_window_halo_matches_full_attention(window, tmp_path):
    mp.start_processes(_run, args=(str(tmp_path / "init"), window), nprocs=WORLD_SIZE, start_method="fork")
