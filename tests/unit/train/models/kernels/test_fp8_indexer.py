import pytest
import torch

from prime_rl.trainer.models.kernels.fp8_indexer import fp8_indexer, fp8_indexer_full

pytestmark = [pytest.mark.gpu]


def _build_ks_ke(seq_lens: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ks_vals: list[int] = []
    ke_vals: list[int] = []
    offset = 0
    for seq_len in seq_lens:
        for pos in range(seq_len):
            ks_vals.append(offset)
            ke_vals.append(offset + pos + 1)
        offset += seq_len
    ks = torch.tensor(ks_vals, dtype=torch.int32, device=device)
    ke = torch.tensor(ke_vals, dtype=torch.int32, device=device)
    return ks, ke


def _assert_indices_in_range(indices: torch.Tensor, ks: torch.Tensor, ke: torch.Tensor, sentinel: int) -> None:
    valid = indices != sentinel
    ks_exp = ks.unsqueeze(1).expand_as(indices)
    ke_exp = ke.unsqueeze(1).expand_as(indices)
    assert torch.all((indices[valid] >= ks_exp[valid]) & (indices[valid] < ke_exp[valid]))


@pytest.mark.parametrize("chunk_size", [64, 127, 256])
def test_fp8_indexer_chunked_matches_full(chunk_size: int):
    torch.manual_seed(42)
    device = torch.device("cuda")
    seq_lens = [96, 128, 80]
    S = sum(seq_lens)
    H = 8
    D = 64
    topk = 96

    q = torch.randn(S, H, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(S, D, device=device, dtype=torch.bfloat16)
    w = torch.randn(S, H, device=device, dtype=torch.bfloat16)
    ks, ke = _build_ks_ke(seq_lens, device=device)

    full_indices = fp8_indexer_full(q, k, w, ks, ke, topk)
    chunked_indices = fp8_indexer(q, k, w, ks, ke, topk, chunk_size=chunk_size)

    assert chunked_indices.dtype == torch.int32
    torch.testing.assert_close(chunked_indices, full_indices)
    _assert_indices_in_range(chunked_indices, ks, ke, S)


def test_fp8_indexer_chunked_respects_padding_and_sentinel():
    torch.manual_seed(123)
    device = torch.device("cuda")
    seq_lens = [40, 56]
    S = sum(seq_lens)
    H = 8
    D = 64
    topk = 128

    q = torch.randn(S, H, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(S, D, device=device, dtype=torch.bfloat16)
    w = torch.randn(S, H, device=device, dtype=torch.bfloat16)
    ks, ke = _build_ks_ke(seq_lens, device=device)

    chunked_indices = fp8_indexer(q, k, w, ks, ke, topk, chunk_size=31)
    full_indices = fp8_indexer_full(q, k, w, ks, ke, topk)

    assert chunked_indices.shape == (S, topk)
    assert torch.any(chunked_indices == S)
    torch.testing.assert_close(chunked_indices, full_indices)
    _assert_indices_in_range(chunked_indices, ks, ke, S)
