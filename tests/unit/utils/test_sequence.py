import pytest
import torch

from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens


def test_get_cu_seqlens_from_seq_lens():
    cu_seqlens, max_seqlen = get_cu_seqlens_from_seq_lens(torch.tensor([4, 3, 2]), total_tokens=9)

    assert cu_seqlens.dtype == torch.int32
    assert cu_seqlens.tolist() == [0, 4, 7, 9]
    assert max_seqlen == 4


def test_get_cu_seqlens_from_seq_lens_rejects_wrong_total():
    with pytest.raises(ValueError, match="sum must equal"):
        get_cu_seqlens_from_seq_lens(torch.tensor([4, 3]), total_tokens=9)
