import pytest
import torch

from prime_rl.utils.sequence import (
    get_cp_local_seq_lens,
    get_cu_seqlens_from_position_ids,
    get_cu_seqlens_from_seq_lens,
)


@pytest.mark.parametrize(
    ("position_ids", "expected_cu_seqlens", "expected_max_seqlen"),
    [
        (torch.arange(8).unsqueeze(0), [0, 8], 8),
        (torch.arange(8, 16).unsqueeze(0), [0, 8], 8),
        (torch.tensor([[0, 1, 2, 3, 0, 1, 2]]), [0, 4, 7], 4),
        (torch.tensor([[5, 6, 7, 0, 1, 2]]), [0, 3, 6], 3),
    ],
)
def test_get_cu_seqlens_from_position_ids_is_local_relative(
    position_ids: torch.Tensor,
    expected_cu_seqlens: list[int],
    expected_max_seqlen: int,
) -> None:
    cu_seqlens, max_seqlen = get_cu_seqlens_from_position_ids(position_ids)

    assert cu_seqlens.dtype == torch.int32
    assert cu_seqlens.tolist() == expected_cu_seqlens
    assert max_seqlen == expected_max_seqlen


def test_get_cu_seqlens_from_seq_lens():
    cu_seqlens, max_seqlen = get_cu_seqlens_from_seq_lens(torch.tensor([4, 3, 2]), total_tokens=9)

    assert cu_seqlens.dtype == torch.int32
    assert cu_seqlens.tolist() == [0, 4, 7, 9]
    assert max_seqlen == 4


def test_get_cu_seqlens_from_seq_lens_rejects_wrong_total():
    with pytest.raises(ValueError, match="sum must equal"):
        get_cu_seqlens_from_seq_lens(torch.tensor([4, 3]), total_tokens=9)


@pytest.mark.parametrize(("cp_rank", "expected"), [(0, [3, 1]), (1, [4])])
def test_get_cp_local_seq_lens(cp_rank: int, expected: list[int]):
    local_seq_lens = get_cp_local_seq_lens(torch.tensor([3, 5]), 8, cp_rank, 2)

    assert local_seq_lens.tolist() == expected
