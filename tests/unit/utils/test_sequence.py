import pytest
import torch

from prime_rl.utils.sequence import (
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


@pytest.mark.parametrize(
    ("seq_lens", "expected_cu_seqlens", "expected_max_seqlen"),
    [
        ([9], [0, 9], 9),
        ([4, 3, 2], [0, 4, 7, 9], 4),
        ([1, 1, 7], [0, 1, 2, 9], 7),
    ],
)
def test_get_cu_seqlens_from_seq_lens(
    seq_lens: list[int], expected_cu_seqlens: list[int], expected_max_seqlen: int
) -> None:
    cu_seqlens, max_seqlen = get_cu_seqlens_from_seq_lens(seq_lens, device="cpu", total_tokens=9)

    assert cu_seqlens.dtype == torch.int32
    assert cu_seqlens.tolist() == expected_cu_seqlens
    assert max_seqlen == expected_max_seqlen


@pytest.mark.parametrize(
    ("seq_lens", "match"), [([4, 3], "sum must equal"), ([9, 0], "must be positive"), ([], "must not be empty")]
)
def test_get_cu_seqlens_from_seq_lens_rejects_invalid(seq_lens: list[int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        get_cu_seqlens_from_seq_lens(seq_lens, device="cpu", total_tokens=9)


@pytest.mark.gpu
def test_get_cu_seqlens_from_seq_lens_does_not_synchronize() -> None:
    torch.cuda.set_sync_debug_mode("error")
    try:
        cu_seqlens, _ = get_cu_seqlens_from_seq_lens([4, 3, 2], device="cuda", total_tokens=9)
    finally:
        torch.cuda.set_sync_debug_mode("default")

    assert cu_seqlens.tolist() == [0, 4, 7, 9]
