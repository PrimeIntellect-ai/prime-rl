import pytest
import torch

from prime_rl.utils.sequence import get_cu_seqlens_from_position_ids
from prime_rl.utils.sequence_packing import (
    build_cu_seqlens,
    infer_cu_seqlens_from_position_ids,
    infer_sequence_lengths_from_position_ids,
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


def test_build_cu_seqlens_from_explicit_lengths() -> None:
    cu_seqlens, max_seqlen = build_cu_seqlens([7, 5, 9])

    assert cu_seqlens.dtype == torch.int32
    assert cu_seqlens.tolist() == [0, 7, 12, 21]
    assert max_seqlen == 9


def test_sequence_packing_inference_does_not_split_trailing_zero_padding() -> None:
    position_ids = torch.tensor([[0, 1, 2, 0, 0, 0]])

    assert infer_sequence_lengths_from_position_ids(position_ids) == [6]

    cu_seqlens, max_seqlen = infer_cu_seqlens_from_position_ids(position_ids)
    assert cu_seqlens.tolist() == [0, 6]
    assert max_seqlen == 6


def test_sequence_packing_inference_splits_position_resets_with_next_one() -> None:
    position_ids = torch.tensor([[0, 1, 2, 0, 1]])

    assert infer_sequence_lengths_from_position_ids(position_ids) == [3, 2]

    cu_seqlens, max_seqlen = infer_cu_seqlens_from_position_ids(position_ids)
    assert cu_seqlens.tolist() == [0, 3, 5]
    assert max_seqlen == 3
