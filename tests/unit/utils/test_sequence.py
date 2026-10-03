import pytest
import torch

from prime_rl.utils.sequence import (
    CPPartition,
    get_cu_seqlens_from_position_ids,
    get_cu_seqlens_from_seq_lens,
)


@pytest.mark.parametrize("degree", [1, 2, 4])
@pytest.mark.parametrize("total", [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 32769])
def test_cp_partition_roundtrip(total, degree):
    partition = CPPartition(total, degree)
    assert len(partition.lengths) == degree
    assert partition.offsets[0] == 0
    assert partition.offsets[-1] == total
    assert sum(partition.lengths) == total
    assert max(partition.lengths) - min(partition.lengths) <= 1
    for dim, shape in [(1, (1, total, 2, 3)), (2, (3, 1, total))]:
        tensor = torch.arange(torch.tensor(shape).prod()).reshape(shape)
        shards = [partition.shard(tensor, rank, dim) for rank in range(degree)]
        assert [shard.shape[dim] for shard in shards] == list(partition.lengths)
        torch.testing.assert_close(torch.cat(shards, dim=dim), tensor)


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
