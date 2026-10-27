import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.parallel import parallelize_module

from prime_rl.trainer.distributed.ngram_lookup import RowShardedLookup, count_ids_per_rank
from prime_rl.trainer.distributed.ngram_parallel import NGramRowParallel
from prime_rl.trainer.models.layers.ngram import NGramTable

HEAD_DIM = 4
MAX_NGRAM_SIZE = 3
NUM_HEADS = 4

IDS_BY_RANK = {
    (9, (4,)): [
        [[0, 2, 3, 5], [6, 8, 2, 3], [2, 2, 8, 8]],
        [[0, 1, 2, 2], [1, 1, 0, 2]],
        [[6, 8, 8, 7], [8, 6, 6, 6]],
        [[5, 6, 2, 3]],
    ],
    (10, (4,)): [
        [[0, 2, 3, 5], [6, 8, 9, 9]],
        [[9, 9, 9, 9], [9, 9, 9, 9]],
        [[1, 1, 2, 0], [2, 0, 1, 1], [0, 0, 0, 0]],
        [[5, 6, 8, 9], [3, 3, 2, 2]],
    ],
    (9, (2, 2)): [
        [[0, 4, 5, 8], [4, 4, 5, 5]],
        [[1, 2, 3, 4], [0, 0, 4, 4]],
        [[5, 6, 7, 8], [8, 8, 5, 5]],
        [[5, 5, 8, 7], [6, 6, 6, 6]],
    ],
}


@pytest.fixture(scope="module")
def process_group():
    if int(os.environ.get("WORLD_SIZE", 1)) != 4:
        pytest.skip("run with torchrun --nproc-per-node=4")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    yield dist.group.WORLD
    dist.destroy_process_group()


def _head_mesh(mesh_shape: tuple[int, ...]):
    mesh_dim_names = ("dp_replicate", "head")[-len(mesh_shape) :]
    return init_device_mesh("cuda", mesh_shape, mesh_dim_names=mesh_dim_names)["head"]


def _integer_upstream_grad(rank: int, shape: torch.Size) -> torch.Tensor:
    values = torch.arange(shape.numel(), device="cuda") * (rank + 3) % 7 - 3
    return values.to(torch.float32).view(shape)


@pytest.mark.gpu
@pytest.mark.parametrize("num_rows, mesh_shape", list(IDS_BY_RANK), ids=["9rows-head4", "10rows-head4", "9rows-2x2"])
def test_row_sharded_lookup_matches_full_table(process_group, num_rows, mesh_shape):
    rank = dist.get_rank()
    head_mesh = _head_mesh(mesh_shape)
    rows_per_rank = (num_rows + head_mesh.size() - 1) // head_mesh.size()
    first_local_row = head_mesh.get_local_rank() * rows_per_rank
    local_row_count = max(0, min(rows_per_rank, num_rows - first_local_row))

    full_weight = torch.arange(num_rows * HEAD_DIM, dtype=torch.float32, device="cuda").view(num_rows, HEAD_DIM)
    table = NGramTable(num_rows, HEAD_DIM, MAX_NGRAM_SIZE, NUM_HEADS).cuda()
    with torch.no_grad():
        table.weight.copy_(full_weight)
    parallelize_module(table, device_mesh=head_mesh, parallelize_plan=NGramRowParallel())
    table.set_lookup(RowShardedLookup(head_mesh))
    assert table.weight.to_local().shape == (local_row_count, HEAD_DIM)

    ids_by_rank = [torch.tensor(ids, device="cuda") for ids in IDS_BY_RANK[num_rows, mesh_shape]]
    upstream_grads = [
        _integer_upstream_grad(r, torch.Size((*ids.shape, HEAD_DIM))) for r, ids in enumerate(ids_by_rank)
    ]

    output = table(ids_by_rank[rank])
    assert output._base is None
    assert torch.equal(output, F.embedding(ids_by_rank[rank], full_weight))
    output.backward(upstream_grads[rank])

    reference_weight = full_weight.clone().requires_grad_()
    for head_group_rank in head_mesh.mesh.tolist():
        reference = F.embedding(ids_by_rank[head_group_rank], reference_weight)
        reference.backward(upstream_grads[head_group_rank])

    grad = table.weight.grad
    assert isinstance(grad, DTensor)
    assert grad.placements == table.weight.placements == (Shard(0),)
    reference_local_grad = reference_weight.grad[first_local_row : first_local_row + local_row_count]
    assert torch.equal(grad.to_local(), reference_local_grad)


@pytest.mark.gpu
def test_row_parallel_shards_meta_weight_and_keeps_requires_grad(process_group):
    head_mesh = _head_mesh((4,))
    with torch.device("meta"):
        table = NGramTable(10, HEAD_DIM, MAX_NGRAM_SIZE, NUM_HEADS)
    table.weight.requires_grad_(False)

    parallelize_module(table, device_mesh=head_mesh, parallelize_plan=NGramRowParallel())

    assert isinstance(table.weight, DTensor)
    assert table.weight.is_meta
    assert table.weight.shape == (10, HEAD_DIM)
    assert table.weight.to_local().shape == ([3, 3, 3, 1][head_mesh.get_local_rank()], HEAD_DIM)
    assert not table.weight.requires_grad
    assert not isinstance(table.ngram_heads_vocab_sizes, DTensor)


@pytest.mark.parametrize(
    "num_rows, num_ranks, ids",
    [
        pytest.param(9, 4, [0, 2, 2, 3, 5, 6, 8, 8], id="last-rank-owns-no-rows"),
        pytest.param(10, 4, [0, 2, 3, 5, 6, 8, 9, 9], id="last-rank-owns-one-row"),
        pytest.param(10, 4, [9, 9, 9], id="one-owner"),
        pytest.param(5, 8, [0, 1, 4, 4], id="more-ranks-than-rows"),
        pytest.param(9, 4, [], id="no-ids"),
    ],
)
def test_count_ids_per_rank_matches_bincount(num_rows, num_ranks, ids):
    rows_per_rank = (num_rows + num_ranks - 1) // num_ranks
    sorted_ids = torch.tensor(sorted(ids), dtype=torch.long)

    counts = count_ids_per_rank(sorted_ids, rows_per_rank, num_ranks)

    assert torch.equal(counts, torch.bincount(sorted_ids // rows_per_rank, minlength=num_ranks))
