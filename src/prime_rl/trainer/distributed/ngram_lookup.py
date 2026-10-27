from typing import Protocol

import torch
import torch.distributed._functional_collectives as funcol
import torch.nn.functional as F
from jaxtyping import Int
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor


def count_ids_per_rank(sorted_ids: Int[Tensor, "ids"], rows_per_rank: int, num_ranks: int) -> Int[Tensor, "ranks"]:
    """How many of `sorted_ids` fall in each rank's rows [rank * rows_per_rank, (rank + 1) * rows_per_rank)."""
    first_rows = torch.arange(num_ranks + 1, device=sorted_ids.device) * rows_per_rank
    return torch.searchsorted(sorted_ids, first_rows).diff()


class NGramLookup(Protocol):
    def __call__(self, weight: torch.Tensor, ids: torch.Tensor) -> torch.Tensor: ...


class LocalLookup:
    def __call__(self, weight: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids, weight)


class RowShardedLookup:
    """Look up a table sharded `Shard(0)` over a 1-D mesh by sending each id to the rank that owns its row."""

    def __init__(self, mesh: DeviceMesh) -> None:
        self.mesh = mesh

    def __call__(self, weight: DTensor, ids: torch.Tensor) -> torch.Tensor:
        num_ranks = self.mesh.size()
        rows_per_rank = (weight.shape[0] + num_ranks - 1) // num_ranks
        first_local_row = self.mesh.get_local_rank() * rows_per_rank
        local_weight = weight.to_local(grad_placements=weight.placements)
        sorted_ids, order = ids.flatten().sort(stable=True)
        send_counts = count_ids_per_rank(sorted_ids, rows_per_rank, num_ranks)
        recv_counts = funcol.all_to_all_single(send_counts, None, None, self.mesh)
        send_splits, recv_splits = torch.stack((send_counts, recv_counts)).cpu().tolist()
        recv_ids = funcol.all_to_all_single(sorted_ids, recv_splits, send_splits, self.mesh)
        rows = F.embedding(recv_ids - first_local_row, local_weight)
        sorted_rows = funcol.all_to_all_single_autograd(rows, send_splits, recv_splits, self.mesh)

        head_dim = local_weight.shape[-1]
        output = sorted_rows.new_empty((*ids.shape, head_dim))
        output.view(-1, head_dim)[order] = sorted_rows
        return output


__all__ = ["LocalLookup", "NGramLookup", "RowShardedLookup", "count_ids_per_rank"]
