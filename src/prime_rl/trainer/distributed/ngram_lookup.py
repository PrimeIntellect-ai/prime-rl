from dataclasses import dataclass
from typing import Protocol

import torch
import torch.distributed._functional_collectives as funcol
import torch.nn.functional as F
from jaxtyping import Float, Int
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


@dataclass(frozen=True)
class RowDispatchState:
    original_positions: Int[Tensor, "ids"]
    send_splits: list[int]
    recv_splits: list[int]
    ids_shape: torch.Size


class ExpandUniqueRows(torch.autograd.Function):
    """`unique_rows[inverse]`, summing the gradients of repeated ids in fp32."""

    @staticmethod
    def forward(ctx, unique_rows: Tensor, inverse: Tensor) -> Tensor:
        ctx.save_for_backward(inverse)
        ctx.num_unique = unique_rows.shape[0]
        return unique_rows[inverse]

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        (inverse,) = ctx.saved_tensors
        head_dim = grad_output.shape[-1]
        grad_unique = grad_output.new_zeros(ctx.num_unique, head_dim, dtype=torch.float32)
        grad_unique.index_add_(0, inverse.flatten(), grad_output.reshape(-1, head_dim).float())
        return grad_unique.to(grad_output.dtype), None


class RowShardedLookup:
    """Look up a table sharded `Shard(0)` over a 1-D mesh by sending each id to the rank that owns its row.

    With `deduplicate`, repeated ids are sent and fetched once.
    """

    def __init__(self, mesh: DeviceMesh, deduplicate: bool = False) -> None:
        self.mesh = mesh
        self.deduplicate = deduplicate

    def __call__(self, weight: DTensor, ids: Int[Tensor, "..."]) -> Float[Tensor, "... head_dim"]:
        if not self.deduplicate:
            return self.lookup(weight, ids)
        unique_ids, inverse = torch.unique(ids, return_inverse=True)
        return ExpandUniqueRows.apply(self.lookup(weight, unique_ids), inverse)

    def lookup(self, weight: DTensor, ids: Int[Tensor, "..."]) -> Float[Tensor, "... head_dim"]:
        rows_per_rank = (weight.shape[0] + self.mesh.size() - 1) // self.mesh.size()
        owned_ids, state = self.dispatch(ids, rows_per_rank)
        owned_rows = self.lookup_owned_rows(weight, owned_ids, rows_per_rank)
        return self.combine(owned_rows, state)

    def dispatch(
        self, ids: Int[Tensor, "..."], rows_per_rank: int
    ) -> tuple[Int[Tensor, "owned_ids"], RowDispatchState]:
        sorted_ids, original_positions = ids.flatten().sort(stable=True)
        send_counts = count_ids_per_rank(sorted_ids, rows_per_rank, self.mesh.size())
        recv_counts = funcol.all_to_all_single(send_counts, None, None, self.mesh)
        send_splits, recv_splits = torch.stack((send_counts, recv_counts)).cpu().tolist()
        owned_ids = funcol.all_to_all_single(sorted_ids, recv_splits, send_splits, self.mesh)
        return owned_ids, RowDispatchState(original_positions, send_splits, recv_splits, ids.shape)

    def lookup_owned_rows(
        self, weight: DTensor, owned_ids: Int[Tensor, "owned_ids"], rows_per_rank: int
    ) -> Float[Tensor, "owned_ids head_dim"]:
        local_weight = weight.to_local(grad_placements=weight.placements)
        first_owned_row = self.mesh.get_local_rank() * rows_per_rank
        return F.embedding(owned_ids - first_owned_row, local_weight)

    def combine(
        self, owned_rows: Float[Tensor, "owned_ids head_dim"], state: RowDispatchState
    ) -> Float[Tensor, "... head_dim"]:
        sorted_rows = funcol.all_to_all_single_autograd(owned_rows, state.send_splits, state.recv_splits, self.mesh)
        head_dim = sorted_rows.shape[-1]
        output = sorted_rows.new_empty((*state.ids_shape, head_dim))
        output.view(-1, head_dim)[state.original_positions] = sorted_rows
        return output


__all__ = ["LocalLookup", "NGramLookup", "RowShardedLookup", "count_ids_per_rank"]
