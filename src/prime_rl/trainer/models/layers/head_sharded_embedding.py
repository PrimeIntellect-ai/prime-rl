from dataclasses import dataclass
from itertools import accumulate
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from prime_rl.trainer.distributed.collectives import all_to_all_single, all_to_all_single_equal


@dataclass
class EmbeddingOffloadState:
    stream: torch.cuda.Stream
    event: torch.cuda.Event
    prefetched_weight: torch.Tensor | None = None


class OffloadedEmbeddingLookup(torch.autograd.Function):
    """Use a prefetched GPU weight while accumulating its gradient on CPU."""

    @staticmethod
    def forward(
        ctx,
        cpu_weight: torch.Tensor,
        gpu_weight: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(indices)
        ctx.weight_shape = cpu_weight.shape
        ctx.weight_dtype = cpu_weight.dtype
        return F.embedding(indices, gpu_weight)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        (indices,) = ctx.saved_tensors
        weight_gradient = torch.zeros(ctx.weight_shape, dtype=ctx.weight_dtype, device="cpu")
        weight_gradient.index_add_(
            0,
            indices.to("cpu"),
            grad_output.reshape(-1, grad_output.shape[-1]).to("cpu"),
        )
        return weight_gradient, None, None


class HeadShardedEmbedding(nn.Module):
    """Flat embedding table sharded by the vocabulary ranges of logical heads."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        head_sizes: Sequence[int],
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.head_sizes = tuple(head_sizes)
        self.head_offsets = (0, *accumulate(self.head_sizes))
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))

        self.device_mesh: DeviceMesh | None = None
        self.communication_group: ProcessGroup | None = None
        self.cpu_device_mesh: DeviceMesh | None = None
        self.local_row_count = num_embeddings
        self.local_row_offset = 0
        self.gradient_divide_factor = 1
        self.cpu_offload = False
        self.offload_state: EmbeddingOffloadState | None = None
        self.register_buffer("dispatch_heads", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("dispatch_head_mask", torch.empty(0, dtype=torch.bool), persistent=False)

    def parallelize(
        self,
        device_mesh: DeviceMesh,
        communication_group: ProcessGroup,
        *,
        cpu_offload: bool,
    ) -> None:
        world_size = device_mesh.size()
        local_rank = device_mesh.get_local_rank()
        self.device_mesh = device_mesh
        self.communication_group = communication_group
        self.local_row_count, self.local_row_offset = Shard.local_shard_size_and_offset(
            self.num_embeddings,
            world_size,
            local_rank,
        )
        self.gradient_divide_factor = world_size
        self.cpu_offload = cpu_offload

        overlapping_heads: list[list[int]] = []
        for rank in range(world_size):
            row_count, row_offset = Shard.local_shard_size_and_offset(
                self.num_embeddings,
                world_size,
                rank,
            )
            row_end = row_offset + row_count
            overlapping_heads.append(
                [
                    head
                    for head, (head_start, head_end) in enumerate(zip(self.head_offsets, self.head_offsets[1:]))
                    if head_start < row_end and head_end > row_offset
                ]
            )

        heads_per_rank = max(map(len, overlapping_heads))
        padded_heads = [heads + [-1] * (heads_per_rank - len(heads)) for heads in overlapping_heads]
        self.dispatch_heads = torch.tensor(padded_heads, dtype=torch.long, device="cuda")
        self.dispatch_head_mask = self.dispatch_heads >= 0

        if self.weight.device.type == "meta":
            local_weight = self.weight.new_empty((self.local_row_count, self.embedding_dim))
            weight = DTensor.from_local(
                local_weight,
                device_mesh,
                [Shard(0)],
                shape=self.weight.shape,
                stride=self.weight.stride(),
                run_check=False,
            )
        else:
            weight = distribute_tensor(self.weight.detach(), device_mesh, [Shard(0)])

        self.weight = nn.Parameter(weight, requires_grad=self.weight.requires_grad)

        if cpu_offload:
            self.cpu_device_mesh = DeviceMesh(
                "cpu",
                device_mesh.mesh.cpu(),
                mesh_dim_names=device_mesh.mesh_dim_names,
            )

    def materialize(self) -> None:
        local_shape = (self.local_row_count, self.embedding_dim)
        if self.cpu_offload:
            local_weight = torch.empty(
                local_shape,
                dtype=self.weight.dtype,
                device="cpu",
                pin_memory=True,
            )
            device_mesh = self.cpu_device_mesh
            self.offload_state = EmbeddingOffloadState(torch.cuda.Stream(), torch.cuda.Event())
        else:
            local_weight = torch.empty(local_shape, dtype=self.weight.dtype, device="cuda")
            device_mesh = self.device_mesh

        weight = DTensor.from_local(
            local_weight,
            device_mesh,
            [Shard(0)],
            shape=self.weight.shape,
            stride=self.weight.stride(),
            run_check=False,
        )
        self.weight = nn.Parameter(weight, requires_grad=self.weight.requires_grad)
        self.weight.register_hook(lambda gradient: gradient / self.gradient_divide_factor)

    @torch.compiler.disable
    def prefetch(self) -> None:
        if not self.cpu_offload or self.offload_state.prefetched_weight is not None:
            return

        with torch.cuda.stream(self.offload_state.stream):
            self.offload_state.prefetched_weight = self.weight.to_local().detach().to("cuda", non_blocking=True)
            self.offload_state.event.record()

    @torch.compiler.disable
    def consume_prefetched_weight(self) -> torch.Tensor:
        if self.offload_state.prefetched_weight is None:
            self.prefetch()
        torch.cuda.current_stream().wait_event(self.offload_state.event)
        weight = self.offload_state.prefetched_weight
        self.offload_state.prefetched_weight = None
        weight.record_stream(torch.cuda.current_stream())
        return weight

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        local_weight = self.weight.to_local()
        lookup_weight = self.consume_prefetched_weight() if self.cpu_offload else local_weight
        flat_indices = indices.reshape(-1, len(self.head_sizes))

        world_size = self.device_mesh.size()
        if world_size == 1:
            embeddings = (
                OffloadedEmbeddingLookup.apply(local_weight, lookup_weight, flat_indices)
                if self.cpu_offload
                else F.embedding(flat_indices, lookup_weight)
            )
            return embeddings.reshape(*indices.shape, self.embedding_dim)

        head_indices = self.dispatch_heads.clamp_min(0)
        dispatched_indices = flat_indices[:, head_indices].movedim(1, 0).contiguous()
        local_token_counts = flat_indices.new_full((world_size,), flat_indices.shape[0])
        source_token_counts = all_to_all_single_equal(local_token_counts, self.communication_group)
        dispatched_indices = all_to_all_single(
            dispatched_indices.flatten(0, 1),
            source_token_counts,
            local_token_counts,
            self.communication_group,
        )

        valid_heads = self.dispatch_head_mask[self.device_mesh.get_local_rank()]
        owned = (
            valid_heads.view(1, -1)
            & (dispatched_indices >= self.local_row_offset)
            & (dispatched_indices < self.local_row_offset + self.local_row_count)
        )
        local_indices = torch.where(owned, dispatched_indices - self.local_row_offset, 0)
        local_embeddings = (
            OffloadedEmbeddingLookup.apply(local_weight, lookup_weight, local_indices)
            if self.cpu_offload
            else F.embedding(local_indices, lookup_weight)
        )
        local_embeddings = torch.where(owned.unsqueeze(-1), local_embeddings, 0)

        returned_embeddings = all_to_all_single(
            local_embeddings,
            local_token_counts,
            source_token_counts,
            self.communication_group,
        )
        returned_embeddings = returned_embeddings.reshape(
            world_size,
            flat_indices.shape[0],
            head_indices.shape[1],
            self.embedding_dim,
        )
        returned_embeddings = returned_embeddings.movedim(0, 1).flatten(1, 2)
        returned_heads = head_indices.flatten()
        output = returned_embeddings.new_zeros((flat_indices.shape[0], len(self.head_sizes), self.embedding_dim))
        output = output.scatter_add(
            1,
            returned_heads.view(1, -1, 1).expand_as(returned_embeddings),
            returned_embeddings,
        )
        return output.reshape(*indices.shape, self.embedding_dim)


__all__ = ["HeadShardedEmbedding"]
