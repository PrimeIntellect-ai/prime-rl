from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
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
        # Keeping the prefetch detached avoids materializing a table-sized GPU
        # gradient before copying it back to the CPU-owned parameter.
        weight_gradient = torch.zeros(ctx.weight_shape, dtype=ctx.weight_dtype, device="cpu")
        weight_gradient.index_add_(
            0,
            indices.to("cpu"),
            grad_output.reshape(-1, grad_output.shape[-1]).to("cpu"),
        )
        return weight_gradient, None, None


class OwnerShardedEmbedding(nn.Module):
    """An embedding table whose rows have one owner across the training ranks."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))

        self.device_mesh: DeviceMesh | None = None
        self.cpu_device_mesh: DeviceMesh | None = None
        self.rows_per_rank = num_embeddings
        self.gradient_divide_factor = 1
        self.cpu_offload = False
        self.offload_state: EmbeddingOffloadState | None = None

    def parallelize(self, device_mesh: DeviceMesh, *, cpu_offload: bool) -> None:
        world_size = device_mesh.size()
        if self.num_embeddings % world_size:
            raise ValueError(
                f"Embedding rows ({self.num_embeddings}) must be divisible by the owner mesh ({world_size})"
            )

        self.device_mesh = device_mesh
        self.rows_per_rank = self.num_embeddings // world_size
        self.gradient_divide_factor = world_size
        self.cpu_offload = cpu_offload

        if self.weight.device.type == "meta":
            local_weight = self.weight.new_empty((self.rows_per_rank, self.embedding_dim))
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

        requires_grad = self.weight.requires_grad
        self.weight = nn.Parameter(weight, requires_grad=requires_grad)

        if cpu_offload:
            self.cpu_device_mesh = DeviceMesh(
                "cpu",
                device_mesh.mesh.cpu(),
                mesh_dim_names=device_mesh.mesh_dim_names,
            )

    def materialize(self) -> None:
        local_shape = (self.rows_per_rank, self.embedding_dim)
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
        gpu_weight = self.consume_prefetched_weight() if self.cpu_offload else None
        local_weight = self.weight.to_local()
        flat_indices = indices.reshape(-1)

        if self.device_mesh.size() == 1:
            embeddings = (
                OffloadedEmbeddingLookup.apply(local_weight, gpu_weight, flat_indices)
                if self.cpu_offload
                else F.embedding(flat_indices, local_weight)
            )
        else:
            owners = torch.div(flat_indices, self.rows_per_rank, rounding_mode="floor")
            order = torch.argsort(owners, stable=True)
            sorted_owners = owners[order]
            local_indices = flat_indices[order] - sorted_owners * self.rows_per_rank

            input_splits = torch.bincount(owners, minlength=self.device_mesh.size())
            output_splits = all_to_all_single_equal(input_splits, self.device_mesh.get_group())
            local_indices = all_to_all_single(
                local_indices,
                output_splits,
                input_splits,
                self.device_mesh.get_group(),
            )
            local_embeddings = (
                OffloadedEmbeddingLookup.apply(local_weight, gpu_weight, local_indices)
                if self.cpu_offload
                else F.embedding(local_indices, local_weight)
            )
            sorted_embeddings = all_to_all_single(
                local_embeddings,
                input_splits,
                output_splits,
                self.device_mesh.get_group(),
            )
            embeddings = sorted_embeddings[torch.argsort(order)]

        return embeddings.reshape(*indices.shape, self.embedding_dim)


__all__ = ["OwnerShardedEmbedding"]
