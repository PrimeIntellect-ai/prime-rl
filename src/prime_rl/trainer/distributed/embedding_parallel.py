import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.parallel import ParallelStyle, RowwiseParallel


class EmbeddingParallel(RowwiseParallel):
    """Vocabulary-parallel embedding with uneven token batches on each rank."""

    def __init__(self) -> None:
        super().__init__(input_layouts=Shard(0), output_layouts=Shard(0))

    def _prepare_input_fn(self, input_layouts, desired_input_layouts, module, inputs, device_mesh):
        indices = inputs[0]
        self.local_tokens = indices.shape[0]
        max_tokens = indices.new_tensor(self.local_tokens)
        dist.all_reduce(max_tokens, op=dist.ReduceOp.MAX, group=device_mesh.get_group())
        # TODO: Avoid the D2H sync in max_tokens.item(); F.pad needs the dynamic
        # cross-rank token count on the host to size the padded tensor.
        indices = F.pad(indices, (0, 0, 0, max_tokens.item() - self.local_tokens))
        return super()._prepare_input_fn(input_layouts, desired_input_layouts, module, (indices,), device_mesh)

    def _prepare_output_fn(self, output_layouts, use_local_output, module, outputs, device_mesh):
        output = super()._prepare_output_fn(output_layouts, use_local_output, module, outputs, device_mesh)
        return output[: self.local_tokens]


def _all_to_all(tensor: torch.Tensor, output_splits: list[int], input_splits: list[int], group) -> torch.Tensor:
    out = tensor.new_empty((sum(output_splits), *tensor.shape[1:]))
    dist.all_to_all_single(out, tensor.contiguous(), output_splits, input_splits, group=group)
    return out


class _AllToAllLookup(torch.autograd.Function):
    """Rows `ids` of a row-sharded table, routed to and from their owners with all-to-alls."""

    @staticmethod
    def forward(ctx, local_weight, ids, rows_per_rank: int, group, grad_scale: float, out_dtype):
        world_size, rank = dist.get_world_size(group), dist.get_rank(group)
        # Repeated ids are fetched once. `unique` sorts, which also orders the ids by owner.
        unique_ids, inverse = torch.unique(ids, sorted=True, return_inverse=True)
        send_counts = torch.bincount(unique_ids // rows_per_rank, minlength=world_size)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=group)
        send_splits, recv_splits = send_counts.tolist(), recv_counts.tolist()

        recv_ids = _all_to_all(unique_ids, recv_splits, send_splits, group)
        local_rows = recv_ids - rank * rows_per_rank
        unique_rows = _all_to_all(local_weight[local_rows].to(out_dtype), send_splits, recv_splits, group)

        ctx.save_for_backward(inverse, local_rows)
        ctx.splits = (send_splits, recv_splits)
        ctx.group, ctx.grad_scale, ctx.n_unique = group, grad_scale, unique_ids.numel()
        ctx.weight_shape, ctx.weight_dtype = local_weight.shape, local_weight.dtype
        return unique_rows[inverse]

    @staticmethod
    def backward(ctx, grad_out):
        inverse, local_rows = ctx.saved_tensors
        send_splits, recv_splits = ctx.splits
        # Repeated ids sum in fp32 before the trip to their owner.
        grad_unique = grad_out.new_zeros(ctx.n_unique, grad_out.shape[-1], dtype=torch.float32)
        grad_unique.index_add_(0, inverse.flatten(), grad_out.reshape(-1, grad_out.shape[-1]).float())
        grad_rows = _all_to_all(grad_unique.to(grad_out.dtype), recv_splits, send_splits, ctx.group)
        grad_weight = torch.zeros(ctx.weight_shape, dtype=ctx.weight_dtype, device=grad_out.device)
        grad_weight.index_add_(0, local_rows, grad_rows.to(ctx.weight_dtype), alpha=ctx.grad_scale)
        return grad_weight, None, None, None, None, None


class AllToAllEmbeddingParallel(ParallelStyle):
    """Row-sharded embedding served by all-to-all lookups, for tables too large to gather or replicate.

    Each rank keeps a contiguous slice of rows as a `Shard(0)` DTensor. A lookup sends the rank's
    deduplicated ids to the ranks owning them and receives only those rows; backward retraces the
    route with gradients, so no rank ever touches another's rows. `RowwiseParallel` instead
    all-gathers every rank's ids and reduce-scatters partial rows for all of them, whose traffic
    grows with the number of ranks. The weight must stay out of FSDP (`ignored_params`).

    Gradients sum over every rank's tokens, then are divided by `grad_divide_factor` to match the
    averaging FSDP applies to other parameters. Rows travel in `output_dtype`.
    """

    def __init__(self, grad_divide_factor: int, output_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.grad_divide_factor = grad_divide_factor
        self.output_dtype = output_dtype

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        assert isinstance(module, nn.Embedding) and device_mesh.ndim == 1
        weight = module.weight
        rows_per_rank = -(-module.num_embeddings // device_mesh.size())
        start = device_mesh.get_local_rank() * rows_per_rank
        # Slicing works on meta and materialized weights alike; ranks past the last row hold none.
        local = weight.detach()[start : start + rows_per_rank].clone()
        sharded = DTensor.from_local(
            local, device_mesh, [Shard(0)], run_check=False, shape=weight.shape, stride=weight.stride()
        )
        module.weight = nn.Parameter(sharded, requires_grad=weight.requires_grad)
        group = device_mesh.get_group()
        grad_scale, output_dtype = 1.0 / self.grad_divide_factor, self.output_dtype

        def forward(ids: torch.Tensor) -> torch.Tensor:
            local_weight = module.weight.to_local()
            rows = _AllToAllLookup.apply(local_weight, ids.flatten(), rows_per_rank, group, grad_scale, output_dtype)
            return rows.view(*ids.shape, -1)

        module.forward = forward
        # NCCL sets up the point-to-point connections an all-to-all needs at its first use. Do that
        # now, before weights and activations fill the GPU, instead of in the middle of a forward.
        probe = torch.zeros(device_mesh.size(), device=torch.cuda.current_device())
        dist.all_to_all_single(torch.empty_like(probe), probe, group=group)
        return module
