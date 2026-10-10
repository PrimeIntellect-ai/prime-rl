import functools
import threading
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton
import triton.language as tl
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


@dataclass
class _Route:
    """Where one lookup's ids live: which rows this rank serves to others, and how to undo the dedup."""

    inverse: torch.Tensor
    n_unique: int
    send_splits: list[int]
    recv_splits: list[int]
    local_rows: torch.Tensor
    # The rows this rank serves, gathered ahead of time (host-resident tables only).
    served: torch.Tensor | None = None
    served_ready: torch.cuda.Event | None = None


def _route(ids: torch.Tensor, rows_per_rank: int, group) -> _Route:
    world_size, rank = dist.get_world_size(group), dist.get_rank(group)
    # Repeated ids are fetched once. `unique` sorts, which also orders the ids by owner.
    unique_ids, inverse = torch.unique(ids, sorted=True, return_inverse=True)
    send_counts = torch.bincount(unique_ids // rows_per_rank, minlength=world_size)
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts, group=group)
    send_splits, recv_splits = send_counts.tolist(), recv_counts.tolist()
    recv_ids = _all_to_all(unique_ids, recv_splits, send_splits, group)
    return _Route(inverse, unique_ids.numel(), send_splits, recv_splits, recv_ids - rank * rows_per_rank)


class _AllToAllLookup(torch.autograd.Function):
    """Rows of a row-sharded table along a `_Route`, sent to and from their owners with all-to-alls."""

    @staticmethod
    def forward(ctx, local_weight, route: _Route, group, grad_scale: float, out_dtype, module: nn.Module):
        if route.served is None:
            served = local_weight[route.local_rows].to(out_dtype)
        else:
            torch.cuda.current_stream().wait_event(route.served_ready)
            served = route.served
        unique_rows = _all_to_all(served, route.send_splits, route.recv_splits, group)

        ctx.save_for_backward(route.inverse, route.local_rows)
        ctx.splits = (route.send_splits, route.recv_splits)
        ctx.group, ctx.grad_scale, ctx.n_unique = group, grad_scale, route.n_unique
        ctx.weight_shape, ctx.weight_dtype = local_weight.shape, local_weight.dtype
        ctx.module = module
        return unique_rows[route.inverse]

    @staticmethod
    def backward(ctx, grad_out):
        inverse, local_rows = ctx.saved_tensors
        send_splits, recv_splits = ctx.splits
        # Repeated ids sum in fp32 before the trip to their owner.
        grad_unique = grad_out.new_zeros(ctx.n_unique, grad_out.shape[-1], dtype=torch.float32)
        grad_unique.index_add_(0, inverse.flatten(), grad_out.reshape(-1, grad_out.shape[-1]).float())
        grad_rows = _all_to_all(grad_unique.to(grad_out.dtype), recv_splits, send_splits, ctx.group)
        param = ctx.module.weight
        if param.grad is not None and not getattr(param, "_post_accumulate_grad_hooks", None):
            # Later micro-batches add their rows straight into the step's gradient instead of returning a dense
            # one for autograd to add. Rows requested by several ranks are summed first, as the dense one would.
            rows, inverse_rows = torch.unique(local_rows, return_inverse=True)
            summed = torch.zeros(rows.numel(), grad_rows.shape[-1], dtype=ctx.weight_dtype, device=grad_out.device)
            summed.index_add_(0, inverse_rows, grad_rows.to(ctx.weight_dtype), alpha=ctx.grad_scale)
            grad = param.grad.to_local() if isinstance(param.grad, DTensor) else param.grad
            grad.index_add_(0, rows, summed)
            return None, None, None, None, None, None
        grad_weight = torch.zeros(ctx.weight_shape, dtype=ctx.weight_dtype, device=grad_out.device)
        grad_weight.index_add_(0, local_rows, grad_rows.to(ctx.weight_dtype), alpha=ctx.grad_scale)
        return grad_weight, None, None, None, None, None


class _CudaArrayInterface:
    def __init__(self, tensor: torch.Tensor) -> None:
        assert tensor.dtype == torch.float32
        self.__cuda_array_interface__ = {
            "shape": tuple(tensor.shape),
            "typestr": "<f4",
            "data": (tensor.data_ptr(), False),
            "strides": None,
            "version": 3,
        }


@triton.jit
def _gather_rows_kernel(src, rows, out, n_rows, D: tl.constexpr, BLOCK_ROWS: tl.constexpr):
    cols = tl.arange(0, D)
    for start in range(tl.program_id(0) * BLOCK_ROWS, n_rows, tl.num_programs(0) * BLOCK_ROWS):
        r = start + tl.arange(0, BLOCK_ROWS)
        mask = r < n_rows
        idx = tl.load(rows + r, mask=mask, other=0)
        vals = tl.load(src + idx[:, None] * D + cols[None, :], mask=mask[:, None])
        out_offsets = r.to(tl.int64)[:, None] * D + cols[None, :]
        tl.store(out + out_offsets, vals.to(out.dtype.element_ty), mask=mask[:, None])


def _gather_host_rows(src: torch.Tensor, rows: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """`src[rows].to(dtype)` for a device alias of pinned host memory, on a capped grid.

    The reads cross PCIe, so each thread block waits most of its life; a full grid (as `src[rows]`
    launches) would hold every SM and stall the compute running beside it."""
    out = torch.empty(rows.numel(), src.shape[1], dtype=dtype, device=rows.device)
    block_rows = 8
    # 16 blocks already saturate PCIe (~50 GB/s on H200).
    grid = (min(triton.cdiv(rows.numel(), block_rows), 16),)
    if rows.numel():
        _gather_rows_kernel[grid](src, rows, out, rows.numel(), D=src.shape[1], BLOCK_ROWS=block_rows, num_warps=4)
    return out


@functools.cache
def _gather_stream() -> torch.cuda.Stream:
    """One side stream per process for all host-table gathers, so they run one at a time."""
    return torch.cuda.Stream()


class HostTable:
    """A table shard in pinned host memory, with its optimizer state.

    GPU kernels gather rows straight out of it over PCIe (with unified addressing a pinned host pointer
    is also a device pointer), so the GPU never holds the shard. The optimizer update streams the whole
    shard through the GPU in row chunks and runs the same update kernel the GPU optimizer would, so
    every row is updated every step exactly as if the shard lived on the GPU.

    To keep the update off the critical path, `step` only records it. Gathers apply it to the rows
    they fetch (an elementwise update gives those rows the same values the full pass will write)
    until the first lookup that consumes them starts the full pass (`start_update`), which then
    overlaps with the rest of the micro-batch; later gathers and the next `step` wait for it.
    """

    def __init__(self, local_weight: torch.Tensor, chunk_rows: int = 1 << 13) -> None:
        assert local_weight.dtype == torch.float32, "the host table keeps the fp32 master weight"
        self.weight = torch.empty(local_weight.shape, dtype=local_weight.dtype, pin_memory=True)
        self.weight.copy_(local_weight)
        self.state: list[torch.Tensor] = []
        self.on_device = [self._device_alias(self.weight)]
        self.chunk_rows = chunk_rows
        self.gather_stream = _gather_stream()
        self.copy_in, self.compute, self.copy_out = torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()
        self.updated: torch.cuda.Event | None = None
        self.pending = None
        self.worker: threading.Thread | None = None
        self.worker_error: Exception | None = None

    @staticmethod
    def _device_alias(host: torch.Tensor) -> torch.Tensor:
        alias = torch.as_tensor(_CudaArrayInterface(host), device=torch.cuda.current_device())
        assert alias.data_ptr() == host.data_ptr(), "pinned host memory is not device-addressable"
        return alias

    def gather(self, rows: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.cuda.Event]:
        """Start fetching `rows` (as of the last optimizer step) on a side stream."""
        stream = self.gather_stream
        stream.wait_stream(torch.cuda.current_stream())
        self._join()
        if self.updated is not None:
            stream.wait_event(self.updated)
        with torch.cuda.stream(stream):
            if self.pending is None:
                out = _gather_host_rows(self.on_device[0], rows, dtype)
            else:
                grad, update = self.pending
                param, *state = (_gather_host_rows(alias, rows, alias.dtype) for alias in self.on_device)
                update(param, grad[rows], state)
                out = param.to(dtype)
            ready = torch.cuda.Event()
            ready.record()
        out.record_stream(torch.cuda.current_stream())
        rows.record_stream(stream)
        return out, ready

    def start_update(self) -> None:
        """Start the recorded update of the whole shard, once a lookup has consumed its gathered rows.

        Starting it any earlier would compete for PCIe with gathers the forward is about to wait on."""
        if self.pending is not None:
            grad, update = self.pending
            self.pending = None
            self._update_all(grad, update, after=self.gather_stream)

    def step(self, grad: torch.Tensor, n_state: int, update) -> None:
        """Record `update(param, grad, state)` (elementwise, in place) for every row; see the class doc.

        `grad` must stay unchanged until the update is done."""
        if self.pending is not None:
            raise RuntimeError("an optimizer step ran with no lookup since the previous one")
        self.wait()
        if not self.state:
            self.state = [torch.zeros(self.weight.shape, pin_memory=True) for _ in range(n_state)]
            self.on_device += [self._device_alias(t) for t in self.state]
        grad.record_stream(self.gather_stream)
        self.pending = (grad, update)

    def wait(self) -> None:
        """Make the current stream wait for the update in flight, which reads the optimizer's step state."""
        self._join()
        if self.updated is not None:
            torch.cuda.current_stream().wait_event(self.updated)

    def flush(self) -> None:
        """Apply a recorded update to the whole shard now (e.g. before reading `weight` on the host)."""
        self.gather_stream.wait_stream(torch.cuda.current_stream())
        self.start_update()
        self._join()
        if self.updated is not None:
            self.updated.synchronize()

    def _join(self) -> None:
        if self.worker is not None:
            self.worker.join()
            self.worker = None
            if self.worker_error is not None:
                raise self.worker_error

    def _update_all(self, grad: torch.Tensor, update, after: torch.cuda.Stream) -> None:
        """Stream every row through `update`, issued from a thread that keeps at most two chunks in flight.

        Copies queue on the GPU's copy engines in issue order, so enqueuing the whole shard at once
        would hold back every other copy (FSDP's all-gather copy-in, host syncs) until it is done."""
        start = torch.cuda.Event()
        start.record(after)
        self.worker_error = None
        self.updated = torch.cuda.Event()
        self.worker = threading.Thread(target=self._stream_chunks, args=(grad, update, start))
        self.worker.start()

    def _stream_chunks(self, grad, update, start) -> None:
        try:
            torch.cuda.set_device(grad.device)
            host, n_rows, chunk = [self.weight, *self.state], self.weight.shape[0], self.chunk_rows
            loaded, computed, stored = ([torch.cuda.Event() for _ in range(2)] for _ in range(3))
            for stream in (self.copy_in, self.compute, self.copy_out):
                stream.wait_event(start)
            with torch.cuda.stream(self.copy_in):
                buffers = [[torch.empty_like(t[:chunk], device=grad.device) for t in host] for _ in range(2)]
            for i, row in enumerate(range(0, n_rows, chunk)):
                b, n = i % 2, min(chunk, n_rows - row)
                if i >= 2:
                    stored[b].synchronize()
                with torch.cuda.stream(self.copy_in):
                    for buf, src in zip(buffers[b], host):
                        buf[:n].copy_(src[row : row + n], non_blocking=True)
                    loaded[b].record()
                with torch.cuda.stream(self.compute):
                    self.compute.wait_event(loaded[b])
                    update(buffers[b][0][:n], grad[row : row + n], [buf[:n] for buf in buffers[b][1:]])
                    computed[b].record()
                with torch.cuda.stream(self.copy_out):
                    self.copy_out.wait_event(computed[b])
                    for buf, dst in zip(buffers[b], host):
                        dst[row : row + n].copy_(buf[:n], non_blocking=True)
                    stored[b].record()
            self.updated.record(self.copy_out)
            # Keep `grad` and the buffers alive until the update is done.
            self.updated.synchronize()
        except Exception as error:
            self.worker_error = error


class AllToAllEmbeddingParallel(ParallelStyle):
    """Row-sharded embedding served by all-to-all lookups, for tables too large to gather or replicate.

    Each rank keeps a contiguous slice of rows as a `Shard(0)` DTensor. A lookup sends the rank's
    deduplicated ids to the ranks owning them and receives only those rows; backward retraces the
    route with gradients, so no rank ever touches another's rows. `RowwiseParallel` instead
    all-gathers every rank's ids and reduce-scatters partial rows for all of them, whose traffic
    grows with the number of ranks. The weight must stay out of FSDP (`ignored_params`).

    Gradients sum over every rank's tokens, then are divided by `grad_divide_factor` to match the
    averaging FSDP applies to other parameters. Rows travel in `output_dtype`.

    `module.prefetch(ids)` routes a lookup ahead of its forward (the ids only depend on the input
    tokens), so the host sync on the split sizes happens before the model's first layer instead of
    in the middle of the forward; forwards consume prefetched routes in order. After
    `offload_to_host(module)` the shard lives in pinned host memory and a prefetch also starts
    gathering the rows this rank serves.
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
        module.host_table = None
        routes: list[_Route] = []

        def route_and_gather(ids: torch.Tensor) -> _Route:
            route = _route(ids.flatten(), rows_per_rank, group)
            if module.host_table is not None:
                route.served, route.served_ready = module.host_table.gather(route.local_rows, output_dtype)
            return route

        def prefetch(ids: torch.Tensor) -> None:
            routes.append(route_and_gather(ids))

        def forward(ids: torch.Tensor) -> torch.Tensor:
            route = routes.pop(0) if routes else route_and_gather(ids)
            assert route.inverse.numel() == ids.numel(), "forward ids differ from the prefetched ones"
            rows = _AllToAllLookup.apply(module.weight.to_local(), route, group, grad_scale, output_dtype, module)
            if module.host_table is not None:
                module.host_table.start_update()
            return rows.view(*ids.shape, -1)

        module.prefetch = prefetch
        module.forward = forward
        # NCCL sets up the point-to-point connections an all-to-all needs at its first use. Do that
        # now, before weights and activations fill the GPU, instead of in the middle of a forward.
        probe = torch.zeros(device_mesh.size(), device=torch.cuda.current_device())
        dist.all_to_all_single(torch.empty_like(probe), probe, group=group)
        return module


def offload_to_host(module: nn.Module) -> None:
    """Move an `AllToAllEmbeddingParallel` table's shard to pinned host memory (`module.host_table`).

    The parameter keeps its shape, so gradients, norms and clipping see it as before, but its local
    tensor becomes a zero-stride view of one element: the GPU only ever holds its gradient."""
    weight = module.weight
    local = weight.to_local()
    module.host_table = HostTable(local)
    placeholder = torch.zeros((), dtype=local.dtype, device=local.device).expand(local.shape)
    module.weight = nn.Parameter(
        DTensor.from_local(
            placeholder,
            weight.device_mesh,
            weight.placements,
            run_check=False,
            shape=weight.shape,
            stride=weight.stride(),
        ),
        requires_grad=weight.requires_grad,
    )
    module.weight.host_table = module.host_table
    # FSDP's ignored-parameter set still references the old parameter; free its storage regardless.
    local.untyped_storage().resize_(0)

    def refuse_state_dict(*args, **kwargs):
        raise NotImplementedError("saving or loading a host-offloaded engram table is not implemented")

    module.register_state_dict_pre_hook(refuse_state_dict)
    module.register_load_state_dict_pre_hook(refuse_state_dict)
