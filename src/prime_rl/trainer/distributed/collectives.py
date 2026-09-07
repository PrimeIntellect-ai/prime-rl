"""Autograd and compile-friendly distributed collectives."""

import prime_kernels
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup


@torch.library.custom_op("prime_rl_collectives::all_to_all_single", mutates_args=())
def _all_to_all_single(
    x: torch.Tensor,
    output_splits: torch.Tensor,
    input_splits: torch.Tensor,
    group_name: str,
) -> torch.Tensor:
    output_split_list = output_splits.tolist()
    input_split_list = input_splits.tolist()
    output = x.new_empty((sum(output_split_list), *x.shape[1:]))
    dist.all_to_all_single(
        output,
        x.contiguous(),
        output_split_list,
        input_split_list,
        group=dist.distributed_c10d._resolve_process_group(group_name),
    )
    return output


@_all_to_all_single.register_fake
def _all_to_all_single_fake(
    x: torch.Tensor,
    output_splits: torch.Tensor,
    input_splits: torch.Tensor,
    group_name: str,
) -> torch.Tensor:
    output_size = torch.library.get_ctx().new_dynamic_size()
    return x.new_empty((output_size, *x.shape[1:]))


def _all_to_all_setup_context(ctx, inputs, output) -> None:
    _, output_splits, input_splits, group_name = inputs
    ctx.save_for_backward(output_splits, input_splits)
    ctx.group_name = group_name


def _all_to_all_backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
    output_splits, input_splits = ctx.saved_tensors
    return (
        _all_to_all_single(
            grad_output,
            input_splits,
            output_splits,
            ctx.group_name,
        ),
        None,
        None,
        None,
    )


_all_to_all_single.register_autograd(_all_to_all_backward, setup_context=_all_to_all_setup_context)


@torch.library.custom_op("prime_rl_collectives::all_to_all_single_equal", mutates_args=())
def _all_to_all_single_equal(x: torch.Tensor, group_name: str) -> torch.Tensor:
    output = x.new_empty(x.shape)
    dist.all_to_all_single(
        output,
        x.contiguous(),
        group=dist.distributed_c10d._resolve_process_group(group_name),
    )
    return output


@_all_to_all_single_equal.register_fake
def _all_to_all_single_equal_fake(x: torch.Tensor, group_name: str) -> torch.Tensor:
    return x.new_empty(x.shape)


def _all_to_all_equal_backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
    return _all_to_all_single_equal(grad_output, ctx.group_name), None


def _all_to_all_equal_setup_context(ctx, inputs, output) -> None:
    _, group_name = inputs
    ctx.group_name = group_name


_all_to_all_single_equal.register_autograd(
    _all_to_all_equal_backward,
    setup_context=_all_to_all_equal_setup_context,
)


@torch.library.custom_op("prime_rl_collectives::mxfp8_all_to_all", mutates_args=())
def _mxfp8_all_to_all(
    x: torch.Tensor,
    output_splits: torch.Tensor,
    input_splits: torch.Tensor,
    group_name: str,
    quantized: bool,
) -> torch.Tensor:
    kernel = prime_kernels.load("mxfp8_moe")
    operation = kernel.all_to_all_dispatch if quantized else kernel.all_to_all_combine
    return operation(
        x,
        output_splits.tolist(),
        input_splits.tolist(),
        dist.distributed_c10d._resolve_process_group(group_name),
    )


@_mxfp8_all_to_all.register_fake
def _mxfp8_all_to_all_fake(
    x: torch.Tensor,
    output_splits: torch.Tensor,
    input_splits: torch.Tensor,
    group_name: str,
    quantized: bool,
) -> torch.Tensor:
    output_size = torch.library.get_ctx().new_dynamic_size()
    return x.new_empty((output_size, *x.shape[1:]))


def _mxfp8_all_to_all_setup_context(ctx, inputs, output) -> None:
    _, output_splits, input_splits, group_name, quantized = inputs
    ctx.save_for_backward(output_splits, input_splits)
    ctx.group_name = group_name
    ctx.quantized = quantized


def _mxfp8_all_to_all_backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None, None]:
    output_splits, input_splits = ctx.saved_tensors
    return (
        _mxfp8_all_to_all(
            grad_output,
            input_splits,
            output_splits,
            ctx.group_name,
            not ctx.quantized,
        ),
        None,
        None,
        None,
        None,
    )


_mxfp8_all_to_all.register_autograd(
    _mxfp8_all_to_all_backward,
    setup_context=_mxfp8_all_to_all_setup_context,
)


@torch.library.custom_op("prime_rl_collectives::all_gather", mutates_args=())
def _all_gather(x: torch.Tensor, dim: int, group_size: int, group_name: str) -> torch.Tensor:
    gathered = x.movedim(dim, 0).contiguous()
    output = gathered.new_empty((gathered.shape[0] * group_size, *gathered.shape[1:]))
    dist.all_gather_into_tensor(
        output,
        gathered,
        group=dist.distributed_c10d._resolve_process_group(group_name),
    )
    return output.movedim(0, dim).contiguous()


@_all_gather.register_fake
def _all_gather_fake(x: torch.Tensor, dim: int, group_size: int, group_name: str) -> torch.Tensor:
    shape = list(x.shape)
    shape[dim] *= group_size
    return x.new_empty(shape)


@torch.library.custom_op("prime_rl_collectives::reduce_scatter_sum", mutates_args=())
def _reduce_scatter_sum(x: torch.Tensor, dim: int, group_size: int, group_name: str) -> torch.Tensor:
    scattered = x.movedim(dim, 0).contiguous()
    output = scattered.new_empty((scattered.shape[0] // group_size, *scattered.shape[1:]))
    dist.reduce_scatter_tensor(
        output,
        scattered,
        group=dist.distributed_c10d._resolve_process_group(group_name),
    )
    return output.movedim(0, dim).contiguous()


@_reduce_scatter_sum.register_fake
def _reduce_scatter_sum_fake(x: torch.Tensor, dim: int, group_size: int, group_name: str) -> torch.Tensor:
    shape = list(x.shape)
    shape[dim] //= group_size
    return x.new_empty(shape)


def _collective_setup_context(ctx, inputs, output) -> None:
    _, dim, group_size, group_name = inputs
    ctx.dim = dim
    ctx.group_size = group_size
    ctx.group_name = group_name


def _all_gather_backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
    return (
        _reduce_scatter_sum(grad_output, ctx.dim, ctx.group_size, ctx.group_name),
        None,
        None,
        None,
    )


def _reduce_scatter_backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
    return (
        _all_gather(grad_output, ctx.dim, ctx.group_size, ctx.group_name),
        None,
        None,
        None,
    )


_all_gather.register_autograd(_all_gather_backward, setup_context=_collective_setup_context)
_reduce_scatter_sum.register_autograd(_reduce_scatter_backward, setup_context=_collective_setup_context)


_all_to_all_work: dict[int, dist.Work] = {}
_all_to_all_handle_counter = 0
_async_all_to_all_lib: torch.library.Library | None = None
_async_all_to_all_registered = False


def _next_all_to_all_handle() -> torch.Tensor:
    global _all_to_all_handle_counter
    _all_to_all_handle_counter += 1
    return torch.tensor([_all_to_all_handle_counter], dtype=torch.int64, device="cpu")


def _all_to_all_single_async_impl(
    x: torch.Tensor,
    output_splits: torch.Tensor,
    input_splits: torch.Tensor,
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    output_split_list = output_splits.tolist()
    input_split_list = input_splits.tolist()
    output = x.new_empty((sum(output_split_list), *x.shape[1:]))
    work = dist.all_to_all_single(
        output,
        x.contiguous(),
        output_split_list,
        input_split_list,
        group=dist.distributed_c10d._resolve_process_group(group_name),
        async_op=True,
    )
    handle_id = _next_all_to_all_handle()
    _all_to_all_work[handle_id.item()] = work
    return output, handle_id


def _all_to_all_single_async_setup_context(ctx, inputs, output) -> None:
    _, output_splits, input_splits, group_name = inputs
    ctx.save_for_backward(output_splits, input_splits)
    ctx.group_name = group_name


def _all_to_all_single_async_backward(
    ctx, grad_output: torch.Tensor | None, grad_handle_id: torch.Tensor | None
) -> tuple[torch.Tensor | None, None, None, None]:
    if grad_output is None:
        return None, None, None, None
    output_splits, input_splits = ctx.saved_tensors
    # Gradient of an all-to-all is the reverse-direction all-to-all; the forward pass already
    # pipelines the data transfer, so the backward can afford to synchronize immediately.
    grad_x = _all_to_all_single(grad_output, input_splits, output_splits, ctx.group_name)
    return grad_x, None, None, None


def register_async_all_to_all_op() -> None:
    """Register `prime_rl_collectives::all_to_all_single_async` on first use.

    Deferred registration (mirroring `deepep.register_deepep_cuda_ops`) keeps this op out of the
    fake-tensor/meta dispatch machinery: the pending `dist.Work` handle can't be represented there,
    so callers must keep synchronization points (`sync_all_to_all`) outside `torch.compile`.
    """
    global _async_all_to_all_lib, _async_all_to_all_registered
    if _async_all_to_all_registered:
        return

    _async_all_to_all_lib = torch.library.Library("prime_rl_collectives", "FRAGMENT")
    _async_all_to_all_lib.define(
        "all_to_all_single_async(Tensor x, Tensor output_splits, Tensor input_splits, str group_name) "
        "-> (Tensor, Tensor)"
    )
    torch.library.impl(_async_all_to_all_lib, "all_to_all_single_async", "CUDA")(_all_to_all_single_async_impl)
    torch.library.register_autograd(
        "prime_rl_collectives::all_to_all_single_async",
        _all_to_all_single_async_backward,
        setup_context=_all_to_all_single_async_setup_context,
    )

    _async_all_to_all_registered = True


@torch.compiler.disable()
def sync_all_to_all(handle_id: torch.Tensor | int | None) -> None:
    """Wait for a pending async all-to-all. Cheap: records a stream-wait, does not block the host."""
    if handle_id is None:
        return
    handle_key = handle_id if isinstance(handle_id, int) else handle_id.item()
    work = _all_to_all_work.pop(handle_key, None)
    if work is not None:
        work.wait()


def all_to_all_single(
    x: torch.Tensor,
    output_splits: torch.Tensor,
    input_splits: torch.Tensor,
    group: ProcessGroup,
) -> torch.Tensor:
    return _all_to_all_single(x, output_splits, input_splits, group.group_name)


def all_to_all_single_async(
    x: torch.Tensor,
    output_splits: torch.Tensor,
    input_splits: torch.Tensor,
    group: ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Issue an all-to-all without blocking the current stream on its completion.

    Returns the (not-yet-populated) output tensor and an opaque handle. Call `sync_all_to_all`
    with the handle right before the output is consumed, so independent work issued in between
    (e.g. another chunk's expert compute) overlaps with this collective.
    """
    register_async_all_to_all_op()
    return torch.ops.prime_rl_collectives.all_to_all_single_async(x, output_splits, input_splits, group.group_name)


def all_to_all_single_equal(x: torch.Tensor, group: ProcessGroup) -> torch.Tensor:
    return _all_to_all_single_equal(x, group.group_name)


def mxfp8_all_to_all_dispatch(
    x: torch.Tensor,
    output_splits: torch.Tensor,
    input_splits: torch.Tensor,
    group: ProcessGroup,
) -> torch.Tensor:
    return _mxfp8_all_to_all(x, output_splits, input_splits, group.group_name, True)


def mxfp8_all_to_all_combine(
    x: torch.Tensor,
    output_splits: torch.Tensor,
    input_splits: torch.Tensor,
    group: ProcessGroup,
) -> torch.Tensor:
    return _mxfp8_all_to_all(x, output_splits, input_splits, group.group_name, False)


def all_gather(x: torch.Tensor, dim: int, group: ProcessGroup) -> torch.Tensor:
    return _all_gather(x, dim, group.size(), group.group_name)


__all__ = [
    "all_gather",
    "all_to_all_single",
    "all_to_all_single_async",
    "all_to_all_single_equal",
    "mxfp8_all_to_all_combine",
    "mxfp8_all_to_all_dispatch",
    "sync_all_to_all",
]
