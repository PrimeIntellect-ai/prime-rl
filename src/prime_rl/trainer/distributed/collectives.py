"""Autograd and compile-friendly distributed collectives."""

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup


def _resolve_group(group_name: str) -> ProcessGroup:
    return dist.distributed_c10d._resolve_process_group(group_name)


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
        group=_resolve_group(group_name),
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
    dist.all_to_all_single(output, x.contiguous(), group=_resolve_group(group_name))
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


def _mxfp8_kernel():
    import prime_kernels

    return prime_kernels.load("mxfp8_moe")


@torch.library.custom_op("prime_rl_collectives::mxfp8_all_to_all", mutates_args=())
def _mxfp8_all_to_all(
    x: torch.Tensor,
    output_splits: torch.Tensor,
    input_splits: torch.Tensor,
    group_name: str,
    quantized: bool,
) -> torch.Tensor:
    kernel = _mxfp8_kernel()
    operation = kernel.all_to_all_dispatch if quantized else kernel.all_to_all_combine
    return operation(
        x,
        output_splits.tolist(),
        input_splits.tolist(),
        _resolve_group(group_name),
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
    dist.all_gather_into_tensor(output, gathered, group=_resolve_group(group_name))
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
    dist.reduce_scatter_tensor(output, scattered, group=_resolve_group(group_name))
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


def all_to_all_single(
    x: torch.Tensor,
    output_splits: torch.Tensor,
    input_splits: torch.Tensor,
    group: ProcessGroup,
) -> torch.Tensor:
    return _all_to_all_single(x, output_splits, input_splits, group.group_name)


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
    "all_to_all_single_equal",
    "mxfp8_all_to_all_combine",
    "mxfp8_all_to_all_dispatch",
]
