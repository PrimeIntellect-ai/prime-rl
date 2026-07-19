"""Recorded tensor operations and transport/replay planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class TensorOperation:
    """One replayable tensor method invocation."""

    name: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


OperationChain = tuple[TensorOperation, ...]

SUPPORTED_OPS: dict[Any, str] = {
    torch.Tensor.narrow: "narrow",
    torch.Tensor.select: "select",
    torch.Tensor.view: "view",
    torch.Tensor.reshape: "reshape",
    torch.Tensor.__getitem__: "__getitem__",
    torch.Tensor.unsqueeze: "unsqueeze",
    torch.Tensor.squeeze: "squeeze",
    torch.Tensor.transpose: "transpose",
    torch.Tensor.t: "t",
    torch.Tensor.permute: "permute",
    torch.Tensor.flatten: "flatten",
    torch.Tensor.contiguous: "contiguous",
    torch.Tensor.chunk: "chunk",
    torch.Tensor.split: "split",
    torch.Tensor.unbind: "unbind",
    torch.Tensor.to: "to",
    torch.Tensor.float: "float",
    torch.Tensor.bfloat16: "bfloat16",
}
MAX_RUNS_PER_COPY = 1 << 16


class UnsupportedOpError(NotImplementedError):
    pass


def apply_chain(value: Any, ops: OperationChain) -> torch.Tensor:
    """Replay a recorded chain, including deferred dtype conversions."""
    # Invariant: LazyWeight records only operations from SUPPORTED_OPS and
    # follows tuple-returning methods with tuple_getitem, so every chain ends
    # in a tensor.
    result = value
    for operation in ops:
        if operation.name == "tuple_getitem":
            result = result[operation.args[0]]
        elif operation.name == "__getitem__":
            result = result[operation.args[0]]
        else:
            result = getattr(result, operation.name)(*operation.args, **operation.kwargs)
    return result


def _shares_root_storage(root: torch.Tensor, result: torch.Tensor) -> bool:
    if result is root:
        return True
    current = result
    seen: set[int] = set()
    while current._base is not None and id(current) not in seen:
        seen.add(id(current))
        current = current._base
        if current is root:
            return True
    return False


def split_transport_chain(
    shape: tuple[int, ...], dtype: torch.dtype, ops: OperationChain
) -> tuple[OperationChain, OperationChain, tuple[int, ...]]:
    """Split a graph into a source-view prefix and local replay suffix.

    The prefix must remain a same-dtype view of the trainer root and can
    therefore be addressed by RDMA. The first materializing or dtype-changing
    operation and everything after it is replayed on the receive arena.
    """
    root = torch.empty(shape, dtype=dtype, device="meta")
    prefix_len = 0
    prefix_value: torch.Tensor = root
    for index in range(len(ops)):
        candidate_ops = ops[: index + 1]
        try:
            candidate = apply_chain(root, candidate_ops)
        except (UnsupportedOpError, RuntimeError, TypeError, ValueError):
            break
        if candidate.dtype != dtype or not _shares_root_storage(root, candidate):
            break
        prefix_len = index + 1
        prefix_value = candidate
    return ops[:prefix_len], ops[prefix_len:], tuple(prefix_value.shape)


def resolve_chain_region(
    shape: tuple[int, ...], dtype: torch.dtype, ops: OperationChain
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    root = torch.empty(shape, dtype=dtype, device="meta")
    result = apply_chain(root, ops)
    if not _shares_root_storage(root, result):
        raise UnsupportedOpError(f"transport prefix materializes data: {ops!r}")
    return result.storage_offset(), tuple(result.shape), tuple(result.stride())


def region_elem_runs(offset_elems: int, shape: tuple[int, ...], stride: tuple[int, ...]) -> list[tuple[int, int]]:
    numel = 1
    for size in shape:
        numel *= size
    if numel == 0:
        return []

    dims = [(size, step) for size, step in zip(shape, stride) if size != 1]
    if any(step < 0 for _, step in dims):
        raise UnsupportedOpError("negative strides are not supported")
    if not dims:
        return [(offset_elems, 1)]

    run_elems = 1
    split_at = len(dims)
    while split_at and dims[split_at - 1][1] == run_elems:
        run_elems *= dims[split_at - 1][0]
        split_at -= 1
    outer = dims[:split_at]

    num_runs = 1
    for size, _ in outer:
        num_runs *= size
    if num_runs > MAX_RUNS_PER_COPY:
        raise UnsupportedOpError(
            f"region shape={shape}, stride={stride} requires {num_runs} RDMA runs (maximum {MAX_RUNS_PER_COPY})"
        )

    runs: list[tuple[int, int]] = []

    def emit(dim: int, offset: int) -> None:
        if dim == len(outer):
            runs.append((offset, run_elems))
            return
        size, step = outer[dim]
        for index in range(size):
            emit(dim + 1, offset + index * step)

    emit(0, offset_elems)
    return runs


def tensor_runs(view: torch.Tensor) -> list[tuple[int, int]]:
    itemsize = view.element_size()
    base = view.data_ptr()
    return [
        (base + offset * itemsize, length * itemsize)
        for offset, length in region_elem_runs(0, tuple(view.shape), tuple(view.stride()))
    ]
