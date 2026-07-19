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


@dataclass(frozen=True)
class TensorTransferPlan:
    """Source view to pull directly and operations to replay locally."""

    source_offset: int
    source_shape: tuple[int, ...]
    source_stride: tuple[int, ...]
    replay_ops: OperationChain


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


def is_view_of(value: torch.Tensor, root: torch.Tensor) -> bool:
    """Whether ``value`` is ``root`` or a view backed by ``root``."""
    if value is root:
        return True
    current = value
    seen: set[int] = set()
    while current._base is not None and id(current) not in seen:
        seen.add(id(current))
        current = current._base
        if current is root:
            return True
    return False


def plan_tensor_transfer(
    shape: tuple[int, ...], dtype: torch.dtype, ops: OperationChain
) -> TensorTransferPlan:
    """Resolve a directly transferable source view and local replay suffix.

    The prefix must remain a same-dtype view of the trainer root and can
    therefore be addressed by RDMA. The first materializing or dtype-changing
    operation and everything after it is replayed on the receive arena.
    """
    root = torch.empty(shape, dtype=dtype, device="meta")
    prefix_len = 0
    source_view = root
    for candidate_len in range(1, len(ops) + 1):
        # Tuple-returning operations and their tuple_getitem are recorded as
        # one logical operation, so they must remain on the same side.
        if candidate_len < len(ops) and ops[candidate_len].name == "tuple_getitem":
            continue
        candidate = apply_chain(root, ops[:candidate_len])
        if candidate.dtype != dtype or not is_view_of(candidate, root):
            break
        prefix_len = candidate_len
        source_view = candidate
    return TensorTransferPlan(
        source_offset=source_view.storage_offset(),
        source_shape=tuple(source_view.shape),
        source_stride=tuple(source_view.stride()),
        replay_ops=ops[prefix_len:],
    )
