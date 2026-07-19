"""Zero-storage tensors used to compose trainer and vLLM load graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from prime_rl.weight_transfer.chains import SUPPORTED_OPS, OpChain, OpSpec, UnsupportedOpError, apply_chain

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float32)


@dataclass
class RecordedCopy:
    source_name: str
    ops: OpChain
    destination_module: Any
    destination_name: str
    destination_offset: int
    destination_shape: tuple[int, ...]
    destination_stride: tuple[int, ...]
    is_persistent: bool = False


@dataclass(frozen=True, eq=False)
class Destination:
    """A vLLM destination tensor and its owning module attribute."""

    module: Any
    name: str
    tensor: torch.Tensor

    def storage_offset_of(self, tensor: torch.Tensor) -> int | None:
        base_addr = self.tensor.data_ptr()
        addr = tensor.data_ptr()
        nbytes = self.tensor.numel() * self.tensor.element_size()
        if base_addr <= addr < base_addr + nbytes:
            return (addr - base_addr) // self.tensor.element_size()
        return None


@dataclass
class BakeRecorder:
    copies: list[RecordedCopy] = field(default_factory=list)
    active_destination: Destination | None = None
    destination_storage_ranges: list[Destination] = field(default_factory=list)

    def register_destination_storage(self, destination: Destination) -> None:
        self.destination_storage_ranges.append(destination)

    def resolve_destination(self, tensor: torch.Tensor) -> tuple[Destination, int] | None:
        if self.active_destination is not None:
            return self.active_destination, tensor.storage_offset()
        if tensor.is_meta:
            return None
        for destination in self.destination_storage_ranges:
            offset = destination.storage_offset_of(tensor)
            if offset is not None:
                return destination, offset
        return None


class LazyWeight(torch.Tensor):
    """Wrapper tensor that records views, casts, and terminal ``copy_`` calls."""

    @staticmethod
    def __new__(
        cls,
        source_name: str,
        source_shape: torch.Size,
        source_dtype: torch.dtype,
        device: torch.device,
        recorder: BakeRecorder,
        ops: OpChain = (),
    ) -> "LazyWeight":
        meta = apply_chain(torch.empty(source_shape, dtype=source_dtype, device="meta"), ops)
        value = torch.Tensor._make_wrapper_subclass(
            cls,
            meta.shape,
            strides=meta.stride(),
            storage_offset=meta.storage_offset(),
            dtype=meta.dtype,
            device=device,
            requires_grad=False,
        )
        value._source_name = source_name
        value._source_shape = torch.Size(source_shape)
        value._source_dtype = source_dtype
        value._ops = tuple(ops)
        value._recorder = recorder
        return value

    def __repr__(self) -> str:
        return (
            f"LazyWeight(source_name={self._source_name!r}, shape={tuple(self.shape)}, "
            f"dtype={self.dtype}, ops={self._ops!r})"
        )

    def _meta(self) -> torch.Tensor:
        source = torch.empty(self._source_shape, dtype=self._source_dtype, device="meta")
        return apply_chain(source, self._ops)

    def _child(self, *ops: OpSpec) -> "LazyWeight":
        return LazyWeight(
            self._source_name,
            self._source_shape,
            self._source_dtype,
            self.device,
            self._recorder,
            self._ops + ops,
        )

    def _record_copy(self, destination: torch.Tensor) -> torch.Tensor:
        if isinstance(destination, LazyWeight):
            raise UnsupportedOpError("copy_ between lazy graph tensors is not supported")
        if tuple(destination.shape) != tuple(self.shape):
            raise UnsupportedOpError(
                f"copy_ shape mismatch for {self._source_name}: "
                f"{tuple(self.shape)} -> {tuple(destination.shape)}"
            )
        if self.dtype not in _SUPPORTED_DTYPES or destination.dtype not in _SUPPORTED_DTYPES:
            raise UnsupportedOpError(
                f"NIXL lazy copies only support BF16/FP32 values, got "
                f"source={self.dtype}, destination={destination.dtype} for {self._source_name!r}"
            )

        resolved_destination = self._recorder.resolve_destination(destination)
        if resolved_destination is not None:
            owner, destination_offset = resolved_destination
            self._recorder.copies.append(
                RecordedCopy(
                    source_name=self._source_name,
                    ops=self._ops,
                    destination_module=owner.module,
                    destination_name=owner.name,
                    destination_offset=destination_offset,
                    destination_shape=tuple(destination.shape),
                    destination_stride=tuple(destination.stride()),
                    is_persistent=not destination.is_meta,
                )
            )
        # Loaders use copy_ for its side effect; the bake must never mutate
        # live kernel storage or attempt a meta-to-device copy.
        return destination

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.Tensor.copy_:
            destination = args[0]
            source = args[1] if len(args) > 1 else kwargs.get("src")
            if isinstance(source, cls):
                return source._record_copy(destination)

        op_name = SUPPORTED_OPS.get(func)
        if op_name is not None and args and isinstance(args[0], cls):
            return cls._intercept(args[0], func, op_name, tuple(args[1:]), kwargs)

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def _intercept(
        cls,
        source: "LazyWeight",
        func: Callable,
        op_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ):
        meta = source._meta()
        with torch._C.DisableTorchFunctionSubclass():
            result = func(meta, *args, **kwargs)
        op: OpSpec = (op_name, args, dict(kwargs))
        if isinstance(result, torch.Tensor):
            if result.dtype not in _SUPPORTED_DTYPES:
                raise UnsupportedOpError(
                    f"NIXL lazy replay only supports BF16/FP32 values, got {result.dtype} "
                    f"after {op_name!r} on {source._source_name!r}"
                )
            return source._child(op)
        if isinstance(result, (tuple, list)) and all(
            isinstance(item, torch.Tensor) and item.dtype in _SUPPORTED_DTYPES for item in result
        ):
            return tuple(
                source._child(op, ("tuple_getitem", (index,), {}))
                for index, _ in enumerate(result)
            )
        raise UnsupportedOpError(f"operation {op_name!r} returned unsupported {type(result).__name__}")

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        for value in (*args, *kwargs.values()):
            if isinstance(value, cls):
                raise UnsupportedOpError(
                    f"unsupported operation {func} on {value._source_name!r}, recorded chain={value._ops!r}"
                )
        return func(*args, **kwargs)
