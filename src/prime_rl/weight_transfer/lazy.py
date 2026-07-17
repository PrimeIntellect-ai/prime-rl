"""Zero-storage tensors used to compose trainer and vLLM load graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from prime_rl.weight_transfer.chains import SUPPORTED_OPS, OpChain, OpSpec, UnsupportedOpError


@dataclass
class RecordedCopy:
    src_name: str
    ops: OpChain
    layer: Any
    param_name: str
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    persistent: bool = False


@dataclass
class BakeRecorder:
    copies: list[RecordedCopy] = field(default_factory=list)
    current: tuple[Any, str] | None = None
    live_destinations: list[tuple[int, int, Any, str, int]] = field(default_factory=list)

    def register_live_destination(self, layer: Any, name: str, tensor: torch.Tensor) -> None:
        self.live_destinations.append(
            (tensor.data_ptr(), tensor.numel() * tensor.element_size(), layer, name, tensor.element_size())
        )

    def destination(self, dst: torch.Tensor) -> tuple[Any, str, int, bool] | None:
        if self.current is not None:
            layer, name = self.current
            return layer, name, dst.storage_offset(), not dst.is_meta
        if dst.is_meta:
            return None
        pointer = dst.data_ptr()
        for base, nbytes, layer, name, itemsize in self.live_destinations:
            if base <= pointer < base + nbytes:
                return layer, name, (pointer - base) // itemsize, True
        return None


class LazyWeight(torch.Tensor):
    """Wrapper tensor that records views, casts, and terminal ``copy_`` calls."""

    @staticmethod
    def __new__(
        cls,
        name: str,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        recorder: BakeRecorder,
        ops: OpChain = (),
    ) -> "LazyWeight":
        value = torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
        value._name = name
        value._ops = tuple(ops)
        value._recorder = recorder
        return value

    def __repr__(self) -> str:
        return f"LazyWeight(name={self._name!r}, shape={tuple(self.shape)}, dtype={self.dtype}, ops={self._ops!r})"

    def _meta(self) -> torch.Tensor:
        return torch.empty(self.shape, dtype=self.dtype, device="meta")

    def _child(self, shape: torch.Size, dtype: torch.dtype, *ops: OpSpec) -> "LazyWeight":
        return LazyWeight(
            self._name,
            shape,
            dtype,
            self.device,
            self._recorder,
            self._ops + ops,
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.Tensor.copy_:
            dst = args[0]
            src = args[1] if len(args) > 1 else kwargs.get("src")
            if isinstance(src, cls):
                if isinstance(dst, cls):
                    raise UnsupportedOpError("copy_ between lazy graph tensors is not supported")
                if tuple(dst.shape) != tuple(src.shape):
                    raise UnsupportedOpError(
                        f"copy_ shape mismatch for {src._name}: {tuple(src.shape)} -> {tuple(dst.shape)}"
                    )
                supported_dtypes = (torch.bfloat16, torch.float32)
                if src.dtype not in supported_dtypes or dst.dtype not in supported_dtypes:
                    raise UnsupportedOpError(
                        f"NIXL lazy copies only support BF16/FP32 values, got "
                        f"source={src.dtype}, destination={dst.dtype} for {src._name!r}"
                    )
                destination = src._recorder.destination(dst)
                if destination is not None:
                    layer, param_name, offset, persistent = destination
                    src._recorder.copies.append(
                        RecordedCopy(
                            src_name=src._name,
                            ops=src._ops,
                            layer=layer,
                            param_name=param_name,
                            offset=offset,
                            shape=tuple(dst.shape),
                            stride=tuple(dst.stride()),
                            persistent=persistent,
                        )
                    )
                # Loaders use copy_ for its side effect; the bake must never
                # mutate live kernel storage or attempt a meta-to-device copy.
                return dst

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
            if result.dtype not in (torch.bfloat16, torch.float32):
                raise UnsupportedOpError(
                    f"NIXL lazy replay only supports BF16/FP32 values, got {result.dtype} "
                    f"after {op_name!r} on {source._name!r}"
                )
            return source._child(result.shape, result.dtype, op)
        if isinstance(result, (tuple, list)) and all(isinstance(item, torch.Tensor) for item in result):
            return tuple(
                source._child(item.shape, item.dtype, op, ("tuple_getitem", (index,), {}))
                for index, item in enumerate(result)
            )
        raise UnsupportedOpError(f"operation {op_name!r} returned unsupported {type(result).__name__}")

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        for value in (*args, *kwargs.values()):
            if isinstance(value, cls):
                raise UnsupportedOpError(
                    f"unsupported operation {func} on {value._name!r}, recorded chain={value._ops!r}"
                )
        return func(*args, **kwargs)
