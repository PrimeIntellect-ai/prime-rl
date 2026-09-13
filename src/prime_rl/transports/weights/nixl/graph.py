"""Trace, plan, and replay composed weight-loading graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import torch

from prime_rl.transports.weights.nixl.copy_regions import CopyRegion, transform_regions
from prime_rl.transports.weights.nixl.trainer_tensor_table import TrainerTensorTable


@dataclass(frozen=True)
class TensorOperation:
    """One replayable tensor method invocation."""

    name: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


OperationChain = tuple[TensorOperation, ...]


@dataclass(frozen=True)
class TensorReplayPlan:
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
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float32)


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


def plan_tensor_replay(shape: tuple[int, ...], dtype: torch.dtype, ops: OperationChain) -> TensorReplayPlan:
    """Resolve a directly transferable source view and local replay suffix.

    The prefix must remain a contiguous, same-dtype view of the trainer root
    and can therefore be transferred in large RDMA runs. The first strided,
    materializing, or dtype-changing operation and everything after it is
    replayed on the receive arena.
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
        if candidate.dtype != dtype or not is_view_of(candidate, root) or not candidate.is_contiguous():
            break
        prefix_len = candidate_len
        source_view = candidate
    return TensorReplayPlan(
        source_offset=source_view.storage_offset(),
        source_shape=tuple(source_view.shape),
        source_stride=tuple(source_view.stride()),
        replay_ops=ops[prefix_len:],
    )


@dataclass
class RecordedCopy:
    source_name: str
    ops: OperationChain
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
class WeightLoadRecorder:
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
        source_name: str | None,
        source_shape: torch.Size,
        source_dtype: torch.dtype,
        device: torch.device,
        recorder: WeightLoadRecorder,
        ops: OperationChain = (),
        *,
        source_operation: TensorOperation | None = None,
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
        value._source_operation = source_operation
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

    def _child(self, *ops: TensorOperation) -> "LazyWeight":
        return LazyWeight(
            self._source_name,
            self._source_shape,
            self._source_dtype,
            self.device,
            self._recorder,
            self._ops + ops,
            source_operation=self._source_operation,
        )

    @classmethod
    def _record_concatenation(cls, tensors: Iterable["LazyWeight"], dim: int = 0) -> "LazyWeight":
        """Record a cat node while keeping every trainer input independent."""
        inputs = tuple(tensors)
        if not inputs or not all(isinstance(weight, cls) for weight in inputs):
            raise UnsupportedOpError("lazy concatenation requires only lazy weight sources")
        first_input = inputs[0]
        for input_weight in inputs:
            if input_weight.device != first_input.device or input_weight._recorder is not first_input._recorder:
                raise UnsupportedOpError("lazy concatenation requires matching device and recorder")

        concatenated_meta = torch.cat([weight._meta() for weight in inputs], dim=dim)
        concat_dimension = dim % concatenated_meta.ndim
        return cls(
            None,
            concatenated_meta.shape,
            concatenated_meta.dtype,
            first_input.device,
            first_input._recorder,
            source_operation=TensorOperation("cat", args=(inputs,), kwargs={"dim": concat_dimension}),
        )

    def _resolve_copy_regions(self) -> list[CopyRegion]:
        """Lower the composed graph to independent sources and destination regions."""
        if self._source_operation is None:
            return [CopyRegion(self, (0,) * self.ndim)]

        (input_weights,) = self._source_operation.args
        concat_dimension = self._source_operation.kwargs["dim"]
        regions = []
        concat_offset = 0
        for input_weight in input_weights:
            if input_weight.numel() == 0:
                continue
            for region in input_weight._resolve_copy_regions():
                offsets = list(region.offsets)
                offsets[concat_dimension] += concat_offset
                value = region.value
                if value.dtype != self._source_dtype:
                    value = value.to(dtype=self._source_dtype)
                regions.append(CopyRegion(value, tuple(offsets)))
            concat_offset += input_weight.shape[concat_dimension]

        meta = torch.empty(self._source_shape, dtype=self._source_dtype, device="meta")
        operations = iter(self._ops)
        for operation in operations:
            output_index = None
            if operation.name in ("chunk", "split", "unbind"):
                selection = next(operations)
                assert selection.name == "tuple_getitem"
                output_index = selection.args[0]
            regions, meta = transform_regions(regions, meta, operation, output_index)
        return regions

    def _record_copy(self, destination: torch.Tensor) -> torch.Tensor:
        if isinstance(destination, LazyWeight):
            raise UnsupportedOpError("copy_ between lazy graph tensors is not supported")
        if tuple(destination.shape) != tuple(self.shape):
            raise UnsupportedOpError(
                f"copy_ shape mismatch for {self._source_name}: {tuple(self.shape)} -> {tuple(destination.shape)}"
            )
        if self.dtype not in _SUPPORTED_DTYPES or destination.dtype not in _SUPPORTED_DTYPES:
            raise UnsupportedOpError(
                f"NIXL lazy copies only support BF16/FP32 values, got "
                f"source={self.dtype}, destination={destination.dtype} for {self._source_name!r}"
            )

        # Indexed assignment enters through ATen with __torch_function__ disabled.
        with torch._C._EnableTorchFunction():
            for region in self._resolve_copy_regions():
                destination_slice = destination
                for dimension, (offset, size) in enumerate(zip(region.offsets, region.value.shape, strict=True)):
                    destination_slice = destination_slice.narrow(dimension, offset, size)
                resolved_destination = self._recorder.resolve_destination(destination_slice)
                if resolved_destination is None:
                    continue
                owner, destination_offset = resolved_destination
                self._recorder.copies.append(
                    RecordedCopy(
                        source_name=region.value._source_name,
                        ops=region.value._ops,
                        destination_module=owner.module,
                        destination_name=owner.name,
                        destination_offset=destination_offset,
                        destination_shape=tuple(destination_slice.shape),
                        destination_stride=tuple(destination_slice.stride()),
                        is_persistent=not destination_slice.is_meta,
                    )
                )
        # Loaders use copy_ for its side effect; the trace must never mutate
        # live kernel storage or attempt a meta-to-device copy.
        return destination

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in (torch.cat, torch.concat, torch.concatenate):
            return cls._record_concatenation(*args, **kwargs)
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
        operation = TensorOperation(name=op_name, args=args, kwargs=dict(kwargs))
        if isinstance(result, torch.Tensor):
            if result.dtype not in _SUPPORTED_DTYPES:
                raise UnsupportedOpError(
                    f"NIXL lazy replay only supports BF16/FP32 values, got {result.dtype} "
                    f"after {op_name!r} on {source._source_name!r}"
                )
            return source._child(operation)
        if isinstance(result, (tuple, list)) and all(
            isinstance(item, torch.Tensor) and item.dtype in _SUPPORTED_DTYPES for item in result
        ):
            return tuple(
                source._child(
                    operation,
                    TensorOperation(name="tuple_getitem", args=(index,)),
                )
                for index, _ in enumerate(result)
            )
        raise UnsupportedOpError(f"operation {op_name!r} returned unsupported {type(result).__name__}")

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        # Indexed assignment lowers to a native copy without calling Tensor.copy_.
        if func is torch.ops.aten.copy_.default and isinstance(args[1], cls):
            return args[1]._record_copy(args[0])

        for value in (*args, *kwargs.values()):
            if isinstance(value, cls):
                raise UnsupportedOpError(
                    f"unsupported operation {func} on {value._source_name!r}, recorded chain={value._ops!r}"
                )
        return func(*args, **kwargs)


def make_hf_lazy_weights(
    table: TrainerTensorTable,
    *,
    device: torch.device,
    recorder: WeightLoadRecorder,
    hf_config,
) -> list[tuple[str, LazyWeight]]:
    """Create HF-named graph values rooted in trainer wire tensors.

    The returned values retain their trainer root name and accumulated view
    chain. Passing them to ``vLLM.model.load_weights`` composes the second half
    of the graph without any handwritten vLLM kernel conversion.
    """
    state: dict[str, LazyWeight] = {
        tensor.name: LazyWeight(
            tensor.name,
            torch.Size(tensor.shape),
            getattr(torch, tensor.wire_dtype),
            device,
            recorder,
        )
        for group in table.groups
        for tensor in group.tensors
    }

    # TODO(matej): Figure out how to avoid depending on trainer code here.
    from prime_rl.trainer.models import get_custom_causal_lm_cls
    from prime_rl.trainer.models.conversion_ops import apply_prime_to_hf

    model_cls = get_custom_causal_lm_cls(hf_config)
    apply_prime_to_hf(state, model_cls.conversion_chain(hf_config))

    # AutoWeightsLoader groups adjacent names by module prefix. Stable sorting
    # matches normal checkpoint iterators and keeps every expert group intact.
    return sorted(state.items(), key=lambda item: item[0])
