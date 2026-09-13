"""Trace, plan, and replay composed weight-loading graphs."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import product
from math import prod
from typing import Any, Callable, Iterable

import torch

from prime_rl.transports.weights.nixl.trainer_tensor_table import TrainerTensorTable


@dataclass(frozen=True)
class TensorOperation:
    """One recorded tensor operation and its arguments."""

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
    """Evaluate recorded tensor operations; cat is evaluated only on metadata."""
    result = value
    for operation in ops:
        if operation.name == "cat":
            if not result.is_meta:
                raise UnsupportedOpError("concatenation must be lowered to independent copies before replay")
            (other_inputs,) = operation.args
            result = torch.cat([result, *(weight._meta() for weight in other_inputs)], **operation.kwargs)
        elif operation.name == "tuple_getitem":
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
        source_name: str,
        source_shape: torch.Size,
        source_dtype: torch.dtype,
        device: torch.device,
        recorder: WeightLoadRecorder,
        ops: OperationChain = (),
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

    def _child(self, *ops: TensorOperation) -> "LazyWeight":
        return LazyWeight(
            self._source_name,
            self._source_shape,
            self._source_dtype,
            self.device,
            self._recorder,
            self._ops + ops,
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

        return first_input._child(TensorOperation("cat", args=(inputs[1:],), kwargs={"dim": dim}))

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

        resolved_destination = self._recorder.resolve_destination(destination)
        if resolved_destination is not None:
            owner, destination_offset = resolved_destination
            copy = RecordedCopy(
                source_name=self._source_name,
                ops=self._ops,
                destination_module=owner.module,
                destination_name=owner.name,
                destination_offset=destination_offset,
                destination_shape=tuple(destination.shape),
                destination_stride=tuple(destination.stride()),
                is_persistent=not destination.is_meta,
            )
            # Indexed assignment enters through ATen with __torch_function__ disabled.
            with torch._C._EnableTorchFunction():
                self._recorder.copies.extend(lower_graph(self, copy))
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


def destination_coordinates(copy: RecordedCopy, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Locate a copy in a contiguous logical output before binding physical strides."""
    strides = torch.empty(shape, device="meta").stride()
    return tuple(copy.destination_offset // stride % size for size, stride in zip(shape, strides, strict=True))


def place_copy(
    copy: RecordedCopy, coordinates: tuple[int, ...], shape: tuple[int, ...], output_shape: tuple[int, ...]
) -> RecordedCopy:
    """Position a copy within the current logical output."""
    strides = torch.empty(output_shape, device="meta").stride()
    return replace(
        copy,
        destination_offset=sum(offset * stride for offset, stride in zip(coordinates, strides, strict=True)),
        destination_shape=shape,
        destination_stride=strides,
    )


def slice_copies(
    copies: list[RecordedCopy], shape: tuple[int, ...], index, output_shape: tuple[int, ...]
) -> list[RecordedCopy]:
    """Intersect a slice with each copy and translate it into source-local indexing."""
    indices = list(index if isinstance(index, tuple) else (index,))
    consumed_dimensions = sum(item is not None and item is not Ellipsis for item in indices)
    expanded = []
    for item in indices:
        if item is Ellipsis:
            expanded.extend([slice(None)] * (len(shape) - consumed_dimensions))
        else:
            expanded.append(item)
    if not any(item is Ellipsis for item in indices):
        expanded.extend([slice(None)] * (len(shape) - consumed_dimensions))

    selected_copies = []
    for copy in copies:
        input_offsets = destination_coordinates(copy, shape)
        source_index = []
        output_offsets = []
        dimension = 0
        for item in expanded:
            if item is None:
                source_index.append(None)
                output_offsets.append(0)
                continue
            input_start = input_offsets[dimension]
            input_end = input_start + copy.destination_shape[dimension]
            if isinstance(item, int):
                selected_index = item if item >= 0 else item + shape[dimension]
                if not input_start <= selected_index < input_end:
                    break
                source_index.append(selected_index - input_start)
            elif isinstance(item, slice):
                start, stop, step = item.indices(shape[dimension])
                first = max(0, (input_start - start + step - 1) // step)
                last = min(len(range(start, stop, step)), (input_end - start + step - 1) // step)
                if first >= last:
                    break
                local_start = start + first * step - input_start
                local_stop = min(input_end - input_start, local_start + (last - first) * step)
                source_index.append(slice(local_start, local_stop, step))
                output_offsets.append(first)
            else:
                raise NotImplementedError(f"unsupported copy index {item!r}")
            dimension += 1
        else:
            operation = TensorOperation("__getitem__", args=(tuple(source_index),))
            source_shape = apply_chain(torch.empty(copy.destination_shape, device="meta"), (operation,)).shape
            selected_copies.append(
                place_copy(
                    replace(copy, ops=copy.ops + (operation,)), tuple(output_offsets), source_shape, output_shape
                )
            )
    return selected_copies


def split_flat_interval(start: int, length: int, shape: tuple[int, ...]):
    """Partition a flat interval into contiguous rectangular output slices."""
    if not shape:
        yield (), (), 1
        return
    strides = tuple(prod(shape[dim + 1 :]) for dim in range(len(shape)))
    while length:
        coordinates = tuple(start // stride % size for size, stride in zip(shape, strides, strict=True))
        for dimension, stride in enumerate(strides):
            if start % stride == 0 and length >= stride:
                width = min(shape[dimension] - coordinates[dimension], length // stride)
                copy_shape = (1,) * dimension + (width,) + shape[dimension + 1 :]
                copy_length = prod(copy_shape)
                yield coordinates, copy_shape, copy_length
                start += copy_length
                length -= copy_length
                break


def reshape_copies(
    copies: list[RecordedCopy], old_shape: tuple[int, ...], new_shape: tuple[int, ...]
) -> list[RecordedCopy]:
    """Split copies where reshaping makes their logical destination slices nonrectangular."""
    if prod(new_shape) == 0:
        return []
    if old_shape == new_shape:
        return copies
    prefix_dimensions = 0
    for old_size, new_size in zip(old_shape, new_shape):
        if old_size != new_size:
            break
        prefix_dimensions += 1
    old_suffix = old_shape[prefix_dimensions:]
    new_suffix = new_shape[prefix_dimensions:]
    old_strides = tuple(prod(old_suffix[dim + 1 :]) for dim in range(len(old_suffix)))
    reshaped_copies = []
    for copy in copies:
        coordinates = destination_coordinates(copy, old_shape)
        prefix_shape = copy.destination_shape[:prefix_dimensions]
        prefix_offsets = coordinates[:prefix_dimensions]
        copy_shape = copy.destination_shape[prefix_dimensions:]
        copy_offsets = coordinates[prefix_dimensions:]
        split_dimension = len(old_suffix) - 1
        while split_dimension > 0 and copy_shape[split_dimension] == old_suffix[split_dimension]:
            split_dimension -= 1
        run_length = prod(copy_shape[split_dimension:])
        flatten_source = TensorOperation("reshape", args=(prefix_shape + (prod(copy_shape),),))
        source_offset = 0
        for outer_index in product(*(range(size) for size in copy_shape[:split_dimension])):
            coordinates = list(copy_offsets)
            for dimension, index in enumerate(outer_index):
                coordinates[dimension] += index
            flat_start = sum(index * stride for index, stride in zip(coordinates, old_strides, strict=True))
            for output_offsets, output_shape, length in split_flat_interval(flat_start, run_length, new_suffix):
                source_shape = prefix_shape + output_shape
                ops = copy.ops + (
                    flatten_source,
                    TensorOperation("narrow", args=(prefix_dimensions, source_offset, length)),
                    TensorOperation("reshape", args=(source_shape,)),
                )
                reshaped_copies.append(
                    place_copy(replace(copy, ops=ops), prefix_offsets + output_offsets, source_shape, new_shape)
                )
                source_offset += length
    return reshaped_copies


def map_operation_copies(
    copies: list[RecordedCopy],
    destination: RecordedCopy,
    meta: torch.Tensor,
    result: torch.Tensor | tuple[torch.Tensor, ...],
    source_ops: OperationChain,
) -> list[RecordedCopy]:
    """Update source chains and destination layouts using an operation's evaluated metadata."""
    operation = source_ops[0]
    name, args, kwargs = operation.name, operation.args, operation.kwargs
    shape = tuple(meta.shape)
    if name == "cat":
        return concatenate_copies(copies, destination, meta, result, operation)
    if len(source_ops) == 2:
        outputs = result
        output_index = source_ops[1].args[0]
        result = outputs[output_index]

    mapped = []
    partial = []
    for copy in copies:
        if copy.destination_offset == 0 and copy.destination_shape == shape:
            mapped.append(
                place_copy(
                    replace(copy, ops=copy.ops + source_ops),
                    (0,) * result.ndim,
                    tuple(result.shape),
                    tuple(result.shape),
                )
            )
        else:
            partial.append(copy)
    copies = partial
    if not copies:
        return mapped
    if name == "__getitem__":
        return mapped + slice_copies(copies, shape, args[0], result.shape)
    if name in ("narrow", "select"):
        dimension = kwargs.get("dim", args[0] if args else None) % meta.ndim
        start = kwargs.get("start" if name == "narrow" else "index", args[1] if len(args) > 1 else None)
        start = start if start >= 0 else start + shape[dimension]
        index = [slice(None)] * meta.ndim
        if name == "narrow":
            length = kwargs.get("length", args[2] if len(args) > 2 else None)
            index[dimension] = slice(start, start + length)
        else:
            index[dimension] = start
        return mapped + slice_copies(copies, shape, tuple(index), result.shape)
    if name in ("chunk", "split", "unbind"):
        argument_position = 0 if name == "unbind" else 1
        dimension = kwargs.get("dim", args[argument_position] if len(args) > argument_position else 0) % meta.ndim
        index = [slice(None)] * meta.ndim
        if name == "unbind":
            index[dimension] = output_index
        else:
            start = sum(output.shape[dimension] for output in outputs[:output_index])
            index[dimension] = slice(start, start + result.shape[dimension])
        return mapped + slice_copies(copies, shape, tuple(index), result.shape)
    if name in ("transpose", "t", "permute"):
        dimensions = list(range(meta.ndim))
        if name == "transpose":
            first = kwargs.get("dim0", args[0] if args else None)
            second = kwargs.get("dim1", args[1] if len(args) > 1 else None)
            dimensions[first], dimensions[second] = dimensions[second], dimensions[first]
        elif name == "t":
            dimensions.reverse()
        else:
            dimensions = kwargs.get("dims", args[0] if len(args) == 1 and isinstance(args[0], (tuple, list)) else args)
        operation = TensorOperation("permute", args=(tuple(dimensions),))
        permuted_copies = []
        for copy in copies:
            offsets = destination_coordinates(copy, shape)
            permuted_copies.append(
                place_copy(
                    replace(copy, ops=copy.ops + (operation,)),
                    tuple(offsets[dim] for dim in dimensions),
                    tuple(copy.destination_shape[dim] for dim in dimensions),
                    result.shape,
                )
            )
        return mapped + permuted_copies
    if name in ("view", "reshape", "flatten", "unsqueeze", "squeeze"):
        return mapped + reshape_copies(copies, shape, tuple(result.shape))
    if name in ("contiguous", "to", "float", "bfloat16"):
        return mapped + [replace(copy, ops=copy.ops + source_ops) for copy in copies]
    raise NotImplementedError(f"unsupported copy operation {name!r}")


def concatenate_copies(
    copies: list[RecordedCopy],
    destination: RecordedCopy,
    first: torch.Tensor,
    result: torch.Tensor,
    operation: TensorOperation,
) -> list[RecordedCopy]:
    """Place the current graph and each additional input into the concatenated output."""
    (other_inputs,) = operation.args
    dimension = operation.kwargs["dim"] % result.ndim
    inputs = [(copies, first)]
    for weight in other_inputs:
        input_destination = place_copy(destination, (0,) * weight.ndim, tuple(weight.shape), tuple(weight.shape))
        inputs.append((lower_graph(weight, input_destination), weight._meta()))

    concatenated = []
    concat_offset = 0
    for input_copies, input_meta in inputs:
        if input_meta.numel() == 0:
            continue
        for copy in input_copies:
            offsets = list(destination_coordinates(copy, tuple(input_meta.shape)))
            offsets[dimension] += concat_offset
            ops = copy.ops
            if input_meta.dtype != result.dtype:
                ops += (TensorOperation("to", kwargs={"dtype": result.dtype}),)
            # Cat's result is contiguous even for one noncontiguous input.
            ops += (TensorOperation("contiguous"),)
            concatenated.append(
                place_copy(replace(copy, ops=ops), tuple(offsets), copy.destination_shape, tuple(result.shape))
            )
        concat_offset += input_meta.shape[dimension]
    return concatenated


def lower_graph(weight: LazyWeight, destination: RecordedCopy) -> list[RecordedCopy]:
    """Interpret a graph from its source and bind all contributions to the destination."""
    root = torch.empty(weight._source_shape, dtype=weight._source_dtype, device="meta")
    source = replace(destination, source_name=weight._source_name, ops=())
    copies = [place_copy(source, (0,) * root.ndim, tuple(root.shape), tuple(root.shape))]
    meta = root
    operations = iter(weight._ops)
    for operation in operations:
        source_ops = (operation,)
        operation_result = apply_chain(meta, source_ops)
        if isinstance(operation_result, (tuple, list)):
            selection = next(operations)
            assert selection.name == "tuple_getitem"
            source_ops += (selection,)
            result = apply_chain(operation_result, (selection,))
        else:
            result = operation_result
        copies = map_operation_copies(copies, source, meta, operation_result, source_ops)
        meta = result
    result = []
    for copy in copies:
        if prod(copy.destination_shape) == 0:
            continue
        offsets = destination_coordinates(copy, tuple(meta.shape))
        result.append(
            replace(
                copy,
                destination_offset=destination.destination_offset
                + sum(offset * stride for offset, stride in zip(offsets, destination.destination_stride, strict=True)),
                destination_stride=destination.destination_stride,
            )
        )
    return result


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
