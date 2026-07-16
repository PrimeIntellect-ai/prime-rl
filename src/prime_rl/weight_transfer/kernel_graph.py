"""Record and replay vLLM's logical-weight to kernel-weight processing.

The graph is discovered by running vLLM's real
``process_weights_after_loading`` under :class:`KernelGraphRecorder`.  Logical
parameters are registered as graph inputs before the call and the tensors left
on the layer afterwards are registered as outputs.  PyTorch dispatcher calls
between those two points form the graph.

This deliberately records operators rather than model names.  A new vLLM
loader or MoE backend therefore describes its own kernel layout by executing
its normal post-load path; prime-rl does not need a parallel table of Qwen,
Nemotron, or backend-specific conversions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import msgspec
import torch
from torch.utils._python_dispatch import TorchDispatchMode


@dataclass(frozen=True)
class TensorMeta:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "TensorMeta":
        return cls(tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)


@dataclass(frozen=True)
class TensorRef:
    kind: Literal["input", "constant", "op"]
    index: int
    output: int = 0


@dataclass(frozen=True)
class _RuntimeDevice:
    pass


@dataclass(frozen=True)
class RecordedOperation:
    target: str
    overload: str
    args: Any
    kwargs: Any
    outputs: tuple[TensorMeta, ...]


@dataclass
class KernelGraph:
    """An executable tensor graph with named logical inputs and kernel outputs."""

    input_names: tuple[str, ...]
    input_metas: tuple[TensorMeta, ...]
    constants: tuple[torch.Tensor, ...]
    operations: tuple[RecordedOperation, ...]
    outputs: dict[str, TensorRef]

    def replay(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if set(inputs) != set(self.input_names):
            missing = sorted(set(self.input_names) - set(inputs))
            extra = sorted(set(inputs) - set(self.input_names))
            raise KeyError(f"kernel graph input mismatch: missing={missing}, extra={extra}")

        input_values = tuple(inputs[name] for name in self.input_names)
        for name, value, meta in zip(self.input_names, input_values, self.input_metas):
            if tuple(value.shape) != meta.shape or value.dtype != meta.dtype:
                raise ValueError(
                    f"kernel graph input {name!r} has shape/dtype "
                    f"{tuple(value.shape)}/{value.dtype}; expected {meta.shape}/{meta.dtype}"
                )

        device = input_values[0].device if input_values else torch.device("cpu")
        constants = tuple(value.to(device=device) for value in self.constants)
        results: list[tuple[torch.Tensor, ...]] = []

        def resolve(value: Any) -> Any:
            if isinstance(value, _RuntimeDevice):
                return device
            if isinstance(value, TensorRef):
                if value.kind == "input":
                    return input_values[value.index]
                if value.kind == "constant":
                    return constants[value.index]
                return results[value.index][value.output]
            if isinstance(value, tuple):
                return tuple(resolve(item) for item in value)
            if isinstance(value, list):
                return [resolve(item) for item in value]
            if isinstance(value, dict):
                return {key: resolve(item) for key, item in value.items()}
            return value

        for operation in self.operations:
            func = _resolve_operator(operation.target, operation.overload)
            result = func(*resolve(operation.args), **resolve(operation.kwargs))
            tensors = tuple(_tensor_leaves(result))
            if len(tensors) != len(operation.outputs):
                raise RuntimeError(
                    f"operator {operation.target}.{operation.overload} produced "
                    f"{len(tensors)} tensor outputs; recorded {len(operation.outputs)}"
                )
            for actual, expected in zip(tensors, operation.outputs):
                if tuple(actual.shape) != expected.shape or actual.dtype != expected.dtype:
                    raise RuntimeError(
                        f"operator {operation.target}.{operation.overload} replay metadata mismatch: "
                        f"{tuple(actual.shape)}/{actual.dtype} != {expected.shape}/{expected.dtype}"
                    )
            results.append(tensors)

        return {name: resolve(reference) for name, reference in self.outputs.items()}

    def encode(self) -> bytes:
        """Encode the graph as data-only msgpack for ModelExpress.

        Operator names are resolved against the local PyTorch installation on
        replay.  No Python callable or pickle payload crosses the control
        plane.
        """

        payload = {
            "input_names": self.input_names,
            "input_metas": [_encode_meta(meta) for meta in self.input_metas],
            "constants": [_encode_tensor(tensor) for tensor in self.constants],
            "operations": [
                {
                    "target": operation.target,
                    "overload": operation.overload,
                    "args": _encode_value(operation.args),
                    "kwargs": _encode_value(operation.kwargs),
                    "outputs": [_encode_meta(meta) for meta in operation.outputs],
                }
                for operation in self.operations
            ],
            "outputs": {name: _encode_ref(reference) for name, reference in self.outputs.items()},
        }
        return msgspec.msgpack.encode(payload)

    @classmethod
    def decode(cls, payload: bytes) -> "KernelGraph":
        value = msgspec.msgpack.decode(payload)
        return cls(
            input_names=tuple(value["input_names"]),
            input_metas=tuple(_decode_meta(meta) for meta in value["input_metas"]),
            constants=tuple(_decode_tensor(tensor) for tensor in value["constants"]),
            operations=tuple(
                RecordedOperation(
                    target=operation["target"],
                    overload=operation["overload"],
                    args=_decode_value(operation["args"]),
                    kwargs=_decode_value(operation["kwargs"]),
                    outputs=tuple(_decode_meta(meta) for meta in operation["outputs"]),
                )
                for operation in value["operations"]
            ),
            outputs={name: _decode_ref(reference) for name, reference in value["outputs"].items()},
        )


class KernelGraphRecorder(TorchDispatchMode):
    """Record operators depending on registered logical weight tensors.

    Operations unrelated to a logical input are not retained.  If such a
    tensor later participates in a retained operation (for example a vLLM
    generated permutation index), it is captured as an immutable constant.
    """

    def __init__(self) -> None:
        super().__init__()
        self._input_names: list[str] = []
        self._input_metas: list[TensorMeta] = []
        self._constants: list[torch.Tensor] = []
        self._operations: list[RecordedOperation] = []
        self._refs: dict[int, TensorRef] = {}
        self._storage_refs: dict[tuple[Any, ...], TensorRef] = {}

    def add_input(self, name: str, tensor: torch.Tensor) -> None:
        if name in self._input_names:
            raise ValueError(f"duplicate kernel graph input {name!r}")
        reference = TensorRef("input", len(self._input_names))
        self._input_names.append(name)
        self._input_metas.append(TensorMeta.from_tensor(tensor))
        self._remember(tensor, reference)

    def finish(self, outputs: dict[str, torch.Tensor]) -> KernelGraph:
        references: dict[str, TensorRef] = {}
        for name, tensor in outputs.items():
            reference = self._lookup(tensor)
            if reference is None:
                raise RuntimeError(
                    f"kernel output {name!r} does not depend on a registered logical input; "
                    "register every post-load parameter and buffer before recording"
                )
            references[name] = reference
        return KernelGraph(
            input_names=tuple(self._input_names),
            input_metas=tuple(self._input_metas),
            constants=tuple(self._constants),
            operations=tuple(self._operations),
            outputs=references,
        )

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        depends_on_input = any(self._lookup(tensor) is not None for tensor in _tensor_leaves((args, kwargs)))
        result = func(*args, **kwargs)
        if not depends_on_input:
            return result

        encoded_args = self._encode(args)
        encoded_kwargs = self._encode(kwargs)
        tensors = tuple(_tensor_leaves(result))
        if not tensors:
            raise NotImplementedError(f"kernel conversion operator {func} returned no tensors")

        target, overload = _operator_name(func)
        operation_index = len(self._operations)
        self._operations.append(
            RecordedOperation(
                target=target,
                overload=overload,
                args=encoded_args,
                kwargs=encoded_kwargs,
                outputs=tuple(TensorMeta.from_tensor(tensor) for tensor in tensors),
            )
        )
        for output_index, tensor in enumerate(tensors):
            self._remember(tensor, TensorRef("op", operation_index, output_index))
        return result

    def _encode(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            reference = self._lookup(value)
            if reference is None:
                reference = TensorRef("constant", len(self._constants))
                self._constants.append(value.detach().cpu().clone())
                self._remember(value, reference)
            return reference
        if isinstance(value, tuple):
            return tuple(self._encode(item) for item in value)
        if isinstance(value, list):
            return [self._encode(item) for item in value]
        if isinstance(value, dict):
            return {key: self._encode(item) for key, item in value.items()}
        return value

    def _lookup(self, tensor: torch.Tensor) -> TensorRef | None:
        reference = self._refs.get(id(tensor))
        if reference is not None:
            return reference
        return self._storage_refs.get(_storage_key(tensor))

    def _remember(self, tensor: torch.Tensor, reference: TensorRef) -> None:
        self._refs[id(tensor)] = reference
        self._storage_refs[_storage_key(tensor)] = reference


def _storage_key(tensor: torch.Tensor) -> tuple[Any, ...]:
    try:
        storage = tensor.untyped_storage()._cdata
    except (NotImplementedError, RuntimeError):
        storage = id(tensor)
    return (
        tensor.device.type,
        tensor.device.index,
        storage,
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
    )


def _tensor_leaves(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensor_leaves(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensor_leaves(item)


def _operator_name(func: Any) -> tuple[str, str]:
    schema = getattr(func, "_schema", None)
    if schema is None or "::" not in schema.name:
        raise NotImplementedError(f"kernel conversion used a non-serializable operator {func!r}")
    return schema.name, getattr(func, "_overloadname", "default") or "default"


def _resolve_operator(target: str, overload: str):
    namespace, name = target.split("::", 1)
    packet = getattr(getattr(torch.ops, namespace), name)
    return getattr(packet, overload)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _dtype_from_name(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown torch dtype {name!r}")
    return dtype


def _encode_meta(meta: TensorMeta) -> dict[str, Any]:
    return {"shape": meta.shape, "stride": meta.stride, "dtype": _dtype_name(meta.dtype)}


def _decode_meta(value: dict[str, Any]) -> TensorMeta:
    return TensorMeta(tuple(value["shape"]), tuple(value["stride"]), _dtype_from_name(value["dtype"]))


def _encode_ref(reference: TensorRef) -> list[Any]:
    return [reference.kind, reference.index, reference.output]


def _decode_ref(value: list[Any]) -> TensorRef:
    return TensorRef(value[0], value[1], value[2])


def _encode_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    cpu = tensor.detach().cpu().contiguous()
    raw = cpu.view(torch.uint8).numpy().tobytes()
    return {"shape": tuple(cpu.shape), "dtype": _dtype_name(cpu.dtype), "data": raw}


def _decode_tensor(value: dict[str, Any]) -> torch.Tensor:
    dtype = _dtype_from_name(value["dtype"])
    # bytearray gives torch writable storage and avoids retaining the msgpack
    # decoder's full payload through a frombuffer view.
    raw = torch.frombuffer(bytearray(value["data"]), dtype=torch.uint8)
    return raw.view(dtype).reshape(tuple(value["shape"])).clone()


def _encode_value(value: Any) -> Any:
    if isinstance(value, TensorRef):
        return {"__type__": "tensor_ref", "value": _encode_ref(value)}
    if isinstance(value, torch.dtype):
        return {"__type__": "dtype", "value": _dtype_name(value)}
    if isinstance(value, torch.device):
        return {"__type__": "runtime_device"}
    if isinstance(value, slice):
        return {
            "__type__": "slice",
            "start": _encode_value(value.start),
            "stop": _encode_value(value.stop),
            "step": _encode_value(value.step),
        }
    if value is Ellipsis:
        return {"__type__": "ellipsis"}
    if isinstance(value, tuple):
        return {"__type__": "tuple", "items": [_encode_value(item) for item in value]}
    if isinstance(value, list):
        return [_encode_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _encode_value(item) for key, item in value.items()}
    if value in (torch.contiguous_format, torch.preserve_format, torch.channels_last, torch.channels_last_3d):
        return {"__type__": "memory_format", "value": str(value).split(".")[-1]}
    if value is torch.strided:
        return {"__type__": "layout", "value": "strided"}
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    raise NotImplementedError(f"cannot serialize kernel graph argument {value!r} ({type(value).__name__})")


def _decode_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    kind = value.get("__type__")
    if kind == "tensor_ref":
        return _decode_ref(value["value"])
    if kind == "dtype":
        return _dtype_from_name(value["value"])
    if kind == "runtime_device":
        return _RuntimeDevice()
    if kind == "slice":
        return slice(_decode_value(value["start"]), _decode_value(value["stop"]), _decode_value(value["step"]))
    if kind == "ellipsis":
        return Ellipsis
    if kind == "tuple":
        return tuple(_decode_value(item) for item in value["items"])
    if kind == "memory_format":
        return getattr(torch, value["value"])
    if kind == "layout":
        return getattr(torch, value["value"])
    return {key: _decode_value(item) for key, item in value.items()}


def encode_graph_value(value: Any) -> bytes:
    """Encode an op argument tree (also used by load-slice plans)."""

    return msgspec.msgpack.encode(_encode_value(value))


def decode_graph_value(payload: bytes) -> Any:
    return _decode_value(msgspec.msgpack.decode(payload))
