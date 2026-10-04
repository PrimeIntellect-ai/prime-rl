from __future__ import annotations

import json
import os
import struct
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Protocol

import numba as nb
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

DELTA_INDEX_SUFFIX = ".__idx"
DELTA_VALUE_SUFFIX = ".__val"
DELTA_METADATA_FORMAT_KEY = "prime_rl_delta_format"
DELTA_METADATA_SHAPES_KEY = "prime_rl_delta_shapes"
DELTA_METADATA_LAYOUT_KEY = "prime_rl_delta_layout"
DELTA_METADATA_FORMAT_V1 = "sparse_delta_v1"
DELTA_METADATA_FORMAT_V2 = "sparse_delta_v2"
DELTA_METADATA_FORMAT_VALUE = DELTA_METADATA_FORMAT_V2
DELTA_LAYOUT_LOGICAL = "huggingface"
STREAMING_DELTA_FILENAME = "delta.stream"
SAFETENSORS_DELTA_FILENAME = "delta.safetensors"

_STREAM_MAGIC = b"PDELSTRM"
_STREAM_VERSION = 2
_STREAM_HEADER = struct.Struct("<8sI")
_STREAM_RECORD = struct.Struct("<I B B H Q Q")
_STREAM_INDEX_VARINT = 0
_STREAM_INDEX_RAW32 = 1
_STREAM_INDEX_RAW64 = 2
_STREAM_DTYPE_TO_CODE = {
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.float32: 3,
    torch.float64: 4,
    torch.int64: 5,
    torch.int32: 6,
    torch.int8: 7,
    torch.uint8: 8,
}
_STREAM_CODE_TO_DTYPE = {code: dtype for dtype, code in _STREAM_DTYPE_TO_CODE.items()}
_STREAM_DTYPE_SIZE = {dtype: torch.empty((), dtype=dtype).element_size() for dtype in _STREAM_DTYPE_TO_CODE}


@dataclass(frozen=True)
class DeltaStats:
    total_params: int
    changed_params: int
    change_percentage: float
    total_size_mb: float
    delta_size_mb: float
    compression_ratio: float


@dataclass(frozen=True)
class DeltaMismatch:
    name: str
    max_diff: float


@dataclass(frozen=True)
class DeltaVerificationResult:
    max_diff: float
    mismatches: tuple[DeltaMismatch, ...]

    @property
    def ok(self) -> bool:
        return len(self.mismatches) == 0


@dataclass(frozen=True)
class StreamingDeltaRecord:
    name: str
    shape: tuple[int, ...] | None
    encoded_indices: torch.Tensor
    values: torch.Tensor
    format_version: int


@dataclass(frozen=True)
class FusedPartRule:
    source_suffix: str
    fused_suffix: str
    order: int
    parts: int


@dataclass(frozen=True)
class MappedTensorName:
    name: str
    part_order: int | None
    parts: int | None


class TensorStore(Protocol):
    def keys(self) -> Iterable[str]: ...

    def get_tensor(self, name: str) -> torch.Tensor: ...


class StateDictTensorStore:
    def __init__(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        self._state_dict = state_dict

    def keys(self) -> Iterable[str]:
        return self._state_dict.keys()

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._state_dict[name]


class StreamingDeltaWriter:
    def __init__(self, path: str | Path):
        self._file = Path(path).open("wb")
        self._file.write(_STREAM_HEADER.pack(_STREAM_MAGIC, _STREAM_VERSION))
        self._file.flush()

    def write_record(
        self,
        name: str,
        shape: torch.Size | tuple[int, ...],
        indices: torch.Tensor,
        values: torch.Tensor,
        *,
        index_encoding: str,
    ) -> None:
        encoded_indices = encode_sparse_indices(indices, encoding=index_encoding).contiguous().cpu()
        values = values.reshape(-1).contiguous().cpu()
        index_format = _stream_index_format(encoded_indices)
        dtype_code = _STREAM_DTYPE_TO_CODE.get(values.dtype)
        if dtype_code is None:
            raise ValueError(f"unsupported streaming delta value dtype: {values.dtype}")

        name_bytes = name.encode("utf-8")
        shape = tuple(int(dim) for dim in shape)
        if len(shape) > 2**16 - 1:
            raise ValueError(f"streaming delta tensor rank is too large: {name}")
        index_bytes = encoded_indices.numel() * encoded_indices.element_size()
        self._file.write(
            _STREAM_RECORD.pack(
                len(name_bytes),
                index_format,
                dtype_code,
                len(shape),
                index_bytes,
                values.numel(),
            )
        )
        self._file.write(name_bytes)
        if shape:
            self._file.write(struct.pack(f"<{len(shape)}Q", *shape))
        _write_tensor_bytes(self._file, encoded_indices)
        _write_tensor_bytes(self._file, values)

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class StreamingDeltaTensorStore:
    def __init__(self, path: str | Path):
        self._tensors: dict[str, torch.Tensor] = {}
        self._shapes: dict[str, tuple[int, ...]] = {}
        self._version = streaming_delta_version(path)
        for record in iter_streaming_delta_records(path):
            index_key = f"{record.name}{DELTA_INDEX_SUFFIX}"
            value_key = f"{record.name}{DELTA_VALUE_SUFFIX}"
            if index_key in self._tensors:
                raise ValueError(f"duplicate streaming delta record: {record.name}")
            self._tensors[index_key] = record.encoded_indices
            self._tensors[value_key] = record.values
            if record.shape is not None:
                self._shapes[record.name] = record.shape

    def keys(self) -> Iterable[str]:
        return self._tensors.keys()

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]

    def metadata(self) -> dict[str, str]:
        if self._version == 1:
            return {DELTA_METADATA_FORMAT_KEY: DELTA_METADATA_FORMAT_V1}
        return _logical_delta_metadata(self._shapes)


def is_streaming_delta_file(path: str | Path) -> bool:
    path = Path(path)
    if not path.is_file() or path.stat().st_size < len(_STREAM_MAGIC):
        return False
    with path.open("rb") as stream:
        return stream.read(len(_STREAM_MAGIC)) == _STREAM_MAGIC


def streaming_delta_version(path: str | Path) -> int:
    with Path(path).open("rb") as stream:
        header = _read_exact(stream, _STREAM_HEADER.size, "streaming delta header")
    magic, version = _STREAM_HEADER.unpack(header)
    if magic != _STREAM_MAGIC:
        raise ValueError("streaming delta magic mismatch")
    if version not in (1, 2):
        raise ValueError(f"unsupported streaming delta version: {version}")
    return version


def iter_streaming_delta_records(path: str | Path) -> Iterator[StreamingDeltaRecord]:
    with Path(path).open("rb") as stream:
        header = _read_exact(stream, _STREAM_HEADER.size, "streaming delta header")
        magic, version = _STREAM_HEADER.unpack(header)
        if magic != _STREAM_MAGIC:
            raise ValueError("streaming delta magic mismatch")
        if version not in (1, 2):
            raise ValueError(f"unsupported streaming delta version: {version}")

        while record_header := stream.read(_STREAM_RECORD.size):
            if len(record_header) != _STREAM_RECORD.size:
                raise ValueError("truncated streaming delta record header")
            name_size, index_format, dtype_code, reserved, index_size, value_count = _STREAM_RECORD.unpack(
                record_header
            )
            if version == 1 and reserved != 0:
                raise ValueError("streaming delta v1 record has unsupported flags")
            dtype = _STREAM_CODE_TO_DTYPE.get(dtype_code)
            if dtype is None:
                raise ValueError(f"unsupported streaming delta dtype code: {dtype_code}")

            name = _read_exact(stream, name_size, "streaming delta tensor name").decode("utf-8")
            shape = None
            if version == 2:
                shape_bytes = _read_exact(stream, reserved * 8, f"streaming delta shape for {name}")
                shape = struct.unpack(f"<{reserved}Q", shape_bytes) if reserved else ()
            index_bytes = _read_exact(stream, index_size, f"streaming delta indices for {name}")
            value_size = value_count * _STREAM_DTYPE_SIZE[dtype]
            value_bytes = _read_exact(stream, value_size, f"streaming delta values for {name}")
            encoded_indices = _tensor_from_stream_bytes(index_bytes, _stream_index_dtype(index_format, index_size))
            values = _tensor_from_stream_bytes(value_bytes, dtype)
            yield StreamingDeltaRecord(
                name=name,
                shape=shape,
                encoded_indices=encoded_indices,
                values=values,
                format_version=version,
            )


def _stream_index_format(indices: torch.Tensor) -> int:
    if indices.dtype == torch.uint8:
        return _STREAM_INDEX_VARINT
    if indices.dtype == torch.int32:
        return _STREAM_INDEX_RAW32
    if indices.dtype == torch.int64:
        return _STREAM_INDEX_RAW64
    raise ValueError(f"unsupported streaming delta index dtype: {indices.dtype}")


def _stream_index_dtype(index_format: int, index_size: int) -> torch.dtype:
    if index_format == _STREAM_INDEX_VARINT:
        return torch.uint8
    if index_format == _STREAM_INDEX_RAW32:
        if index_size % 4 != 0:
            raise ValueError("streaming delta int32 index payload has invalid size")
        return torch.int32
    if index_format == _STREAM_INDEX_RAW64:
        if index_size % 8 != 0:
            raise ValueError("streaming delta int64 index payload has invalid size")
        return torch.int64
    raise ValueError(f"unsupported streaming delta index format: {index_format}")


def _read_exact(stream: BinaryIO, size: int, description: str) -> bytearray:
    data = bytearray(stream.read(size))
    if len(data) != size:
        raise ValueError(f"truncated {description}")
    return data


def _tensor_from_stream_bytes(data: bytearray, dtype: torch.dtype) -> torch.Tensor:
    if not data:
        return torch.empty((0,), dtype=dtype)
    return torch.frombuffer(data, dtype=dtype).clone()


def _write_tensor_bytes(stream: BinaryIO, tensor: torch.Tensor) -> None:
    array = tensor.view(torch.uint16).numpy() if tensor.dtype == torch.bfloat16 else tensor.numpy()
    stream.write(memoryview(array).cast("B"))


FUSED_WEIGHT_RULES = (
    FusedPartRule(".self_attn.q_proj.weight", ".self_attn.qkv_proj.weight", 0, 3),
    FusedPartRule(".self_attn.k_proj.weight", ".self_attn.qkv_proj.weight", 1, 3),
    FusedPartRule(".self_attn.v_proj.weight", ".self_attn.qkv_proj.weight", 2, 3),
    FusedPartRule(".mlp.gate_proj.weight", ".mlp.gate_up_proj.weight", 0, 2),
    FusedPartRule(".mlp.up_proj.weight", ".mlp.gate_up_proj.weight", 1, 2),
)
FUSED_BIAS_RULES = (
    FusedPartRule(".self_attn.q_proj.bias", ".self_attn.qkv_proj.bias", 0, 3),
    FusedPartRule(".self_attn.k_proj.bias", ".self_attn.qkv_proj.bias", 1, 3),
    FusedPartRule(".self_attn.v_proj.bias", ".self_attn.qkv_proj.bias", 2, 3),
    FusedPartRule(".mlp.gate_proj.bias", ".mlp.gate_up_proj.bias", 0, 2),
    FusedPartRule(".mlp.up_proj.bias", ".mlp.gate_up_proj.bias", 1, 2),
)


@nb.njit(cache=False)
def _encode_varint_delta_indices_numba(indices: np.ndarray) -> np.ndarray:
    count = indices.size
    encoded_size = 0
    previous = 0
    for i in range(count):
        current = indices[i]
        if current < 0:
            raise ValueError("sparse delta indices must be non-negative")
        if current < previous:
            raise ValueError("varint delta encoding expects sorted indices")
        diff = current - previous
        previous = current
        encoded_size += 1
        while diff >= 0x80:
            diff >>= 7
            encoded_size += 1

    encoded = np.empty((encoded_size,), dtype=np.uint8)
    output_index = 0
    previous = 0
    for i in range(count):
        current = indices[i]
        diff = current - previous
        previous = current
        while diff >= 0x80:
            encoded[output_index] = (diff & 0x7F) | 0x80
            output_index += 1
            diff >>= 7
        encoded[output_index] = diff
        output_index += 1
    return encoded


def encode_varint_delta_indices(indices: torch.Tensor) -> torch.Tensor:
    indices = indices.reshape(-1).detach().to(torch.int64).cpu()
    if indices.numel() == 0:
        return torch.empty((0,), dtype=torch.uint8)

    encoded = _encode_varint_delta_indices_numba(indices.contiguous().numpy())
    return torch.from_numpy(encoded)


def decode_varint_delta_indices(index_bytes: torch.Tensor, expected_count: int) -> torch.Tensor:
    if index_bytes.dtype != torch.uint8:
        raise ValueError("varint delta indices must be stored as uint8")
    if expected_count == 0:
        if index_bytes.numel() != 0:
            raise ValueError("expected no indices, but varint stream is not empty")
        return torch.empty((0,), dtype=torch.int64)
    index_bytes = index_bytes.detach().cpu().reshape(-1)
    if index_bytes.numel() == expected_count and not bool((index_bytes & 0x80).any().item()):
        return index_bytes.to(torch.int64).cumsum(dim=0)

    return _decode_varint_delta_indices_numpy(index_bytes, expected_count)


def _decode_varint_delta_indices_numpy(index_bytes: torch.Tensor, expected_count: int) -> torch.Tensor:
    encoded = index_bytes.numpy()
    is_terminal = (encoded & 0x80) == 0
    end_positions = np.flatnonzero(is_terminal)

    if end_positions.size != expected_count:
        if encoded.size > 0 and not bool(is_terminal[-1]):
            raise ValueError("truncated varint stream")
        if end_positions.size > expected_count:
            raise ValueError("decoded more indices than expected")
        raise ValueError(f"decoded index count mismatch: {end_positions.size} vs {expected_count}")

    start_positions = np.empty_like(end_positions)
    start_positions[0] = 0
    start_positions[1:] = end_positions[:-1] + 1
    lengths = end_positions - start_positions + 1
    max_length = int(lengths.max(initial=0))
    if max_length > 9:
        raise ValueError("varint stream contains an index delta that does not fit int64")

    payload = encoded & 0x7F
    diffs = np.zeros(expected_count, dtype=np.int64)
    for offset in range(max_length):
        active = lengths > offset
        byte_positions = start_positions[active] + offset
        diffs[active] |= payload[byte_positions].astype(np.int64) << (7 * offset)

    indices = np.cumsum(diffs, dtype=np.int64)
    return torch.from_numpy(indices)


def _decode_varint_delta_indices_python(index_bytes: torch.Tensor, expected_count: int) -> torch.Tensor:
    if expected_count == 0:
        if index_bytes.numel() != 0:
            raise ValueError("expected no indices, but varint stream is not empty")
        return torch.empty((0,), dtype=torch.int64)

    out = torch.empty((expected_count,), dtype=torch.int64)
    out_index = 0
    current = 0
    diff = 0
    shift = 0

    for byte in memoryview(index_bytes.numpy()):
        diff |= (int(byte) & 0x7F) << shift
        if byte & 0x80:
            shift += 7
            continue

        if out_index >= expected_count:
            raise ValueError("decoded more indices than expected")
        current += diff
        out[out_index] = current
        out_index += 1
        diff = 0
        shift = 0

    if shift != 0:
        raise ValueError("truncated varint stream")
    if out_index != expected_count:
        raise ValueError(f"decoded index count mismatch: {out_index} vs {expected_count}")
    return out


def encode_sparse_indices(indices: torch.Tensor, *, encoding: str = "optimized") -> torch.Tensor:
    indices = indices.reshape(-1).detach().to(torch.int64).cpu()
    if encoding == "optimized":
        return encode_varint_delta_indices(indices)
    if encoding == "naive":
        if indices.numel() == 0:
            return indices.to(torch.int32)
        return indices.to(torch.int32 if int(indices.max().item()) < 2**31 else torch.int64)
    raise ValueError(f"unsupported sparse index encoding: {encoding}")


def decode_sparse_indices(encoded_indices: torch.Tensor, value_count: int) -> torch.Tensor:
    if encoded_indices.dtype == torch.uint8:
        return decode_varint_delta_indices(encoded_indices, value_count)
    if encoded_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"unsupported sparse index dtype: {encoded_indices.dtype}")
    indices = encoded_indices.reshape(-1).to(torch.int64)
    if indices.numel() != value_count:
        raise ValueError(f"index/value length mismatch: {indices.numel()} vs {value_count}")
    return indices


def map_tensor_name_to_fused(name: str, *, include_bias: bool = False) -> MappedTensorName:
    for rule in FUSED_WEIGHT_RULES:
        if name.endswith(rule.source_suffix):
            return MappedTensorName(
                name=name.replace(rule.source_suffix, rule.fused_suffix),
                part_order=rule.order,
                parts=rule.parts,
            )
    if include_bias:
        for rule in FUSED_BIAS_RULES:
            if name.endswith(rule.source_suffix):
                return MappedTensorName(
                    name=name.replace(rule.source_suffix, rule.fused_suffix),
                    part_order=rule.order,
                    parts=rule.parts,
                )
    return MappedTensorName(name=name, part_order=None, parts=None)


def collect_tied_lm_head_skip_keys(store: TensorStore) -> set[str]:
    keys = set(store.keys())
    if "model.embed_tokens.weight" not in keys or "lm_head.weight" not in keys:
        return set()

    embedding = store.get_tensor("model.embed_tokens.weight")
    lm_head = store.get_tensor("lm_head.weight")
    if embedding.shape == lm_head.shape and torch.equal(embedding, lm_head):
        return {"lm_head.weight"}
    return set()


def build_fused_groups(
    keys: Iterable[str],
    *,
    include_bias: bool = False,
) -> tuple[dict[str, dict[int, str]], dict[str, int], set[str]]:
    fused_parts: dict[str, dict[int, str]] = defaultdict(dict)
    parts_expected: dict[str, int] = {}
    single_keys: set[str] = set()

    for key in keys:
        mapped = map_tensor_name_to_fused(key, include_bias=include_bias)
        if mapped.part_order is None:
            single_keys.add(key)
            continue

        if mapped.part_order in fused_parts[mapped.name]:
            raise ValueError(f"duplicate fused part {mapped.part_order} for {mapped.name}")
        fused_parts[mapped.name][mapped.part_order] = key
        parts_expected.setdefault(mapped.name, mapped.parts or 0)
        if parts_expected[mapped.name] != mapped.parts:
            raise ValueError(f"parts mismatch for {mapped.name}: {parts_expected[mapped.name]} vs {mapped.parts}")

    for fused_name, parts in list(fused_parts.items()):
        _validate_part_orders(fused_name, parts.keys(), parts_expected[fused_name])
        single_keys.discard(fused_name)

    return dict(fused_parts), parts_expected, single_keys


def load_fused_tensor(store: TensorStore, fused_name: str, parts: Mapping[int, str]) -> torch.Tensor:
    _validate_part_orders(fused_name, parts.keys(), len(parts))
    tensors = [store.get_tensor(parts[order]) for order in sorted(parts)]
    _validate_fused_shapes(fused_name, {order: tensor.shape for order, tensor in zip(sorted(parts), tensors)})
    return torch.cat(tensors, dim=0)


def reconstruct_sparse_delta_tensor(
    base_tensor: torch.Tensor,
    encoded_indices: torch.Tensor,
    values: torch.Tensor,
    *,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    indices = decode_sparse_indices(encoded_indices, values.numel())
    flat = base_tensor.reshape(-1).clone()
    if indices.numel() == 0:
        return flat.reshape_as(base_tensor)
    if int(indices.min().item()) < 0 or int(indices.max().item()) >= flat.numel():
        raise ValueError("sparse delta index out of range")

    indices = indices.to(device=flat.device)
    delta_values = values.reshape(-1).to(device=flat.device)
    if scale_factor != 1.0:
        delta_values = delta_values * scale_factor
    updated = flat.index_select(0, indices).to(torch.promote_types(flat.dtype, delta_values.dtype)) + delta_values
    flat.index_copy_(0, indices, updated.to(flat.dtype))
    return flat.reshape_as(base_tensor)


def verify_sparse_delta_file(
    base_model_path: str | Path,
    target_model_path: str | Path,
    delta_path: str | Path,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    max_report: int = 10,
    include_bias: bool = True,
) -> DeltaVerificationResult:
    with (
        safe_open(base_model_path, framework="pt", device="cpu") as base,
        safe_open(target_model_path, framework="pt", device="cpu") as target,
    ):
        if is_streaming_delta_file(delta_path):
            return verify_sparse_delta_stores(
                base,
                target,
                StreamingDeltaTensorStore(delta_path),
                atol=atol,
                rtol=rtol,
                max_report=max_report,
                include_bias=include_bias,
            )
        with safe_open(delta_path, framework="pt", device="cpu") as delta:
            return verify_sparse_delta_stores(
                base,
                target,
                delta,
                atol=atol,
                rtol=rtol,
                max_report=max_report,
                include_bias=include_bias,
            )


def _verify_sparse_delta_state_stores(
    base: TensorStore,
    target: TensorStore,
    delta_path: str | Path,
    *,
    atol: float,
    rtol: float,
    max_report: int,
    include_bias: bool,
) -> DeltaVerificationResult:
    if is_streaming_delta_file(delta_path):
        return verify_sparse_delta_stores(
            base,
            target,
            StreamingDeltaTensorStore(delta_path),
            atol=atol,
            rtol=rtol,
            max_report=max_report,
            include_bias=include_bias,
        )
    with safe_open(delta_path, framework="pt", device="cpu") as delta:
        return verify_sparse_delta_stores(
            base,
            target,
            delta,
            atol=atol,
            rtol=rtol,
            max_report=max_report,
            include_bias=include_bias,
        )


def verify_sparse_delta_state_dicts(
    base_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    delta_path: str | Path,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    max_report: int = 10,
    include_bias: bool = True,
) -> DeltaVerificationResult:
    base = StateDictTensorStore(base_state)
    target = StateDictTensorStore(target_state)
    return _verify_sparse_delta_state_stores(
        base,
        target,
        delta_path,
        atol=atol,
        rtol=rtol,
        max_report=max_report,
        include_bias=include_bias,
    )


def verify_sparse_delta_stores(
    base: TensorStore,
    target: TensorStore,
    delta: TensorStore,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    max_report: int = 10,
    include_bias: bool = True,
) -> DeltaVerificationResult:
    base_keys = set(base.keys())
    target_keys = set(target.keys())
    delta_keys = set(delta.keys())

    if _looks_like_delta(base_keys):
        raise ValueError("--base looks like a sparse delta file")
    if _looks_like_delta(target_keys):
        raise ValueError("--target looks like a sparse delta file")
    if not _looks_like_delta(delta_keys) and not is_sparse_delta_store(delta):
        raise ValueError("--delta does not look like a sparse delta file")

    model_keys = _validated_model_keys(base, target, include_bias=include_bias)
    delta_format = sparse_delta_format(delta)
    if delta_format not in (None, DELTA_METADATA_FORMAT_V1, DELTA_METADATA_FORMAT_V2):
        raise ValueError(f"unsupported sparse delta format: {delta_format}")
    logical_layout = delta_format == DELTA_METADATA_FORMAT_V2
    if logical_layout and sparse_delta_layout(delta) != DELTA_LAYOUT_LOGICAL:
        raise ValueError(f"unsupported logical sparse delta layout: {sparse_delta_layout(delta)}")
    if logical_layout:
        fused_parts: dict[str, dict[int, str]] = {}
        all_names = set(model_keys)
    else:
        fused_parts, _parts_expected, single_keys = build_fused_groups(model_keys, include_bias=include_bias)
        all_names = set(single_keys) | set(fused_parts)

    idx_names = {key[: -len(DELTA_INDEX_SUFFIX)] for key in delta_keys if key.endswith(DELTA_INDEX_SUFFIX)}
    val_names = {key[: -len(DELTA_VALUE_SUFFIX)] for key in delta_keys if key.endswith(DELTA_VALUE_SUFFIX)}
    if idx_names != val_names:
        raise ValueError(
            f"delta index/value names mismatch: idx-only={idx_names - val_names}, val-only={val_names - idx_names}"
        )
    if unexpected := idx_names - all_names:
        raise ValueError(f"delta has names not present in the model: {sorted(unexpected)[:10]}")
    shapes = sparse_delta_shapes(delta) if logical_layout else {}
    if logical_layout and set(shapes) != idx_names:
        raise ValueError(
            f"logical delta shape/name mismatch: missing={idx_names - set(shapes)}, unexpected={set(shapes) - idx_names}"
        )

    max_diff = 0.0
    mismatches: list[DeltaMismatch] = []
    report_limit = max(max_report, 1)

    for name in sorted(all_names):
        if name in fused_parts:
            base_tensor = load_fused_tensor(base, name, fused_parts[name])
            target_tensor = load_fused_tensor(target, name, fused_parts[name])
        else:
            base_tensor = base.get_tensor(name)
            target_tensor = target.get_tensor(name)

        if base_tensor.shape != target_tensor.shape:
            raise ValueError(f"shape mismatch for {name}: {tuple(base_tensor.shape)} vs {tuple(target_tensor.shape)}")

        idx_key = f"{name}{DELTA_INDEX_SUFFIX}"
        val_key = f"{name}{DELTA_VALUE_SUFFIX}"
        if name in idx_names:
            if logical_layout and tuple(base_tensor.shape) != shapes[name]:
                raise ValueError(
                    f"logical delta shape mismatch for {name}: metadata={shapes[name]} model={tuple(base_tensor.shape)}"
                )
            reconstructed = reconstruct_sparse_delta_tensor(
                base_tensor,
                delta.get_tensor(idx_key),
                delta.get_tensor(val_key),
            )
        else:
            reconstructed = base_tensor

        diff = (reconstructed - target_tensor).abs()
        diff_max = float(diff.max().item()) if diff.numel() else 0.0
        max_diff = max(max_diff, diff_max)
        if not torch.allclose(reconstructed, target_tensor, atol=atol, rtol=rtol) and len(mismatches) < report_limit:
            mismatches.append(DeltaMismatch(name=name, max_diff=diff_max))

    return DeltaVerificationResult(max_diff=max_diff, mismatches=tuple(mismatches))


def count_sparse_delta_values(delta_path: str | Path) -> int:
    if is_streaming_delta_file(delta_path):
        return sum(record.values.numel() for record in iter_streaming_delta_records(delta_path))
    total = 0
    with safe_open(delta_path, framework="pt", device="cpu") as delta:
        for key in delta.keys():
            if key.endswith(DELTA_VALUE_SUFFIX):
                total += delta.get_tensor(key).numel()
    return total


class ModelDeltaManager:
    def extract_sparse_delta(
        self,
        base_model_path: str | Path,
        finetuned_model_path: str | Path,
        delta_output_path: str | Path,
        *,
        index_encoding: str = "optimized",
        save_stats: bool = False,
    ) -> DeltaStats | None:
        with (
            safe_open(base_model_path, framework="pt", device="cpu") as base,
            safe_open(finetuned_model_path, framework="pt", device="cpu") as target,
        ):
            return self._extract_sparse_delta_from_stores(
                base,
                target,
                delta_output_path,
                index_encoding=index_encoding,
                include_bias=True,
                threshold=0.0,
                collect_stats=save_stats,
            )

    def extract_sparse_delta_from_state_dicts(
        self,
        base_state: Mapping[str, torch.Tensor],
        finetuned_state: Mapping[str, torch.Tensor],
        delta_output_path: str | Path,
        *,
        index_encoding: str = "optimized",
        save_stats: bool = False,
    ) -> DeltaStats | None:
        return self._extract_sparse_delta_from_stores(
            StateDictTensorStore(base_state),
            StateDictTensorStore(finetuned_state),
            delta_output_path,
            index_encoding=index_encoding,
            include_bias=True,
            threshold=0.0,
            collect_stats=save_stats,
        )

    def extract_sparse_delta_streaming(
        self,
        base_model_path: str | Path,
        finetuned_model_path: str | Path,
        delta_output_path: str | Path,
        *,
        group_size: int = 4,
        index_encoding: str = "optimized",
        save_stats: bool = False,
    ) -> DeltaStats | None:
        with (
            safe_open(base_model_path, framework="pt", device="cpu") as base,
            safe_open(finetuned_model_path, framework="pt", device="cpu") as target,
        ):
            return self._extract_sparse_delta_streaming_from_stores(
                base,
                target,
                delta_output_path,
                group_size=group_size,
                index_encoding=index_encoding,
                collect_stats=save_stats,
            )

    def extract_sparse_delta_streaming_from_state_dicts(
        self,
        base_state: Mapping[str, torch.Tensor],
        finetuned_state: Mapping[str, torch.Tensor],
        delta_output_path: str | Path,
        *,
        group_size: int = 4,
        index_encoding: str = "optimized",
        save_stats: bool = False,
    ) -> DeltaStats | None:
        return self._extract_sparse_delta_streaming_from_stores(
            StateDictTensorStore(base_state),
            StateDictTensorStore(finetuned_state),
            delta_output_path,
            group_size=group_size,
            index_encoding=index_encoding,
            collect_stats=save_stats,
        )

    def extract_delta_fused_sparse(
        self,
        base_model_path: str | Path,
        finetuned_model_path: str | Path,
        delta_output_path: str | Path,
        threshold: float = 0.0,
        include_bias: bool = False,
        save_stats: bool = False,
        save_delta_values: bool = False,
        plot_hist: bool = False,
        index_encoding: str = "optimized",
    ) -> dict[str, bool | float | int]:
        if save_delta_values or plot_hist:
            raise ValueError("delta value dumps and histograms are not part of the Phase 1 utility layer")

        with (
            safe_open(base_model_path, framework="pt", device="cpu") as base,
            safe_open(finetuned_model_path, framework="pt", device="cpu") as target,
        ):
            stats = self._extract_sparse_delta_from_stores(
                base,
                target,
                delta_output_path,
                index_encoding=index_encoding,
                include_bias=include_bias,
                threshold=threshold,
                collect_stats=save_stats,
            )

        summary: dict[str, bool | float | int] = {
            "threshold": float(threshold),
            "include_bias": bool(include_bias),
        }
        if stats is not None:
            summary.update(asdict(stats))
            stats_path = _stats_path(delta_output_path)
            stats_path.write_text(json.dumps({"summary": summary}, indent=2), encoding="utf-8")
        return summary

    def _extract_sparse_delta_from_stores(
        self,
        base: TensorStore,
        target: TensorStore,
        delta_output_path: str | Path,
        *,
        index_encoding: str,
        include_bias: bool,
        threshold: float,
        collect_stats: bool,
    ) -> DeltaStats | None:
        keys = _validated_model_keys(base, target, include_bias=include_bias)

        total_params = 0
        total_bytes = 0
        changed_params = 0
        delta_tensors: dict[str, torch.Tensor] = {}
        shapes: dict[str, tuple[int, ...]] = {}

        for key in sorted(keys):
            base_tensor = base.get_tensor(key)
            target_tensor = target.get_tensor(key)
            _validate_tensor_pair(key, base_tensor, target_tensor)

            if collect_stats:
                total_params += base_tensor.numel()
                total_bytes += base_tensor.numel() * base_tensor.element_size()

            indices, values = _extract_changed_values(base_tensor, target_tensor, threshold=threshold)
            if indices.numel() == 0:
                continue

            changed_params += int(indices.numel())
            shapes[key] = tuple(base_tensor.shape)
            _add_sparse_tensor(delta_tensors, key, [indices], [values], index_encoding=index_encoding)

        save_file(delta_tensors, delta_output_path, metadata=_logical_delta_metadata(shapes))
        if not collect_stats:
            return None

        delta_size_mb = os.path.getsize(delta_output_path) / (1024 * 1024)
        total_size_mb = total_bytes / (1024 * 1024)
        change_percentage = (changed_params / total_params * 100) if total_params else 0.0
        compression_ratio = total_size_mb / delta_size_mb if delta_size_mb > 0 else float("inf")
        return DeltaStats(
            total_params=total_params,
            changed_params=changed_params,
            change_percentage=change_percentage,
            total_size_mb=total_size_mb,
            delta_size_mb=delta_size_mb,
            compression_ratio=compression_ratio,
        )

    def _extract_sparse_delta_streaming_from_stores(
        self,
        base: TensorStore,
        target: TensorStore,
        delta_output_path: str | Path,
        *,
        group_size: int,
        index_encoding: str,
        collect_stats: bool,
    ) -> DeltaStats | None:
        if group_size < 0:
            raise ValueError("streaming delta group_size must be non-negative")

        keys = _validated_model_keys(base, target, include_bias=True)
        record_names = sorted(keys, key=_stream_record_order)
        total_params = 0
        total_bytes = 0
        changed_params = 0

        writer = StreamingDeltaWriter(delta_output_path)
        try:
            for record_index, name in enumerate(record_names):
                base_tensor = base.get_tensor(name)
                target_tensor = target.get_tensor(name)
                _validate_tensor_pair(name, base_tensor, target_tensor)
                total_params += base_tensor.numel()
                total_bytes += base_tensor.numel() * base_tensor.element_size()
                indices, values = _extract_changed_values(base_tensor, target_tensor)
                if indices.numel() > 0:
                    changed_params += indices.numel()
                    writer.write_record(
                        name,
                        base_tensor.shape,
                        indices,
                        values,
                        index_encoding=index_encoding,
                    )

                current_group = _stream_group_id(name, group_size)
                next_group = (
                    _stream_group_id(record_names[record_index + 1], group_size)
                    if record_index + 1 < len(record_names)
                    else None
                )
                if group_size == 0 or current_group is None or current_group != next_group:
                    writer.flush()
        finally:
            writer.close()

        if not collect_stats:
            return None
        delta_size_mb = os.path.getsize(delta_output_path) / (1024 * 1024)
        total_size_mb = total_bytes / (1024 * 1024)
        change_percentage = (changed_params / total_params * 100) if total_params else 0.0
        compression_ratio = total_size_mb / delta_size_mb if delta_size_mb > 0 else float("inf")
        return DeltaStats(
            total_params=total_params,
            changed_params=changed_params,
            change_percentage=change_percentage,
            total_size_mb=total_size_mb,
            delta_size_mb=delta_size_mb,
            compression_ratio=compression_ratio,
        )


def _add_sparse_tensor(
    delta_tensors: dict[str, torch.Tensor],
    name: str,
    index_chunks: list[torch.Tensor],
    value_chunks: list[torch.Tensor],
    *,
    index_encoding: str,
) -> None:
    if len(index_chunks) != len(value_chunks):
        raise ValueError(f"index/value chunks mismatch for {name}")

    indices = (index_chunks[0] if len(index_chunks) == 1 else torch.cat(index_chunks, dim=0)).to(torch.int64)
    values = value_chunks[0] if len(value_chunks) == 1 else torch.cat(value_chunks, dim=0)

    delta_tensors[f"{name}{DELTA_INDEX_SUFFIX}"] = encode_sparse_indices(indices, encoding=index_encoding)
    delta_tensors[f"{name}{DELTA_VALUE_SUFFIX}"] = values


def _validated_model_keys(base: TensorStore, target: TensorStore, *, include_bias: bool) -> set[str]:
    base_keys = set(base.keys())
    target_keys = set(target.keys())
    skip_keys = collect_tied_lm_head_skip_keys(base) & collect_tied_lm_head_skip_keys(target)
    base_keys -= skip_keys
    target_keys -= skip_keys
    if not include_bias:
        base_keys = {key for key in base_keys if not key.endswith(".bias")}
        target_keys = {key for key in target_keys if not key.endswith(".bias")}
    if base_keys != target_keys:
        raise ValueError(f"keys mismatch: base-only={base_keys - target_keys}, target-only={target_keys - base_keys}")
    return base_keys


def _validate_tensor_pair(name: str, base_tensor: torch.Tensor, target_tensor: torch.Tensor) -> None:
    if base_tensor.shape != target_tensor.shape:
        raise ValueError(f"shape mismatch for {name}: {tuple(base_tensor.shape)} vs {tuple(target_tensor.shape)}")
    if base_tensor.dtype != target_tensor.dtype:
        raise ValueError(f"dtype mismatch for {name}: {base_tensor.dtype} vs {target_tensor.dtype}")


def _extract_changed_values(
    base_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    *,
    threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat_base = base_tensor.reshape(-1)
    flat_target = target_tensor.reshape(-1)
    if threshold == 0.0:
        indices = torch.nonzero(flat_target != flat_base, as_tuple=False).flatten()
    else:
        delta = flat_target - flat_base
        indices = torch.nonzero(delta.abs() > threshold, as_tuple=False).flatten()
    selected_base = flat_base.index_select(0, indices)
    selected_target = flat_target.index_select(0, indices)
    values = selected_target - selected_base
    if not torch.equal(selected_base + values, selected_target):
        # Widen only tensors whose additive deltas would otherwise lose target bits.
        wider_dtypes = (
            (torch.float32, torch.float64) if base_tensor.dtype in (torch.float16, torch.bfloat16) else (torch.float64,)
        )
        for dtype in wider_dtypes:
            values = selected_target.to(dtype) - selected_base.to(dtype)
            reconstructed = (selected_base.to(dtype) + values).to(base_tensor.dtype)
            if torch.equal(reconstructed, selected_target):
                break
        else:
            mismatch = torch.nonzero(reconstructed != selected_target, as_tuple=False)[0, 0].item()
            raise ValueError(
                "weight update cannot be represented exactly as an additive sparse delta "
                f"(dtype={base_tensor.dtype}, base={selected_base[mismatch].item()}, "
                f"target={selected_target[mismatch].item()})"
            )
    return indices, values.contiguous()


def _stream_layer_index(name: str) -> int | None:
    marker = ".layers."
    if marker not in name:
        return None
    layer = name.split(marker, 1)[1].split(".", 1)[0]
    return int(layer) if layer.isdecimal() else None


def _stream_record_order(name: str) -> tuple[int, int, str]:
    layer_index = _stream_layer_index(name)
    if layer_index is None:
        return (0, 0, name)
    return (1, layer_index, name)


def _stream_group_id(name: str, group_size: int) -> int | None:
    layer_index = _stream_layer_index(name)
    if layer_index is None or group_size == 0:
        return None
    return layer_index // group_size


def _validate_part_orders(name: str, orders: Iterable[int], expected: int) -> None:
    actual = set(orders)
    wanted = set(range(expected))
    if actual != wanted:
        raise ValueError(f"missing fused parts for {name}: got {sorted(actual)} expected {sorted(wanted)}")


def _validate_fused_shapes(name: str, part_shapes: Mapping[int, torch.Size]) -> None:
    sorted_shapes = [part_shapes[order] for order in sorted(part_shapes)]
    if not sorted_shapes:
        return
    rank = len(sorted_shapes[0])
    trailing = tuple(sorted_shapes[0][1:])
    for shape in sorted_shapes:
        if len(shape) != rank or tuple(shape[1:]) != trailing:
            raise ValueError(f"incompatible fused part shapes for {name}: {[tuple(shape) for shape in sorted_shapes]}")


def _fused_offsets(part_sizes: Mapping[int, int], expected: int) -> dict[int, int]:
    _validate_part_orders("fused tensor", part_sizes.keys(), expected)
    offsets: dict[int, int] = {}
    running = 0
    for order in range(expected):
        offsets[order] = running
        running += part_sizes[order]
    return offsets


def _looks_like_delta(keys: Iterable[str]) -> bool:
    return any(key.endswith(DELTA_INDEX_SUFFIX) or key.endswith(DELTA_VALUE_SUFFIX) for key in keys)


def sparse_delta_format(store: TensorStore) -> str | None:
    return _tensor_store_metadata(store).get(DELTA_METADATA_FORMAT_KEY)


def sparse_delta_layout(store: TensorStore) -> str | None:
    return _tensor_store_metadata(store).get(DELTA_METADATA_LAYOUT_KEY)


def sparse_delta_shapes(store: TensorStore) -> dict[str, tuple[int, ...]]:
    metadata = _tensor_store_metadata(store)
    encoded = metadata.get(DELTA_METADATA_SHAPES_KEY)
    if encoded is None:
        return {}
    raw_shapes = json.loads(encoded)
    if not isinstance(raw_shapes, dict):
        raise ValueError("logical sparse delta shapes metadata must be an object")
    shapes: dict[str, tuple[int, ...]] = {}
    for name, shape in raw_shapes.items():
        if not isinstance(name, str) or not isinstance(shape, list) or not all(isinstance(dim, int) for dim in shape):
            raise ValueError("logical sparse delta shapes metadata is malformed")
        if any(dim < 0 for dim in shape):
            raise ValueError(f"logical sparse delta shape has a negative dimension: {name}")
        shapes[name] = tuple(shape)
    return shapes


def _logical_delta_metadata(shapes: Mapping[str, tuple[int, ...]]) -> dict[str, str]:
    return {
        DELTA_METADATA_FORMAT_KEY: DELTA_METADATA_FORMAT_V2,
        DELTA_METADATA_LAYOUT_KEY: DELTA_LAYOUT_LOGICAL,
        DELTA_METADATA_SHAPES_KEY: json.dumps({name: list(shape) for name, shape in sorted(shapes.items())}),
    }


def is_sparse_delta_store(store: TensorStore) -> bool:
    if _looks_like_delta(store.keys()):
        return True
    return sparse_delta_format(store) in (DELTA_METADATA_FORMAT_V1, DELTA_METADATA_FORMAT_V2)


def _tensor_store_metadata(store: TensorStore) -> dict[str, str]:
    metadata = getattr(store, "metadata", None)
    if not callable(metadata):
        return {}
    return metadata() or {}


def _stats_path(delta_output_path: str | Path) -> Path:
    path = Path(delta_output_path)
    suffix = "".join(path.suffixes)
    if suffix:
        return path.with_name(f"{path.name.removesuffix(suffix)}_stats.json")
    return path.with_name(f"{path.name}_stats.json")
