from __future__ import annotations

from typing import Any, Mapping, cast

import msgspec

INT16_BYTES = 2


class RoutedExperts(msgspec.Struct, gc=False, omit_defaults=True):
    shape: list[int]
    data: bytes
    dtype: str = "int16"


def _shape_numel(shape: list[int]) -> int:
    seq_len, num_layers, topk = shape
    return seq_len * num_layers * topk


def _token_stride(shape: list[int]) -> int:
    return shape[1] * shape[2] * INT16_BYTES


def validate_routed_experts(payload: RoutedExperts) -> RoutedExperts:
    assert payload.dtype == "int16"
    assert len(payload.shape) == 3
    assert len(payload.data) == _shape_numel(payload.shape) * INT16_BYTES
    return payload


def routed_experts_from_raw(raw: Any) -> RoutedExperts:
    if isinstance(raw, RoutedExperts):
        return validate_routed_experts(raw)
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump(mode="python")
    raw = cast(Mapping[str, Any], raw)
    return validate_routed_experts(
        RoutedExperts(
            dtype=raw["dtype"],
            shape=[int(dim) for dim in raw["shape"]],
            data=raw["data"],
        )
    )


def routed_experts_len(payload: RoutedExperts) -> int:
    return validate_routed_experts(payload).shape[0]


def align_routed_experts(payload: RoutedExperts | None, expected_len: int) -> RoutedExperts | None:
    if payload is None:
        return None
    payload = validate_routed_experts(payload)
    deficit = expected_len - payload.shape[0]
    assert deficit >= 0
    assert deficit <= 1
    if deficit == 0:
        return payload
    return append_zero_tokens(payload, deficit)


def slice_routed_experts(payload: RoutedExperts, end: int) -> RoutedExperts:
    payload = validate_routed_experts(payload)
    assert 0 <= end <= payload.shape[0]
    stride = _token_stride(payload.shape)
    return RoutedExperts(
        dtype=payload.dtype,
        shape=[end, payload.shape[1], payload.shape[2]],
        data=payload.data[: end * stride],
    )


def append_zero_tokens(payload: RoutedExperts, count: int) -> RoutedExperts:
    payload = validate_routed_experts(payload)
    assert count >= 0
    if count == 0:
        return payload
    stride = _token_stride(payload.shape)
    return RoutedExperts(
        dtype=payload.dtype,
        shape=[payload.shape[0] + count, payload.shape[1], payload.shape[2]],
        data=payload.data + (b"\0" * stride * count),
    )


def concat_routed_experts(left: RoutedExperts, right: RoutedExperts) -> RoutedExperts:
    left = validate_routed_experts(left)
    right = validate_routed_experts(right)
    assert left.dtype == right.dtype
    assert left.shape[1:] == right.shape[1:]
    return RoutedExperts(
        dtype=left.dtype,
        shape=[left.shape[0] + right.shape[0], left.shape[1], left.shape[2]],
        data=left.data + right.data,
    )


def extend_routed_experts(base: RoutedExperts, step: RoutedExperts, prefix_len: int) -> RoutedExperts:
    base = validate_routed_experts(base)
    step = validate_routed_experts(step)
    assert base.dtype == step.dtype
    assert base.shape[1:] == step.shape[1:]
    assert 0 <= prefix_len <= step.shape[0]

    stride = _token_stride(base.shape)
    data = bytearray(base.data)
    if prefix_len > 0:
        start = (prefix_len - 1) * stride
        data[start : start + stride] = step.data[start : start + stride]
    data.extend(step.data[prefix_len * stride :])

    return validate_routed_experts(
        RoutedExperts(
            dtype=base.dtype,
            shape=[base.shape[0] + step.shape[0] - prefix_len, base.shape[1], base.shape[2]],
            data=bytes(data),
        )
    )
