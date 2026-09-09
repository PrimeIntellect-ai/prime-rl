"""Paired routing byte transforms, independent of the trainer/GPU runtime.

Weights v1 are the Qwen3 captured FP32 coefficients (route scale 1), not logits.
The arrays returned by validation are read-only views into immutable payload bytes.
Token-causality validity is checked at sample/loader boundaries, not by this codec.
"""

import math
from collections.abc import Sequence

import numpy as np

from prime_rl.transports.batch.types import RoutedExperts


def validate_routed_experts(
    routing: RoutedExperts,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Validate structure and values; return IDs, optional coefficients and validity.

    Legacy integer dtypes remain accepted. Full mode requires compact little-endian
    IDs. The actual model's expert/layer/top-k bounds must also be checked by the
    trainer, which has the model configuration.
    """
    shape = routing.shape
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or any(type(n) is not int for n in shape)
        or shape[0] < 0
        or shape[1] <= 0
        or shape[2] <= 0
    ):
        raise ValueError("routing shape must be [tokens >= 0, layers > 0, top_k > 0]")
    try:
        dtype = np.dtype(routing.dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid routing ID dtype") from exc
    if not np.issubdtype(dtype, np.integer):
        raise ValueError("routing IDs must have an integer dtype")
    count = math.prod(shape)
    if not isinstance(routing.data, bytes) or len(routing.data) != count * dtype.itemsize:
        raise ValueError("routing ID byte count does not match shape/dtype")
    ids = np.frombuffer(routing.data, dtype=dtype).reshape(shape)
    if np.issubdtype(dtype, np.signedinteger) and ids.size and ids.min() < 0:
        raise ValueError("routing IDs must be nonnegative")
    if routing.weights is None:
        if type(routing.format_version) is not int or routing.format_version != 0 or routing.valid is not None:
            raise ValueError("legacy routing must have format_version=0 without coefficient validity")
        return ids, None, None
    if type(routing.format_version) is not int or routing.format_version != 1:
        raise ValueError("routing coefficients require format_version=1")
    if dtype not in (np.dtype("u1"), np.dtype("<u2")):
        raise ValueError("routing v1 IDs must be little-endian uint8/uint16")
    if not isinstance(routing.weights, bytes) or len(routing.weights) != count * 4:
        raise ValueError("routing coefficient byte count does not match FP32 shape")
    if not isinstance(routing.valid, bytes) or len(routing.valid) != shape[0]:
        raise ValueError("routing validity must have one byte per token")
    valid_bytes = np.frombuffer(routing.valid, dtype=np.uint8)
    if np.any(valid_bytes > 1):
        raise ValueError("routing validity bytes must be 0 or 1")
    valid = valid_bytes.view(np.bool_)
    weights = np.frombuffer(routing.weights, dtype="<f4").reshape(shape)
    # NaNs propagate through min/max, so these scalar bounds also reject NaN
    # and infinities without allocating coefficient-sized boolean temporaries.
    if not (weights.min(initial=0) >= 0 and weights.max(initial=0) <= 1):
        raise ValueError("routing v1 coefficients must be finite and in [0, 1]")
    if np.any(weights[~valid] != 0):
        raise ValueError("invalid routing rows must have zero placeholder coefficients")
    return ids, weights, valid


def routing_compatible(left: RoutedExperts, right: RoutedExperts) -> bool:
    """Whether the two routing records can share one packed microbatch."""
    return (
        left.dtype == right.dtype
        and left.shape[1:] == right.shape[1:]
        and left.format_version == right.format_version
        and (left.weights is None) == (right.weights is None)
        and (left.valid is None) == (right.valid is None)
    )


def copy_routed_experts(routing: RoutedExperts) -> RoutedExperts:
    """Own mutable shape metadata; reuse immutable byte storage."""
    return RoutedExperts(
        data=routing.data,
        shape=list(routing.shape),
        dtype=routing.dtype,
        weights=routing.weights,
        valid=routing.valid,
        format_version=routing.format_version,
    )


def slice_routed_experts(routing: RoutedExperts, start: int, stop: int) -> RoutedExperts:
    if not 0 <= start <= stop <= routing.shape[0]:
        raise ValueError("routing slice is out of token range")
    layer_slots = routing.shape[1] * routing.shape[2]
    row_bytes = layer_slots * np.dtype(routing.dtype).itemsize
    return RoutedExperts(
        data=routing.data[start * row_bytes : stop * row_bytes],
        shape=[stop - start, *routing.shape[1:]],
        dtype=routing.dtype,
        weights=(
            routing.weights[start * layer_slots * 4 : stop * layer_slots * 4] if routing.weights is not None else None
        ),
        valid=routing.valid[start:stop] if routing.valid is not None else None,
        format_version=routing.format_version,
    )


def concatenate_routed_experts(parts: Sequence[RoutedExperts]) -> RoutedExperts:
    """Concatenate paired streams once, not repeated growing-prefix byte copies."""
    if not parts:
        raise ValueError("cannot concatenate empty routing list")
    first = parts[0]
    if not all(routing_compatible(first, part) for part in parts):
        raise ValueError("cannot pack routing with different modes/dtypes/layouts")
    return RoutedExperts(
        data=b"".join(part.data for part in parts),
        shape=[sum(part.shape[0] for part in parts), *first.shape[1:]],
        dtype=first.dtype,
        weights=b"".join(part.weights for part in parts) if first.weights is not None else None,
        valid=b"".join(part.valid for part in parts) if first.valid is not None else None,
        format_version=first.format_version,
    )


def pad_routed_experts(routing: RoutedExperts, count: int) -> RoutedExperts:
    if count < 0:
        raise ValueError("routing padding count must be nonnegative")
    if count == 0:
        return copy_routed_experts(routing)
    ids, weights, valid = validate_routed_experts(routing)
    shape = [count, *routing.shape[1:]]
    if weights is None:
        # Preserve legacy IDs-only padding behavior.
        padding = RoutedExperts(data=bytes(math.prod(shape) * ids.dtype.itemsize), shape=shape, dtype=routing.dtype)
    else:
        # Spread zero-weight padding over a proven valid expert range. Avoid every
        # padding slot dispatching to expert 0; actual model bounds are checked later.
        upper = max(routing.shape[2], int(ids.max()) + 1 if ids.size else routing.shape[2])
        padded_ids = (np.arange(math.prod(shape), dtype=np.int64) % upper).astype(ids.dtype).reshape(shape)
        padding = RoutedExperts(
            data=padded_ids.tobytes(),
            shape=shape,
            dtype=routing.dtype,
            weights=bytes(math.prod(shape) * 4),
            valid=bytes(count),
            format_version=1,
        )
    return concatenate_routed_experts([routing, padding])
