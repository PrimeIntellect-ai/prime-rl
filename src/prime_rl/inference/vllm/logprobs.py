"""Unified packed completion-logprob contract for `/inference/v1/generate`.

Wire schema on ``choice["completion_logprobs"]``
------------------------------------------------
Both modes place ``"data"`` first so HTTP clients can splice large base64 payloads
out of raw JSON bytes as zero-copy ``memoryview`` buffers before ``json.loads``.
In both modes, the leading ``4 * T`` decoded bytes (``T = len(choice["token_ids"])``)
are little-endian ``float32[T]`` sampled-token logprobs where index ``i`` is
explicitly the logprob of ``choice["token_ids"][i]``, regardless of whether the
sampled token appears inside any top-k candidate head.

1. Sampled-token-only mode (``format = "f32-le-v1"``, ordinary RL):
   ``{"data": b64(<f4[T]), "format": "f32-le-v1", "shape": [T], "dtype": "float32"}``
   Zero per-token overhead over legacy ``completion_logprobs``; ``shape`` and
   ``dtype`` are kept so unmigrated ``renderers`` clients decode it directly.

2. Sampled-token plus candidate rows (``format = "f32-u32-f32-le-v1"``, top-k
   heads and replayed supports):
   ``{"data": b64(<f4[T] || <u4[N] || <f4[N]), "format": "f32-u32-f32-le-v1",
      "offsets": [0, ..., N], "num_top_logprobs": int}``
   - ``[0 : 4*T]``: ``<f4[T]`` sampled logprobs for ``token_ids[0..T-1]``.
   - ``[4*T : 4*T + 4*N]``: ``<u4[N]`` flat candidate token IDs.
   - ``[4*T + 4*N : 4*T + 8*N]``: ``<f4[N]`` flat candidate logprobs.
   - ``offsets``: length ``T + 1`` non-decreasing row bounds (``N = offsets[-1]``).

Migration & removal of legacy ``compact_logprobs``
--------------------------------------------------
Clients opt into unified candidate-row responses with
``sampling_params.extra_args["prl_packed_logprobs"] = True`` (which takes
precedence if ``prl_compact_logprobs`` is also sent for backwards compatibility
with older servers). Legacy ``prl_compact_logprobs = True`` alone continues to
emit ``choice["compact_logprobs"]`` (``u32-f32-le-v1``).

The legacy ``compact_logprobs`` field and ``prl_compact_logprobs`` request flag
can be removed once ``PrimeIntellect-ai/renderers`` (and all deployed rollout
clients) parse ``f32-u32-f32-le-v1`` on ``completion_logprobs`` and stop
requesting ``prl_compact_logprobs``.
"""

from __future__ import annotations

import binascii
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import pybase64
from vllm.logprobs import FlatLogprobs, SampleLogprobs

# vLLM clamp for missing or -inf logprobs; renderers treat it as "no sampling evidence".
LOGPROB_SENTINEL = -9999.0
NEG_INF_BITS = -8388608  # IEEE float32 -inf; plain IDs decode to missing evidence.

SAMPLED_LOGPROBS_FORMAT = "f32-le-v1"
PACKED_LOGPROBS_FORMAT = "f32-u32-f32-le-v1"
LEGACY_COMPACT_LOGPROBS_FORMAT = "u32-f32-le-v1"

PACKED_LOGPROBS_ARG = "prl_packed_logprobs"
LEGACY_COMPACT_LOGPROBS_ARG = "prl_compact_logprobs"

ROUTED_EXPERTS_DATA_PREFIX = b'"routed_experts":{"data":"'
COMPLETION_LOGPROBS_DATA_PREFIX = b'"completion_logprobs":{"data":"'
LEGACY_COMPACT_LOGPROBS_DATA_PREFIX = b'"compact_logprobs":{"data":"'


class PackedLogprobsError(ValueError):
    """Raised when packed logprob transport evidence is malformed or missing."""


@dataclass(frozen=True)
class DecodedLogprobs:
    sampled: np.ndarray
    ids: np.ndarray | None = None
    values: np.ndarray | None = None
    offsets: list[int] | None = None
    num_top_logprobs: int | None = None


def _as_nonnegative_token_ids(token_ids: Sequence[int]) -> np.ndarray:
    arr = np.asarray(token_ids, dtype=np.int64)
    if arr.ndim != 1 or (arr < 0).any() or (arr > 0xFFFFFFFF).any():
        raise ValueError("token_ids must be 1-D non-negative uint32 integers")
    return arr.astype("<u4", copy=False)


def pack_sampled_logprobs(logprobs: FlatLogprobs) -> np.ndarray:
    """The sampled token logprob per position (vLLM stores it first), clamped like upstream."""
    flat = np.asarray(logprobs.logprobs, dtype=np.float32)
    starts = np.asarray(logprobs.start_indices, dtype=np.int64)
    has_entry = starts < np.asarray(logprobs.end_indices, dtype=np.int64)
    values = np.full(len(starts), LOGPROB_SENTINEL, dtype=np.float32)
    values[has_entry] = flat[starts[has_entry]]
    return np.maximum(values, LOGPROB_SENTINEL).astype("<f4", copy=False)


def encode_sampled_logprobs(values: np.ndarray) -> dict[str, Any]:
    """Encode sampled-only logprobs in ``f32-le-v1`` (compatible with legacy renderers)."""
    arr = np.ascontiguousarray(values, dtype="<f4")
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D sampled logprobs, got shape {arr.shape}")
    return {
        "data": pybase64.b64encode(memoryview(arr)).decode("ascii"),
        "format": SAMPLED_LOGPROBS_FORMAT,
        "shape": [len(arr)],
        "dtype": "float32",
    }


def encode_packed_logprobs(
    sampled: np.ndarray,
    ids: np.ndarray,
    values: np.ndarray,
    offsets: Sequence[int] | np.ndarray,
    num_top_logprobs: int,
) -> dict[str, Any]:
    """Encode sampled logprobs and ragged candidate rows in ``f32-u32-f32-le-v1``."""
    sampled_arr = np.ascontiguousarray(sampled, dtype="<f4")
    ids_arr = np.ascontiguousarray(ids, dtype="<u4")
    values_arr = np.ascontiguousarray(values, dtype="<f4")
    if sampled_arr.ndim != 1 or ids_arr.ndim != 1 or values_arr.ndim != 1:
        raise ValueError("Packed logprob arrays must be 1-D")
    offsets_list = [int(x) for x in offsets]
    if (
        len(offsets_list) != len(sampled_arr) + 1
        or offsets_list[0] != 0
        or any(end < start for start, end in pairwise(offsets_list))
        or offsets_list[-1] != len(ids_arr)
        or len(ids_arr) != len(values_arr)
    ):
        raise ValueError("Invalid packed logprob offsets or buffer lengths")
    if type(num_top_logprobs) is not int or num_top_logprobs < -1:
        raise ValueError("num_top_logprobs must be an integer >= -1")
    sampled_clamped = np.maximum(sampled_arr, LOGPROB_SENTINEL).astype("<f4", copy=False)
    values_clamped = np.maximum(values_arr, LOGPROB_SENTINEL).astype("<f4", copy=False)
    buf = sampled_clamped.tobytes() + ids_arr.tobytes() + values_clamped.tobytes()
    return {
        "data": pybase64.b64encode(buf).decode("ascii"),
        "format": PACKED_LOGPROBS_FORMAT,
        "offsets": offsets_list,
        "num_top_logprobs": num_top_logprobs,
    }


def unpack_paired_support(packed: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unpack GPU-captured int64 ``(logprob_bits << 32) | id`` rows into ``(ids, values, counts)``."""
    bits = np.asarray(packed, dtype=np.uint64)
    ids = (bits & np.uint64(0xFFFFFFFF)).astype("<u4")
    values = ((bits >> np.uint64(32)).astype(np.uint32).view(np.int32) ^ NEG_INF_BITS).view("<f4")
    counts_arr = np.asarray(counts, dtype=np.int32)
    if (counts_arr < 0).any() or int(counts_arr.sum()) != len(ids) or not np.isfinite(values).all():
        raise ValueError("Incomplete paired sampling-support evidence")
    return ids, values, counts_arr


def serialize_paired_support_logprobs(
    packed: np.ndarray,
    counts: np.ndarray,
    token_ids: Sequence[int],
    *,
    strip_leading_sampled: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return aligned int32 mask IDs and unified ``completion_logprobs`` wire data.

    When ``strip_leading_sampled`` is true (``prl_sampler_rows``), position 0 of each
    captured row holds the sampled token and the remaining entries hold the top-k head;
    stripping the leading slot avoids duplicating the sampled ID in candidate rows.
    When false (``prl_paired_capture``), each row is a complete replayed support in
    ascending token-ID order and the sampled token is located by matching ``token_ids``.
    """
    ids, values, counts_arr = unpack_paired_support(packed, counts)
    expected_ids = _as_nonnegative_token_ids(token_ids)
    num_tokens = len(expected_ids)
    if len(counts_arr) != num_tokens or (counts_arr <= 0).any():
        raise ValueError("Incomplete paired sampling-support evidence")
    if num_tokens == 0:
        return ids.astype(np.int32, copy=False), encode_packed_logprobs(np.empty(0, dtype="<f4"), ids, values, [0], 0)
    offsets = np.zeros(num_tokens + 1, dtype=np.int64)
    np.cumsum(counts_arr, out=offsets[1:])
    if strip_leading_sampled:
        starts = offsets[:-1]
        if not np.array_equal(ids[starts], expected_ids):
            raise ValueError("Leading sampler-row token IDs do not match completion token_ids")
        sampled = values[starts]
        keep = np.ones(len(ids), dtype=bool)
        keep[starts] = False
        cand_ids = ids[keep]
        cand_values = values[keep]
        cand_counts = counts_arr - 1
        if (cand_counts == 0).all():
            return ids.astype(np.int32, copy=False), encode_sampled_logprobs(sampled)
        cand_offsets = [0, *np.cumsum(cand_counts).tolist()]
        payload = encode_packed_logprobs(
            sampled,
            cand_ids,
            cand_values,
            cand_offsets,
            int(cand_counts.max(initial=0)),
        )
    else:
        row_idx = np.repeat(np.arange(num_tokens, dtype=np.int64), counts_arr)
        matches = ids == expected_ids[row_idx]
        row_matches = np.add.reduceat(matches.astype(np.int32), offsets[:-1])
        if not (row_matches == 1).all():
            raise ValueError("Sampled token must appear exactly once in each replayed support row")
        sampled = values[matches]
        payload = encode_packed_logprobs(
            sampled,
            ids,
            values,
            offsets.tolist(),
            int(counts_arr.max(initial=0)),
        )
    return ids.astype(np.int32, copy=False), payload


def serialize_sample_logprobs(
    logprobs: SampleLogprobs,
    token_ids: Sequence[int],
    num_top_logprobs: int | None = None,
) -> dict[str, Any]:
    """Serialize vLLM ``FlatLogprobs`` or per-token dicts into unified ``completion_logprobs``."""
    expected_ids = _as_nonnegative_token_ids(token_ids)
    num_tokens = len(expected_ids)
    if len(logprobs) != num_tokens:
        raise ValueError(f"Logprob count ({len(logprobs)}) does not match token count ({num_tokens})")

    if num_top_logprobs is None or num_top_logprobs == 0:
        if isinstance(logprobs, FlatLogprobs):
            return encode_sampled_logprobs(pack_sampled_logprobs(logprobs))
        sampled = np.full(num_tokens, LOGPROB_SENTINEL, dtype="<f4")
        for idx, (tid, row) in enumerate(zip(expected_ids.tolist(), logprobs, strict=True)):
            if row and tid in row:
                sampled[idx] = max(float(np.float32(row[tid].logprob)), LOGPROB_SENTINEL)
        return encode_sampled_logprobs(sampled)

    if isinstance(logprobs, FlatLogprobs):
        starts = logprobs.start_indices
        ends = logprobs.end_indices
        raw_ids = np.asarray(logprobs.token_ids, dtype="<u4")
        raw_vals = np.maximum(np.asarray(logprobs.logprobs, dtype="<f4"), LOGPROB_SENTINEL).astype("<f4", copy=False)
        lengths = np.asarray(ends, dtype=np.int64) - np.asarray(starts, dtype=np.int64)
        if starts != [0, *ends[:-1]] or (ends and ends[-1] != len(raw_ids)):
            flat_ids = np.concatenate([raw_ids[s:e] for s, e in zip(starts, ends)]) if starts else raw_ids[:0]
            flat_vals = np.concatenate([raw_vals[s:e] for s, e in zip(starts, ends)]) if starts else raw_vals[:0]
        else:
            flat_ids = raw_ids
            flat_vals = raw_vals
        offsets = [0, *np.cumsum(lengths).tolist()]
    else:
        lengths_list: list[int] = []
        ids_list: list[int] = []
        vals_list: list[float] = []
        for row in logprobs:
            if not row:
                lengths_list.append(0)
                continue
            lengths_list.append(len(row))
            ids_list.extend(row.keys())
            vals_list.extend(item.logprob for item in row.values())
        flat_ids = np.asarray(ids_list, dtype="<u4")
        flat_vals = np.maximum(np.asarray(vals_list, dtype="<f4"), LOGPROB_SENTINEL).astype("<f4", copy=False)
        offsets = [0, *np.cumsum(lengths_list).tolist()]

    sampled = np.full(num_tokens, LOGPROB_SENTINEL, dtype="<f4")
    for idx, (tid, start, end) in enumerate(zip(expected_ids.tolist(), offsets[:-1], offsets[1:], strict=True)):
        row_ids = flat_ids[start:end]
        row_vals = flat_vals[start:end]
        hit = row_vals[row_ids == tid]
        if len(hit):
            if len(hit) > 1 and not np.all(hit == hit[0]):
                raise ValueError("Repeated sampled token logprobs disagree")
            sampled[idx] = hit[0]

    return encode_packed_logprobs(sampled, flat_ids, flat_vals, offsets, num_top_logprobs)


def _strip_base64_fields(raw: bytes, prefix: bytes) -> tuple[bytes, list[memoryview]]:
    """Splice a large base64 string field out of raw JSON bytes as zero-copy memoryviews."""
    chunks: list[bytes] = []
    fields: list[memoryview] = []
    cursor = 0
    while (start := raw.find(prefix, cursor)) >= 0:
        start += len(prefix)
        end = raw.index(b'"', start)
        chunks.append(raw[cursor:start])
        fields.append(memoryview(raw)[start:end])
        cursor = end
    if not fields:
        return raw, fields
    chunks.append(raw[cursor:])
    return b"".join(chunks), fields


def parse_generate_response_packed(raw: bytes) -> dict[str, Any]:
    """Parse a ``/inference/v1/generate`` JSON body without decoding packed base64 buffers into ``str``."""
    stripped, routed_data = _strip_base64_fields(raw, ROUTED_EXPERTS_DATA_PREFIX)
    stripped, completion_data = _strip_base64_fields(stripped, COMPLETION_LOGPROBS_DATA_PREFIX)
    stripped, compact_data = _strip_base64_fields(stripped, LEGACY_COMPACT_LOGPROBS_DATA_PREFIX)
    payload: dict[str, Any] = json.loads(stripped)
    for field, buffers in (
        ("routed_experts", routed_data),
        ("completion_logprobs", completion_data),
        ("compact_logprobs", compact_data),
    ):
        if buffers:
            pending = iter(buffers)
            for choice in payload["choices"]:
                if isinstance(choice.get(field), dict) and choice[field].get("data") == "":
                    choice[field]["data"] = next(pending)
    return payload


def _b64decode_strict(data: Any, label: str) -> bytes:
    try:
        return pybase64.b64decode(data, validate=True)
    except (TypeError, ValueError, binascii.Error) as exc:
        raise PackedLogprobsError(f"Invalid {label} base64") from exc


def _validate_sampled_values(values: np.ndarray) -> None:
    if not np.isfinite(values).all() or (values <= LOGPROB_SENTINEL).any() or (values > 1e-6).any():
        raise PackedLogprobsError("Missing, non-finite, or positive sampled-token logprob evidence")


def decode_completion_logprobs(
    packed: Mapping[str, Any],
    completion_ids: Sequence[int],
) -> DecodedLogprobs:
    """Decode unified ``completion_logprobs`` (``f32-le-v1``, ``f32-u32-f32-le-v1``, or legacy sampled-only)."""
    if not isinstance(packed, Mapping):
        raise PackedLogprobsError("completion_logprobs must be an object")
    num_tokens = len(completion_ids)
    fmt = packed.get("format")
    if fmt is None or fmt == SAMPLED_LOGPROBS_FORMAT:
        dtype = packed.get("dtype")
        if fmt is None and dtype != "float32":
            raise PackedLogprobsError("Unsupported legacy completion_logprobs dtype")
        if dtype is not None and dtype != "float32":
            raise PackedLogprobsError("Invalid completion_logprobs dtype")
        shape = packed.get("shape")
        if shape is not None and shape != [num_tokens]:
            raise PackedLogprobsError("completion_logprobs shape mismatch")
        if "data" not in packed:
            raise PackedLogprobsError("Missing completion_logprobs data")
        raw = _b64decode_strict(packed["data"], "completion_logprobs")
        if len(raw) != num_tokens * 4:
            raise PackedLogprobsError("Invalid completion_logprobs buffer length")
        sampled = np.frombuffer(raw, dtype="<f4", count=num_tokens)
        _validate_sampled_values(sampled)
        return DecodedLogprobs(sampled=sampled)

    if fmt != PACKED_LOGPROBS_FORMAT:
        raise PackedLogprobsError(f"Unsupported completion_logprobs format: {fmt!r}")

    offsets = packed.get("offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != num_tokens + 1
        or any(type(x) is not int or x < 0 for x in offsets)
        or offsets[0] != 0
        or any(end < start for start, end in pairwise(offsets))
    ):
        raise PackedLogprobsError("Invalid completion_logprobs offsets")
    num_top = packed.get("num_top_logprobs")
    if type(num_top) is not int or num_top < -1:
        raise PackedLogprobsError("Invalid completion_logprobs num_top_logprobs")
    if "data" not in packed:
        raise PackedLogprobsError("Missing completion_logprobs data")
    raw = _b64decode_strict(packed["data"], "completion_logprobs")
    count = offsets[-1]
    sampled_bytes = num_tokens * 4
    if len(raw) != sampled_bytes + count * 8:
        raise PackedLogprobsError("Invalid completion_logprobs buffer length")
    sampled = np.frombuffer(raw, dtype="<f4", count=num_tokens, offset=0)
    ids = np.frombuffer(raw, dtype="<u4", count=count, offset=sampled_bytes)
    values = np.frombuffer(raw, dtype="<f4", count=count, offset=sampled_bytes + count * 4)
    _validate_sampled_values(sampled)
    if not np.isfinite(values).all() or (values > 1e-6).any():
        raise PackedLogprobsError("Invalid candidate logprob values")
    return DecodedLogprobs(
        sampled=sampled,
        ids=ids,
        values=values,
        offsets=offsets,
        num_top_logprobs=num_top,
    )


def _decode_legacy_compact_logprobs(
    packed: Mapping[str, Any],
    completion_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, list[int], int]:
    if not isinstance(packed, Mapping) or packed.get("format") != LEGACY_COMPACT_LOGPROBS_FORMAT:
        raise PackedLogprobsError("Unsupported legacy compact_logprobs format")
    offsets = packed.get("offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != len(completion_ids) + 1
        or any(type(x) is not int or x < 0 for x in offsets)
        or offsets[0] != 0
        or any(end <= start for start, end in pairwise(offsets))
    ):
        raise PackedLogprobsError("Invalid legacy compact_logprobs offsets")
    num_top = packed.get("num_top_logprobs")
    if type(num_top) is not int or num_top < -1:
        raise PackedLogprobsError("Invalid legacy compact_logprobs num_top_logprobs")
    if "data" not in packed:
        raise PackedLogprobsError("Missing legacy compact_logprobs data")
    raw = _b64decode_strict(packed["data"], "compact_logprobs")
    count = offsets[-1]
    if len(raw) != count * 8:
        raise PackedLogprobsError("Invalid legacy compact_logprobs buffer length")
    ids = np.frombuffer(raw, dtype="<u4", count=count)
    values = np.frombuffer(raw, dtype="<f4", count=count, offset=count * 4)
    if not np.isfinite(values).all() or (values > 1e-6).any():
        raise PackedLogprobsError("Invalid legacy compact_logprobs values")
    return ids, values, offsets, num_top


def _decode_packed_sampling_mask(mask: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name in ("ids", "counts"):
        packed = mask.get(name)
        if not isinstance(packed, Mapping) or packed.get("dtype") != "int32" or "data" not in packed:
            raise PackedLogprobsError("Invalid packed sampling mask")
        raw = _b64decode_strict(packed["data"], "sampling_mask")
        values = np.frombuffer(raw, dtype=np.int32)
        if packed.get("shape") != [len(values)] or (values < 0).any():
            raise PackedLogprobsError("Invalid packed sampling mask")
        arrays[name] = values
    return arrays["ids"], arrays["counts"]


def _select_sampler_head(
    head: dict[int, float],
    sampled_id: int,
    sampled_logprob: float,
    k: int,
) -> tuple[list[int], list[float]]:
    if sampled_id in head and not math.isclose(head[sampled_id], sampled_logprob, rel_tol=1e-6, abs_tol=1e-5):
        raise PackedLogprobsError("Sampler head and sampled-token logprob disagree")
    ranked = sorted(
        ((i, p) for i, p in head.items() if p != LOGPROB_SENTINEL),
        key=lambda x: (-x[1], x[0]),
    )
    if len(ranked) > k and ranked[k - 1][1] == ranked[k][1]:
        boundary = ranked[k][1]
        ranked = [(i, p) for i, p in ranked if p > boundary]
    else:
        ranked = ranked[:k]
    if not ranked or math.fsum(math.exp(p) for _, p in ranked) > 1.0001:
        raise PackedLogprobsError("Invalid sampler head probability mass")
    if len(head) < k + 2:
        raise PackedLogprobsError("Engine returned fewer top logprobs than requested")
    return [i for i, _ in ranked], [p for _, p in ranked]


def _validate_replayed_rows(
    ids: np.ndarray,
    values: np.ndarray,
    offsets: list[int],
    completion_ids: Sequence[int],
    sampled: np.ndarray | None,
    mask: Any,
) -> np.ndarray:
    counts = np.diff(np.asarray(offsets, dtype=np.int64)).astype(np.int32)
    if len(counts) != len(completion_ids) or (counts <= 0).any():
        raise PackedLogprobsError("Replayed support rows must be non-empty")
    if isinstance(mask, Mapping):
        mask_ids, mask_counts = _decode_packed_sampling_mask(mask)
        if not np.array_equal(counts, mask_counts) or not np.array_equal(mask_ids, ids):
            raise PackedLogprobsError("Sampler logprobs must cover exactly the recorded sampling support")
    if (values == LOGPROB_SENTINEL).any():
        raise PackedLogprobsError("Missing sampling-support probabilities")
    ids_i64 = ids.astype(np.int64)
    rows = np.repeat(np.arange(len(counts)), counts)
    order = np.lexsort((ids_i64, rows))
    sorted_ids, sorted_rows = ids_i64[order], rows[order]
    repeated = (sorted_ids[1:] == sorted_ids[:-1]) & (sorted_rows[1:] == sorted_rows[:-1])
    sampled_hits = ids_i64 == np.asarray(completion_ids, dtype=np.int64)[rows]
    if repeated.any() or np.count_nonzero(sampled_hits) != len(counts):
        raise PackedLogprobsError("Invalid recorded sampling support")
    support_sampled = values[sampled_hits]
    if sampled is not None and not np.allclose(support_sampled, sampled, rtol=1e-6, atol=1e-5):
        raise PackedLogprobsError("Sampled logprobs disagree with replayed support")
    if len(counts):
        mass = np.add.reduceat(np.exp(values.astype(np.float64)), offsets[:-1])
        if not np.allclose(mass, 1.0, rtol=0, atol=1e-4):
            raise PackedLogprobsError("Sampler probabilities must normalize over the recorded support")
    return support_sampled


def parse_completion_logprobs(
    choice: Mapping[str, Any],
    completion_ids: Sequence[int],
    *,
    head_size: int | None = None,
    replayed: bool | None = None,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Parse unified ``completion_logprobs`` (or legacy ``compact_logprobs``) into ``(sampled, sampler_support)``."""
    is_replayed = (choice.get("sampling_mask") is not None) if replayed is None else replayed

    if choice.get("completion_logprobs") is not None:
        decoded = decode_completion_logprobs(choice["completion_logprobs"], completion_ids)
        if decoded.ids is None:
            if head_size is not None:
                raise PackedLogprobsError("Missing requested top logprobs")
            return decoded.sampled.astype(np.float32, copy=False), None

        assert decoded.values is not None and decoded.offsets is not None
        if is_replayed:
            _validate_replayed_rows(
                decoded.ids,
                decoded.values,
                decoded.offsets,
                completion_ids,
                decoded.sampled,
                choice.get("sampling_mask"),
            )
            counts = np.diff(np.asarray(decoded.offsets, dtype=np.int64)).astype(np.int32)
            return decoded.sampled.astype(np.float32, copy=False), {
                "ids": decoded.ids.astype(np.int32, copy=False),
                "counts": counts,
                "logprobs": decoded.values.astype(np.float32, copy=False),
                "replayed": True,
            }

        top_ids: list[list[int]] = []
        top_logprobs: list[list[float]] = []
        for token_id, sampled_lp, start, end in zip(
            completion_ids, decoded.sampled.tolist(), decoded.offsets[:-1], decoded.offsets[1:], strict=True
        ):
            items = list(zip(decoded.ids[start:end].tolist(), decoded.values[start:end].tolist()))
            row = dict(items)
            if any(row[i] != p for i, p in items):
                raise PackedLogprobsError("Repeated candidate logprobs disagree")
            if token_id in row and not math.isclose(row[token_id], sampled_lp, rel_tol=1e-6, abs_tol=1e-5):
                raise PackedLogprobsError("Sampled logprob disagrees with candidate row")
            if head_size is not None:
                selected_ids, selected_lps = _select_sampler_head(row, token_id, sampled_lp, head_size)
                top_ids.append(selected_ids)
                top_logprobs.append(selected_lps)

        if head_size is None:
            return decoded.sampled.astype(np.float32, copy=False), None
        return decoded.sampled.astype(np.float32, copy=False), {
            "ids": np.fromiter((i for r in top_ids for i in r), dtype=np.int32),
            "counts": np.fromiter((len(r) for r in top_ids), dtype=np.int32, count=len(top_ids)),
            "logprobs": np.fromiter((p for r in top_logprobs for p in r), dtype=np.float32),
            "replayed": False,
        }

    if choice.get("compact_logprobs") is not None:
        ids, values, offsets, _ = _decode_legacy_compact_logprobs(choice["compact_logprobs"], completion_ids)
        if is_replayed:
            sampled = _validate_replayed_rows(ids, values, offsets, completion_ids, None, choice.get("sampling_mask"))
            counts = np.diff(np.asarray(offsets, dtype=np.int64)).astype(np.int32)
            return sampled.astype(np.float32, copy=False), {
                "ids": ids.astype(np.int32, copy=False),
                "counts": counts,
                "logprobs": values.astype(np.float32, copy=False),
                "replayed": True,
            }
        sampled_list: list[float] = []
        top_ids = []
        top_logprobs = []
        for token_id, start, end in zip(completion_ids, offsets[:-1], offsets[1:], strict=True):
            items = list(zip(ids[start:end].tolist(), values[start:end].tolist()))
            row = dict(items)
            if any(row[i] != p for i, p in items):
                raise PackedLogprobsError("Repeated compact logprobs disagree")
            if token_id not in row or row[token_id] == LOGPROB_SENTINEL:
                raise PackedLogprobsError("Missing compact sampled-token evidence")
            sampled_list.append(row[token_id])
            if head_size is not None:
                selected_ids, selected_lps = _select_sampler_head(row, token_id, row[token_id], head_size)
                top_ids.append(selected_ids)
                top_logprobs.append(selected_lps)
        sampled_arr = np.asarray(sampled_list, dtype=np.float32)
        if head_size is None:
            return sampled_arr, None
        return sampled_arr, {
            "ids": np.fromiter((i for r in top_ids for i in r), dtype=np.int32),
            "counts": np.fromiter((len(r) for r in top_ids), dtype=np.int32, count=len(top_ids)),
            "logprobs": np.fromiter((p for r in top_logprobs for p in r), dtype=np.float32),
            "replayed": False,
        }

    raise PackedLogprobsError("Choice contains neither completion_logprobs nor compact_logprobs")
