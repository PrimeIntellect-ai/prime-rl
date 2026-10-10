"""Per-token side arrays (router-replay ids and weights, sampling masks and their sampler
logprobs) passed by handle: the inference server writes a request's rows to a shared file and
returns ``PayloadSegment``s, the orchestrator clips and shifts the segments into micro-batch
positions, and each trainer rank reads the rows of its window."""

import math
import os
import time
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, NamedTuple

import msgspec
import numpy as np


class PayloadField(NamedTuple):
    fill: int | float
    """Value at positions no segment covers, and padding of rows narrower than the widest
    one. An int fill makes ragged rows int32, a float fill float32."""
    window: Literal["inputs", "labels", "sequence"]
    """What a trainer rank reads on text micro batches: its CP chunk of input positions
    (routing, consumed by the forward pass), its CP chunk shifted one position ahead onto
    the labels (sampling masks ride at the sampled token's own position), or the whole
    sequence at its own positions (consumed by the loss after the CP gather). Multimodal
    micro batches read every field whole, because the model may defer CP sharding."""
    ragged: bool = False
    """Rows of varying width (sampling masks, their logprobs), stored as CSR (``ragged_bytes``):
    the segment's rows are the uint32 file offsets where each row's values start, followed in
    the file by one more offset where the last row's values end, so clipping a segment needs
    no knowledge of the row widths. Values are int32 for an int fill, float32 for a float fill.
    Fixed-width fields store the rows themselves."""


PAYLOAD_FIELDS = {
    "routed_experts": PayloadField(0, "inputs"),
    "routed_expert_weights": PayloadField(0.0, "inputs"),
    "sampling_mask": PayloadField(-1, "labels", ragged=True),
    "sampling_mask_logprobs": PayloadField(-math.inf, "sequence", ragged=True),
}


class PayloadSegment(msgspec.Struct, array_like=True, gc=False, frozen=True):
    """``rows`` consecutive rows of ``field`` starting at token position ``pos``, stored at
    byte ``offset`` of ``file`` as a C-contiguous ``[rows, *shape]`` ``dtype`` array."""

    field: str
    file: str
    offset: int
    pos: int
    rows: int
    dtype: str
    shape: list[int]

    @property
    def end(self) -> int:
        return self.pos + self.rows

    @property
    def row_bytes(self) -> int:
        return np.dtype(self.dtype).itemsize * math.prod(self.shape)


def clip_segments(segments: Iterable[PayloadSegment], lo: int, hi: int) -> list[PayloadSegment]:
    """The parts of ``segments`` inside positions ``[lo, hi)``."""
    clipped = []
    for segment in segments:
        start, end = max(segment.pos, lo), min(segment.end, hi)
        if start < end:
            offset = segment.offset + (start - segment.pos) * segment.row_bytes
            clipped.append(msgspec.structs.replace(segment, offset=offset, pos=start, rows=end - start))
    return clipped


def _values_dtype(field: str) -> type[np.number]:
    return np.float32 if isinstance(PAYLOAD_FIELDS[field].fill, float) else np.int32


def ragged_bytes(field: str, counts: np.ndarray, values: np.ndarray, offset: int) -> bytes:
    """CSR rows (``counts[i]`` values for token ``i``) laid out to be written at byte ``offset``
    of a file: ``len(counts) + 1`` uint32 start offsets, then the values."""
    values = np.ascontiguousarray(values, dtype=_values_dtype(field))
    starts = offset + 4 * (len(counts) + 1) + values.itemsize * np.concatenate([[0], np.cumsum(counts)])
    assert starts[-1] <= np.iinfo(np.uint32).max
    return starts.astype(np.uint32).tobytes() + values.tobytes()


def read_field(segments: list[PayloadSegment], field: str, lo: int, hi: int) -> np.ndarray | None:
    """Rows ``[lo, hi)`` of ``field``, as float32 for float fields and int32 otherwise, the
    field's fill where no segment covers a position; None when no segment carries ``field``.
    Ragged rows are padded to the widest row of all of the field's segments, as inline masks
    are padded to the micro batch's widest row, so every CP rank gets the same width. Later
    segments win."""
    field_segments = [segment for segment in segments if segment.field == field]
    if not field_segments:
        return None
    if PAYLOAD_FIELDS[field].ragged:
        return _read_ragged(field, field_segments, lo, hi)
    shape = [max(dims) for dims in zip(*(segment.shape for segment in field_segments))]
    dtype = np.float32 if np.dtype(field_segments[0].dtype).kind == "f" else np.int32
    out = np.full((hi - lo, *shape), PAYLOAD_FIELDS[field].fill, dtype=dtype)
    segments = clip_segments(field_segments, lo, hi)
    for segment, data in zip(segments, _pread([(s.file, s.offset, s.rows * s.row_bytes) for s in segments])):
        rows = np.frombuffer(data, dtype=segment.dtype).reshape(segment.rows, *segment.shape)
        out[(slice(segment.pos - lo, segment.end - lo), *(slice(0, dim) for dim in segment.shape))] = rows
    return out


def _read_ragged(field: str, segments: list[PayloadSegment], lo: int, hi: int) -> np.ndarray:
    dtype = _values_dtype(field)
    itemsize = np.dtype(dtype).itemsize
    segments = [segment for segment in segments if segment.rows]
    # Every segment's offsets (4 bytes per row) set the width; only the window's values are read.
    starts = [
        np.frombuffer(data, dtype=np.uint32).astype(np.int64)
        for data in _pread([(s.file, s.offset, 4 * (s.rows + 1)) for s in segments])
    ]
    width = max([int(np.diff(s).max()) // itemsize for s in starts] + [1])
    out = np.full((hi - lo, width), PAYLOAD_FIELDS[field].fill, dtype=dtype)
    window = []
    for segment, segment_starts in zip(segments, starts):
        a, b = max(segment.pos, lo) - segment.pos, min(segment.end, hi) - segment.pos
        if a < b:
            window.append((segment.file, segment.pos + a - lo, segment_starts[a : b + 1]))
    values = _pread([(file, int(w[0]), int(w[-1] - w[0])) for file, _, w in window])
    for (_, row, w), data in zip(window, values):
        counts = np.diff(w) // itemsize
        rows = out[row : row + len(counts)]
        rows[:] = PAYLOAD_FIELDS[field].fill
        rows[np.arange(width) < counts[:, None]] = np.frombuffer(data, dtype=dtype)
    return out


# Reads are latency-bound on a shared filesystem (a few rows per turn, spread over many files),
# so the files are read concurrently.
_READ_POOL = ThreadPoolExecutor(32, thread_name_prefix="payload-read")


def _pread(reads: list[tuple[str, int, int]]) -> list[bytes]:
    """The bytes of each ``(file, offset, size)``, in order. Each file is opened once."""
    by_file: dict[str, list[int]] = defaultdict(list)
    for i, (file, _, _) in enumerate(reads):
        by_file[file].append(i)
    data: list[bytes] = [b""] * len(reads)
    files = _READ_POOL.map(_pread_file, ([reads[i] for i in indices] for indices in by_file.values()))
    for indices, chunks in zip(by_file.values(), files):
        for i, chunk in zip(indices, chunks):
            data[i] = chunk
    return data


def _pread_file(reads: list[tuple[str, int, int]], attempts: int = 5) -> list[bytes]:
    """The bytes of each ``(file, offset, size)`` of one file. A shared filesystem can briefly
    serve a short read of a file another node just wrote, so a short read reopens the file
    and retries with backoff."""
    file = reads[0][0]
    for attempt in range(attempts):
        if attempt:
            time.sleep(0.05 * 2**attempt)
        fd = os.open(file, os.O_RDONLY)
        try:
            data = [os.pread(fd, size, offset) for _, offset, size in reads]
        finally:
            os.close(fd)
        short = [(offset, size, len(chunk)) for (_, offset, size), chunk in zip(reads, data) if len(chunk) != size]
        if not short:
            return data
    offset, size, got = short[0]
    raise OSError(f"Payload file {file} returned {got} of {size} bytes at offset {offset} after {attempts} reads")
