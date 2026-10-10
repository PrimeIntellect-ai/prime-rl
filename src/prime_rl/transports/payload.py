"""Per-token side arrays (router-replay ids, sampling masks) passed by handle: the
inference server writes a request's rows to a shared file and returns ``PayloadSegment``s,
the orchestrator clips and shifts the segments into micro-batch positions, and each
trainer rank reads the rows of its window."""

import math
import os
import time
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

import msgspec
import numpy as np


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


RAGGED_FIELDS = frozenset({"sampling_mask"})
"""Fields whose rows vary in width (a few kept ids per sampled token), stored as CSR
(``ragged_bytes``): the segment's rows are the uint32 file offsets where each row's int32
values start, followed in the file by one more offset where the last row's values end, so
clipping a segment needs no knowledge of the row widths. Other fields store the rows
themselves."""


def ragged_bytes(counts: np.ndarray, values: np.ndarray, offset: int) -> bytes:
    """CSR rows (``counts[i]`` values for token ``i``) laid out to be written at byte ``offset``
    of a file: ``len(counts) + 1`` uint32 start offsets, then the int32 values."""
    values = np.ascontiguousarray(values, dtype=np.int32)
    starts = offset + 4 * (len(counts) + 1) + values.itemsize * np.concatenate([[0], np.cumsum(counts)])
    assert starts[-1] <= np.iinfo(np.uint32).max
    return starts.astype(np.uint32).tobytes() + values.tobytes()


def read_field(segments: list[PayloadSegment], field: str, lo: int, hi: int, fill: int) -> np.ndarray | None:
    """Rows ``[lo, hi)`` of ``field`` as int32, ``fill`` where no segment covers a position;
    None when no segment carries ``field``. Ragged rows are padded to the widest row of all of
    the field's segments, as inline masks are padded to the micro batch's widest row, so every
    CP rank gets the same width. Later segments win."""
    field_segments = [segment for segment in segments if segment.field == field]
    if not field_segments:
        return None
    if field in RAGGED_FIELDS:
        return _read_ragged(field_segments, lo, hi, fill)
    shape = [max(dims) for dims in zip(*(segment.shape for segment in field_segments))]
    out = np.full((hi - lo, *shape), fill, dtype=np.int32)
    segments = clip_segments(field_segments, lo, hi)
    for segment, data in zip(segments, _pread([(s.file, s.offset, s.rows * s.row_bytes) for s in segments])):
        rows = np.frombuffer(data, dtype=segment.dtype).reshape(segment.rows, *segment.shape)
        out[(slice(segment.pos - lo, segment.end - lo), *(slice(0, dim) for dim in segment.shape))] = rows
    return out


def _read_ragged(segments: list[PayloadSegment], lo: int, hi: int, fill: int) -> np.ndarray:
    segments = [segment for segment in segments if segment.rows]
    # Every segment's offsets (4 bytes per row) set the width; only the window's values are read.
    starts = [
        np.frombuffer(data, dtype=np.uint32).astype(np.int64)
        for data in _pread([(s.file, s.offset, 4 * (s.rows + 1)) for s in segments])
    ]
    width = max([int(np.diff(s).max()) // 4 for s in starts] + [1])
    out = np.full((hi - lo, width), fill, dtype=np.int32)
    window = []
    for segment, segment_starts in zip(segments, starts):
        a, b = max(segment.pos, lo) - segment.pos, min(segment.end, hi) - segment.pos
        if a < b:
            window.append((segment.file, segment.pos + a - lo, segment_starts[a : b + 1]))
    values = _pread([(file, int(w[0]), int(w[-1] - w[0])) for file, _, w in window])
    for (_, row, w), data in zip(window, values):
        counts = np.diff(w) // 4
        rows = out[row : row + len(counts)]
        rows[:] = fill
        rows[np.arange(width) < counts[:, None]] = np.frombuffer(data, dtype=np.int32)
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
