"""Per-token side arrays passed by handle instead of inline.

The inference server writes a request's payload rows (router-replay expert ids,
sampling masks) to one file under a shared payload root and returns
``PayloadSegment``s. Env servers, the orchestrator and the transport only move
the segments: the orchestrator clips them to a sample and shifts them into
micro-batch positions, and each trainer rank reads the rows of its window.
"""

import math
import os
from collections.abc import Iterable

import msgspec
import numpy as np


class PayloadSegment(msgspec.Struct, array_like=True, gc=False, frozen=True):
    """``rows`` consecutive rows of ``field`` starting at token position ``pos``,
    stored at byte ``offset`` of ``file`` as a C-contiguous ``[rows, *shape]``
    ``dtype`` array. ``pos`` is relative to whatever sequence holds the segment
    (request, branch, sample or micro batch)."""

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
    """The parts of ``segments`` inside positions ``[lo, hi)``, positions unchanged."""
    clipped = []
    for segment in segments:
        start, end = max(segment.pos, lo), min(segment.end, hi)
        if start >= end:
            continue
        if (start, end) == (segment.pos, segment.end):
            clipped.append(segment)
            continue
        clipped.append(
            msgspec.structs.replace(
                segment,
                offset=segment.offset + (start - segment.pos) * segment.row_bytes,
                pos=start,
                rows=end - start,
            )
        )
    return clipped


def shift_segments(segments: Iterable[PayloadSegment], delta: int) -> list[PayloadSegment]:
    return [msgspec.structs.replace(segment, pos=segment.pos + delta) for segment in segments]


def covers(segments: Iterable[PayloadSegment], field: str, lo: int, hi: int) -> bool:
    """Whether the ``field`` segments cover every position in ``[lo, hi)``."""
    covered = np.zeros(max(hi - lo, 0), dtype=bool)
    for segment in clip_segments((s for s in segments if s.field == field), lo, hi):
        covered[segment.pos - lo : segment.end - lo] = True
    return bool(covered.all())


def read_field(
    segments: Iterable[PayloadSegment], field: str, lo: int, hi: int, fill: int | float, dtype: np.dtype
) -> np.ndarray | None:
    """Rows ``[lo, hi)`` of ``field`` as a ``[hi - lo, *shape]`` array, ``fill`` where no
    segment covers a position, or None when no segment carries ``field``. The row shape is
    the elementwise max over the field's segments; narrower rows fill the leading entries
    (sampling masks are written at each request's own width). Later segments win."""
    field_segments = [segment for segment in segments if segment.field == field]
    if not field_segments:
        return None
    shape = [max(dims) for dims in zip(*(segment.shape for segment in field_segments), strict=True)]
    out = np.full((max(hi - lo, 0), *shape), fill, dtype=dtype)
    fds: dict[str, int] = {}
    try:
        for segment in clip_segments(field_segments, lo, hi):
            fd = fds.get(segment.file)
            if fd is None:
                fd = fds[segment.file] = os.open(segment.file, os.O_RDONLY)
            data = os.pread(fd, segment.rows * segment.row_bytes, segment.offset)
            if len(data) != segment.rows * segment.row_bytes:
                raise ValueError(f"short payload read from {segment.file} at {segment.offset}: {segment}")
            rows = np.frombuffer(data, dtype=segment.dtype).reshape(segment.rows, *segment.shape)
            out[(slice(segment.pos - lo, segment.end - lo), *(slice(0, dim) for dim in segment.shape))] = rows
    finally:
        for fd in fds.values():
            os.close(fd)
    return out
