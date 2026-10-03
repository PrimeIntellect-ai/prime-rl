"""Per-token side arrays (router-replay ids, sampling masks) passed by handle: the
inference server writes a request's rows to a shared file and returns ``PayloadSegment``s,
the orchestrator clips and shifts the segments into micro-batch positions, and each
trainer rank reads the rows of its window."""

import math
import os
from collections.abc import Iterable

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


def read_field(segments: list[PayloadSegment], field: str, lo: int, hi: int, fill: int) -> np.ndarray | None:
    """Rows ``[lo, hi)`` of ``field`` as int32, ``fill`` where no segment covers a position;
    None when no segment carries ``field``. Rows narrower than the field's widest segment
    fill the leading entries (sampling masks are written at each request's own width).
    Later segments win."""
    field_segments = [segment for segment in segments if segment.field == field]
    if not field_segments:
        return None
    shape = [max(dims) for dims in zip(*(segment.shape for segment in field_segments))]
    out = np.full((hi - lo, *shape), fill, dtype=np.int32)
    fds: dict[str, int] = {}
    try:
        for segment in clip_segments(field_segments, lo, hi):
            if segment.file not in fds:
                fds[segment.file] = os.open(segment.file, os.O_RDONLY)
            data = os.pread(fds[segment.file], segment.rows * segment.row_bytes, segment.offset)
            rows = np.frombuffer(data, dtype=segment.dtype).reshape(segment.rows, *segment.shape)
            out[(slice(segment.pos - lo, segment.end - lo), *(slice(0, dim) for dim in segment.shape))] = rows
    finally:
        for fd in fds.values():
            os.close(fd)
    return out
