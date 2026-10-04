"""Per-token side arrays (router-replay ids and weights, sampling masks and their sampler
logprobs) passed by handle: the inference server writes a request's rows to a shared file and
returns ``PayloadSegment``s, the orchestrator clips and shifts the segments into micro-batch
positions, and each trainer rank reads the rows of its window."""

import math
import os
from collections.abc import Iterable, Sequence
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


PAYLOAD_FIELDS = {
    "routed_experts": PayloadField(0, "inputs"),
    "routed_expert_weights": PayloadField(0.0, "inputs"),
    "sampling_mask": PayloadField(-1, "labels"),
    "sampling_mask_logprobs": PayloadField(-math.inf, "sequence"),
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


def pad_rows(field: str, rows: Sequence[Sequence[int | float]]) -> np.ndarray:
    """Ragged per-token rows (one list per token) as a ``[tokens, widest]`` array padded with
    the field's fill."""
    fill = PAYLOAD_FIELDS[field].fill
    out = np.full((len(rows), max(map(len, rows))), fill, dtype=np.float32 if isinstance(fill, float) else np.int32)
    for index, row in enumerate(rows):
        out[index, : len(row)] = row
    return out


def read_field(segments: list[PayloadSegment], field: str, lo: int, hi: int) -> np.ndarray | None:
    """Rows ``[lo, hi)`` of ``field``, as float32 for float fields and int32 otherwise, the
    field's fill where no segment covers a position; None when no segment carries ``field``.
    Rows narrower than the field's widest segment fill the leading entries (ragged fields
    are written at each request's own width). Later segments win."""
    field_segments = [segment for segment in segments if segment.field == field]
    if not field_segments:
        return None
    shape = [max(dims) for dims in zip(*(segment.shape for segment in field_segments))]
    dtype = np.float32 if np.dtype(field_segments[0].dtype).kind == "f" else np.int32
    out = np.full((hi - lo, *shape), PAYLOAD_FIELDS[field].fill, dtype=dtype)
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
