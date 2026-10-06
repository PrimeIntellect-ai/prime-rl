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


def csr_rows(field: str, values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """CSR per-token rows (``counts[i]`` values for token ``i``) as a ``[tokens, widest]``
    array padded with the field's fill."""
    fill = PAYLOAD_FIELDS[field].fill
    width = max(int(counts.max(initial=0)), 1)
    out = np.full((len(counts), width), fill, dtype=np.float32 if isinstance(fill, float) else np.int32)
    out[np.arange(width) < counts[:, None]] = values
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
    segments = clip_segments(field_segments, lo, hi)
    for segment, data in zip(segments, _read_segments(segments)):
        rows = np.frombuffer(data, dtype=segment.dtype).reshape(segment.rows, *segment.shape)
        out[(slice(segment.pos - lo, segment.end - lo), *(slice(0, dim) for dim in segment.shape))] = rows
    return out


# Reads are latency-bound on a shared filesystem (a few rows per turn, spread over many files),
# so the files are read concurrently.
_READ_POOL = ThreadPoolExecutor(32, thread_name_prefix="payload-read")


def _read_segments(segments: list[PayloadSegment]) -> list[bytes]:
    """Each segment's bytes, in order. Each file is opened once."""
    by_file: dict[str, list[int]] = defaultdict(list)
    for i, segment in enumerate(segments):
        by_file[segment.file].append(i)
    data: list[bytes] = [b""] * len(segments)
    files = _READ_POOL.map(_read_file, ([segments[i] for i in indices] for indices in by_file.values()))
    for indices, chunks in zip(by_file.values(), files):
        for i, chunk in zip(indices, chunks):
            data[i] = chunk
    return data


def _read_file(segments: list[PayloadSegment], attempts: int = 5) -> list[bytes]:
    """The segments' bytes from their one file. A shared filesystem can briefly serve a short
    read of a file another node just wrote, so a short read reopens the file and retries with
    backoff."""
    for attempt in range(attempts):
        if attempt:
            time.sleep(0.05 * 2**attempt)
        fd = os.open(segments[0].file, os.O_RDONLY)
        try:
            data = [os.pread(fd, segment.rows * segment.row_bytes, segment.offset) for segment in segments]
        finally:
            os.close(fd)
        short = [(s, len(d)) for s, d in zip(segments, data) if len(d) != s.rows * s.row_bytes]
        if not short:
            return data
    segment, got = short[0]
    raise OSError(
        f"Payload file {segment.file} returned {got} of {segment.rows * segment.row_bytes} bytes at offset "
        f"{segment.offset} after {attempts} reads ({segment.field})"
    )
