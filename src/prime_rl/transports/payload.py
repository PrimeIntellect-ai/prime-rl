"""Per-token side arrays (router-replay ids, sampling masks) passed by handle: the
inference server writes a request's rows to a shared file and returns ``PayloadSegment``s,
the orchestrator clips and shifts the segments into micro-batch positions, and each
trainer rank reads the rows of its window."""

import errno
import math
import os
import time
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

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
"""Fields whose rows vary in width (a few kept token ids per sampled token), stored as CSR
(``ragged_bytes``): the segment's rows are uint32 start indices, one per row plus one where the
last row ends, of the row's ids in the file. The ids are packed at ``bits`` bits each (see
``id_bits``), and a start index counts ``bits``-bit slots from byte 0 of the file, so clipping
a segment needs no knowledge of the row widths. Other fields store the rows themselves."""


def id_bits(vocab_size: int) -> int:
    """Bits per packed sampling-mask token id: ``ceil(log2(vocab_size))``."""
    return max(1, (vocab_size - 1).bit_length())


def ragged_bytes(counts: np.ndarray, values: np.ndarray, offset: int, bits: int) -> bytes:
    """CSR rows (``counts[i]`` values for token ``i``) laid out to be written at byte ``offset``
    of a file: ``len(counts) + 1`` uint32 start indices, zero padding up to a byte offset that is
    a multiple of ``bits`` (so slot indices are whole numbers), then the values packed
    little-endian at ``bits`` bits each."""
    values = np.asarray(values, dtype=np.int64)
    assert values.size == 0 or (values.min() >= 0 and values.max() < 1 << bits)
    values_offset = offset + 4 * (len(counts) + 1)
    padding = -values_offset % bits
    first = (values_offset + padding) * 8 // bits
    starts = first + np.concatenate([[0], np.cumsum(counts, dtype=np.int64)])
    assert starts[-1] <= np.iinfo(np.uint32).max
    packed = np.packbits((values[:, None] >> np.arange(bits)) & 1, axis=None, bitorder="little")
    return starts.astype(np.uint32).tobytes() + bytes(padding) + packed.tobytes()


def _unpack_ids(data: bytes, first_bit: int, count: int, bits: int) -> np.ndarray:
    """``count`` ``bits``-bit values starting at bit ``first_bit`` of ``data``."""
    flat = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little")
    slots = flat[first_bit : first_bit + count * bits].reshape(count, bits).astype(np.int32)
    return slots @ (1 << np.arange(bits, dtype=np.int32))


def routed_layers(config: Any) -> list[int]:
    """The layers whose experts a learned router picks, from a model's HF config: the only layers
    router replay needs rows for. Dense MLP, Mamba, attention-only (Nemotron-H's
    ``hybrid_override_pattern``) and hash-routed (DeepSeek-V4 ``num_hash_layers``) layers are left
    out. The inference server writes only these layers' rows, and the trainer scatters them back."""
    config = config.get_text_config()
    if (block_types := getattr(config, "layers_block_type", None)) is not None:
        return [i for i, block in enumerate(block_types) if block == "moe"]
    if (pattern := getattr(config, "hybrid_override_pattern", None)) is not None:
        return [i for i, block in enumerate(pattern) if block == "E"]
    if (mlp_types := getattr(config, "mlp_layer_types", None)) is not None:
        return [i for i, mlp in enumerate(mlp_types) if mlp in ("sparse", "moe")]
    first = max(
        getattr(config, name, 0) or 0 for name in ("first_k_dense_replace", "num_dense_layers", "num_hash_layers")
    )
    dense = set(getattr(config, "mlp_only_layers", None) or ())
    step = getattr(config, "decoder_sparse_step", None) or 1
    return [i for i in range(first, config.num_hidden_layers) if i not in dense and (i + 1) % step == 0]


def read_field(segments: list[PayloadSegment], field: str, lo: int, hi: int, fill: int) -> np.ndarray | None:
    """Rows ``[lo, hi)`` of fixed-width ``field`` in the segments' dtype, ``fill`` where no segment
    covers a position; None when no segment carries ``field``. Later segments win."""
    field_segments = [segment for segment in segments if segment.field == field]
    if not field_segments:
        return None
    shape = [max(dims) for dims in zip(*(segment.shape for segment in field_segments))]
    dtype = np.result_type(*(segment.dtype for segment in field_segments))
    out = np.full((hi - lo, *shape), fill, dtype=dtype)
    segments = clip_segments(field_segments, lo, hi)
    for segment, data in zip(segments, _pread([(s.file, s.offset, s.rows * s.row_bytes) for s in segments])):
        rows = np.frombuffer(data, dtype=segment.dtype).reshape(segment.rows, *segment.shape)
        out[(slice(segment.pos - lo, segment.end - lo), *(slice(0, dim) for dim in segment.shape))] = rows
    return out


def read_ragged(
    segments: list[PayloadSegment], field: str, lo: int, hi: int, fill: int, bits: int
) -> np.ndarray | None:
    """Like ``read_field`` for a ragged field packed at ``bits`` bits per id. Rows are padded to
    the widest row of all of the field's segments, as inline masks are padded to the micro
    batch's widest row, so every CP rank gets the same width."""
    segments = [segment for segment in segments if segment.field == field]
    if not segments:
        return None
    segments = [segment for segment in segments if segment.rows]
    # Every segment's start indices (4 bytes per row) set the width; only the window's ids are read.
    starts = [
        np.frombuffer(data, dtype=np.uint32).astype(np.int64)
        for data in _pread([(s.file, s.offset, 4 * (s.rows + 1)) for s in segments])
    ]
    width = max([int(np.diff(s).max()) for s in starts] + [1])
    out = np.full((hi - lo, width), fill, dtype=np.int32)
    window = []
    for segment, segment_starts in zip(segments, starts):
        a, b = max(segment.pos, lo) - segment.pos, min(segment.end, hi) - segment.pos
        if a < b:
            window.append((segment.file, segment.pos + a - lo, segment_starts[a : b + 1]))
    first_bytes = [int(w[0]) * bits // 8 for _, _, w in window]
    values = _pread(
        [(file, byte, -(-int(w[-1]) * bits // 8) - byte) for (file, _, w), byte in zip(window, first_bytes)]
    )
    for (_, row, w), byte, data in zip(window, first_bytes, values):
        counts = np.diff(w)
        rows = out[row : row + len(counts)]
        rows[:] = fill
        ids = _unpack_ids(data, int(w[0]) * bits - 8 * byte, int(counts.sum()), bits)
        rows[np.arange(width) < counts[:, None]] = ids
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
    serve a stale lookup (``ENOENT``, ``ESTALE``) or a short read of a file another node just
    wrote, so either reopens the file and retries with backoff."""
    file = reads[0][0]
    for attempt in range(attempts):
        if attempt:
            time.sleep(0.05 * 2**attempt)
        try:
            fd = os.open(file, os.O_RDONLY)
            try:
                data = [os.pread(fd, size, offset) for _, offset, size in reads]
            finally:
                os.close(fd)
        except OSError as error:
            if error.errno not in (errno.ENOENT, errno.ESTALE) or attempt == attempts - 1:
                raise
            continue
        short = [(offset, size, len(chunk)) for (_, offset, size), chunk in zip(reads, data) if len(chunk) != size]
        if not short:
            return data
    offset, size, got = short[0]
    raise OSError(f"Payload file {file} returned {got} of {size} bytes at offset {offset} after {attempts} reads")
