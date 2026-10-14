"""Sparse weight deltas for the filesystem weight broadcast.

A delta lives in the ``delta/`` subdir of a broadcast step dir (next to the
full checkpoint on anchor steps): one ``rank_{r}.safetensors`` per trainer rank
and a ``manifest.json``, written last. A rank file stores two zstd streams over its
entries ``(hf_name, count)``: ``gaps`` (uint32 gaps between sorted flat indices,
restarted per entry) and ``values`` (the XOR of the new and the previous raw
bits). XOR is exact for any dtype, but applying it twice reverts it, so a delta
must be applied on top of exactly its ``base_step``.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyzstd
import torch
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

DELTA_DIR = "delta"
MANIFEST = "manifest.json"

_BITS_DTYPE = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def as_bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(_BITS_DTYPE[tensor.element_size()])


def bits_dtype(dtype: str) -> torch.dtype:
    """The integer dtype holding the raw bits of the dtype named ``dtype``."""
    return _BITS_DTYPE[getattr(torch, dtype).itemsize]


def encode_gaps(indices: torch.Tensor, entry_starts: torch.Tensor) -> torch.Tensor:
    """Sorted flat indices -> ``index - previous - 1``, restarting at every
    ``entry_starts`` position (where the gap is the index itself)."""
    previous = torch.empty_like(indices)
    previous[1:] = indices[:-1]
    previous[entry_starts] = -1
    return (indices - previous - 1).to(torch.int32)


def write_rank_file(path: Path, entries: list[tuple[str, int]], gaps: np.ndarray, values: np.ndarray) -> int:
    options = {
        pyzstd.CParameter.compressionLevel: 1,
        pyzstd.CParameter.nbWorkers: 8,
        pyzstd.CParameter.checksumFlag: 1,
    }
    arrays = {
        "gaps": np.frombuffer(pyzstd.compress(gaps.tobytes(), options), np.uint8),
        "values": np.frombuffer(pyzstd.compress(values.tobytes(), options), np.uint8),
    }
    save_file(arrays, str(path), metadata={"entries": json.dumps(entries)})
    return sum(array.nbytes for array in arrays.values())


def decode_gaps(gaps: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Inverse of ``encode_gaps`` for consecutive entries of ``counts`` indices each."""
    indices = torch.cumsum(gaps.long() + 1, 0) - 1
    entry_starts = torch.cumsum(counts, 0) - counts
    restart = torch.where(entry_starts > 0, indices[(entry_starts - 1).clamp(min=0)] + 1, 0)
    return indices - torch.repeat_interleave(restart, counts)


def _read_rank_file(path: Path) -> tuple[list[tuple[str, int]], torch.Tensor, torch.Tensor]:
    with safe_open(str(path), framework="np") as f:
        entries = json.loads(f.metadata()["entries"])
    arrays = load_file(str(path))
    gaps = torch.frombuffer(bytearray(pyzstd.decompress(arrays["gaps"])), dtype=torch.int32)
    values = torch.frombuffer(bytearray(pyzstd.decompress(arrays["values"])), dtype=torch.uint8)
    if torch.cuda.is_available():
        gaps, values = gaps.pin_memory(), values.pin_memory()
    return entries, gaps, values


def read_manifest(step_dir: Path) -> dict:
    return json.loads((step_dir / DELTA_DIR / MANIFEST).read_text())


def read_delta(step_dir: Path) -> list[tuple[list[tuple[str, int]], torch.Tensor, torch.Tensor]]:
    """Read and decompress every rank file of a delta step dir into ``(entries,
    gaps, xor value bytes)`` in (pinned) host memory; ``decode_gaps`` turns the
    gaps into flat indices on the device that applies them."""
    files = read_manifest(step_dir)["files"]
    with ThreadPoolExecutor(len(files)) as pool:
        return list(pool.map(_read_rank_file, [step_dir / DELTA_DIR / name for name in files]))
