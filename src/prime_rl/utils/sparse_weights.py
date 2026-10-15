"""Sparse weight deltas for the filesystem weight broadcast.

A delta lives in the ``delta/`` subdir of a broadcast step dir (next to the
full checkpoint on anchor steps): one ``rank_{r}.safetensors`` per trainer rank
and a ``manifest.json``, written last. A rank file stores zstd frames over its
entries ``(hf_name, count)``: gaps between sorted flat indices (restarted per
entry) and the XOR of the new and the previous raw bits. XOR is exact for any
dtype, but applying it twice reverts it, so a delta must be applied on top of
exactly its ``base_step``; per-tensor checksums of the new weights catch it if not.
"""

import json
import os
import re
import threading
from collections import defaultdict
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from math import prod
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pyzstd
import torch
from safetensors import safe_open
from safetensors.numpy import save

DELTA_DIR = "delta"
MANIFEST = "manifest.json"
# Dir next to the step dirs where every engine worker writes which checkpoint tensors it holds in fp8.
ENGINE_LAYOUT = "engine_layout"
FP8_BLOCK = [128, 128]

_BITS_DTYPE = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def as_bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(_BITS_DTYPE[tensor.element_size()])


def bits_dtype(dtype: str) -> torch.dtype:
    """The integer dtype holding the raw bits of the dtype named ``dtype``."""
    return _BITS_DTYPE[getattr(torch, dtype).itemsize]


def quantize_fp8_blocks(weight: torch.Tensor, ue8m0: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM's online ``fp8_per_block`` quantization of a 2-D weight: fp8 bits (uint8) and fp32 block
    scales. With ``ue8m0``, also DeepGEMM's requantization to power-of-two scales. Uses vLLM's own
    compiled kernels: a reimplementation rounds differently on a few elements."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import requant_weight_ue8m0_inplace
    from vllm.utils.deep_gemm import per_block_cast_to_fp8

    quantized, scales = per_block_cast_to_fp8(weight, FP8_BLOCK, use_ue8m0=False)
    if ue8m0:
        requant_weight_ue8m0_inplace(quantized, scales, FP8_BLOCK)
    return quantized.view(torch.uint8), scales


def encode_gaps(indices: torch.Tensor, entry_starts: torch.Tensor) -> torch.Tensor:
    """Sorted flat indices -> ``index - previous - 1``, restarting at every
    ``entry_starts`` position (where the gap is the index itself)."""
    previous = torch.empty_like(indices)
    previous[1:] = indices[:-1]
    previous[entry_starts] = -1
    return (indices - previous - 1).to(torch.int32)


def write_rank_file(
    path: Path,
    entries: list[tuple[str, int]],
    gaps: np.ndarray,
    values: np.ndarray,
    tensors: dict,
    checksums: dict[str, int],
    frame_size: int = 1 << 20,
) -> None:
    """Write a rank's delta as zstd frames of about ``frame_size`` changes each (whole entries), so
    readers can skip frames without entries they hold. Gaps are stored as uint8, with gaps of 255 or
    more in a separate int32 stream. ``checksums`` (HF name -> ``bits_checksum`` of the new weights
    this rank holds) let engines verify every step. Written under a temporary name, then renamed."""
    options = {pyzstd.CParameter.compressionLevel: 1, pyzstd.CParameter.checksumFlag: 1}
    frames, size = [[]], 0
    for index, (_, count) in enumerate(entries):
        if size >= frame_size:
            frames.append([])
            size = 0
        frames[-1].append(index)
        size += count
    gap_ends = np.cumsum([count for _, count in entries], dtype=np.int64)
    value_ends = np.cumsum([count * bits_dtype(tensors[name][1]).itemsize for name, count in entries], dtype=np.int64)

    def compress(frame: list[int]) -> tuple[bytes, bytes, bytes]:
        first, last = frame[0], frame[-1]
        frame_gaps = gaps[(gap_ends[first - 1] if first else 0) : gap_ends[last]]
        frame_values = values[(value_ends[first - 1] if first else 0) : value_ends[last]]
        return (
            pyzstd.compress(np.minimum(frame_gaps, 255).astype(np.uint8).tobytes(), options),
            pyzstd.compress(frame_gaps[frame_gaps >= 255].tobytes(), options),
            pyzstd.compress(frame_values.tobytes(), options),
        )

    # Few threads: this runs in the background next to training (e.g. a CPU optimizer step)
    with ThreadPoolExecutor(4) as pool:
        compressed = list(pool.map(compress, frames if entries else []))
    arrays = {
        stream: np.frombuffer(b"".join(parts[index] for parts in compressed), np.uint8)
        for index, stream in enumerate(("gaps", "escapes", "values"))
    }
    metadata = {
        "entries": json.dumps(entries),
        "frames": json.dumps([[len(frame), *map(len, parts)] for frame, parts in zip(frames, compressed)]),
        "checksums": json.dumps(checksums),
    }
    _write_atomically(path, save(arrays, metadata=metadata))


def decode_gaps(gaps: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Inverse of ``encode_gaps`` for consecutive entries of ``counts`` indices each."""
    indices = torch.cumsum(gaps.long() + 1, 0) - 1
    entry_starts = torch.cumsum(counts, 0) - counts
    restart = torch.where(entry_starts > 0, indices[(entry_starts - 1).clamp(min=0)] + 1, 0)
    return indices - torch.repeat_interleave(restart, counts)


def _rank_file_frames(
    file: BinaryIO, tensors: dict, names: set[str] | None
) -> tuple[list[tuple[list[tuple[str, int]], list[tuple[int, int]]]], dict[str, int]]:
    """The frames of an open rank file with entries in ``names``: ``(entries, [(offset, size)] of the
    gaps, escapes and values streams)``, and the file's checksums."""
    header_size = int.from_bytes(file.read(8), "little")
    header = json.loads(file.read(header_size))
    metadata = header["__metadata__"]
    offsets = [8 + header_size + header[stream]["data_offsets"][0] for stream in ("gaps", "escapes", "values")]
    entries = [tuple(entry) for entry in json.loads(metadata["entries"])]
    frames, start = [], 0
    for num_entries, *sizes in json.loads(metadata["frames"]):
        frame_entries = entries[start : start + num_entries]
        if names is None or any(name in names for name, _ in frame_entries):
            frames.append((frame_entries, list(zip(offsets, sizes))))
        start += num_entries
        offsets = [offset + size for offset, size in zip(offsets, sizes)]
    return frames, json.loads(metadata["checksums"])


def _read_frame(fd: int, frame, tensors: dict, names: set[str] | None) -> list[tuple[str, np.ndarray, np.ndarray]]:
    frame_entries, ((gaps_at, gaps_size), (escapes_at, escapes_size), (values_at, values_size)) = frame
    gaps = np.frombuffer(pyzstd.decompress(os.pread(fd, gaps_size, gaps_at)), np.uint8).astype(np.int32)
    gaps[gaps == 255] = np.frombuffer(pyzstd.decompress(os.pread(fd, escapes_size, escapes_at)), np.int32)
    values = np.frombuffer(pyzstd.decompress(os.pread(fd, values_size, values_at)), np.uint8)
    out, gap_offset, value_offset = [], 0, 0
    for name, count in frame_entries:
        num_bytes = count * bits_dtype(tensors[name][1]).itemsize
        if names is None or name in names:
            out.append((name, gaps[gap_offset : gap_offset + count], values[value_offset : value_offset + num_bytes]))
        gap_offset += count
        value_offset += num_bytes
    return out


def _write_atomically(path: Path, data: bytes) -> None:
    """Write under a temporary name, then rename. Serializing in memory and writing with plain file
    I/O releases the GIL during the (possibly slow) write, so background writes don't stall training."""
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "wb") as file:
        file.write(data)
        drop_page_cache(file.fileno())
    tmp.rename(path)


def write_anchor_shards(
    parts: list[tuple[str, list[int], int, torch.Tensor]], step_dir: Path, rank: int, world_size: int
) -> None:
    """Write one trainer rank's share of a full checkpoint: ``(hf_name, shape, flat start, flat values)``
    slices, stored as tensors ``"{hf_name}@{start}"`` in ~1 GiB safetensors files, straight from host
    memory and without collectives. The ``anchor_rank{rank}.done`` marker (the world size) comes last."""
    files, size = [[]], 0
    for part in parts:
        if size > 2**30:
            files.append([])
            size = 0
        files[-1].append(part)
        size += part[3].nbytes
    for index, file_parts in enumerate(files):
        header, offset = {"__metadata__": {"shapes": json.dumps({name: shape for name, shape, _, _ in file_parts})}}, 0
        for name, _, start, values in file_parts:
            dtype = _SAFETENSORS_DTYPES[values.dtype]
            header[f"{name}@{start}"] = {
                "dtype": dtype,
                "shape": [values.numel()],
                "data_offsets": [offset, offset + values.nbytes],
            }
            offset += values.nbytes
        encoded = json.dumps(header).encode()
        encoded += b" " * (-len(encoded) % 8)
        path = step_dir / f"model-rank{rank:05d}-{index:05d}.safetensors"
        tmp = path.with_name(f".{path.name}.tmp")
        with open(tmp, "wb") as file:
            file.write(len(encoded).to_bytes(8, "little") + encoded)
            for *_, values in file_parts:
                file.write(memoryview(values.view(torch.uint8).numpy()))
            drop_page_cache(file.fileno())
        tmp.rename(path)
    (step_dir / f"anchor_rank{rank}.done").write_text(str(world_size))


_SAFETENSORS_DTYPES = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32"}


def anchor_index(step_dir: Path) -> tuple[dict[str, list[tuple[Path, str, int]]], dict[str, tuple[list[int], str]]]:
    """Where each tensor's slices are in an anchor written by ``write_anchor_shards``, and its shape and dtype."""
    slices: dict[str, list[tuple[Path, str, int]]] = defaultdict(list)
    tensors: dict[str, tuple[list[int], str]] = {}
    for path in sorted(step_dir.glob("model-rank*.safetensors")):
        with safe_open(path, framework="pt") as file:
            shapes = json.loads(file.metadata()["shapes"])
            for key in file.keys():
                name, start = key.rsplit("@", 1)
                slices[name].append((path, key, int(start)))
                dtype = str(file.get_slice(key).get_dtype())
                tensors[name] = (shapes[name], {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}[dtype])
    return dict(slices), tensors


def _numeric_order(name: str) -> list:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def anchor_tensors(
    step_dir: Path, names: Iterable[str] | None = None, index=None
) -> Iterator[tuple[str, torch.Tensor]]:
    """Assemble the full tensors of an anchor (all, or ``names``)."""
    slices, tensors = index or anchor_index(step_dir)
    # Layer by layer (numeric order), so the engine's layerwise reload holds few layers at a time
    for name in sorted(slices, key=_numeric_order) if names is None else names:
        shape, dtype = tensors[name]
        tensor = torch.empty(prod(shape), dtype=getattr(torch, dtype))
        for path, key, start in slices[name]:
            with safe_open(path, framework="pt") as file:
                values = file.get_tensor(key)
            tensor[start : start + values.numel()] = values
        yield name, tensor.view(shape)
    if names is None:
        for path in step_dir.glob("model-rank*.safetensors"):
            with open(path, "rb") as file:
                drop_page_cache(file.fileno())


def anchor_complete(step_dir: Path) -> bool:
    """Whether ``step_dir`` holds a complete full checkpoint written by ``write_anchor_shards``."""
    markers = list(step_dir.glob("anchor_rank*.done"))
    return bool(markers) and len(markers) == int(markers[0].read_text())


def read_engine_layout(broadcast_dir: Path) -> dict[str, str] | None:
    """Checkpoint tensor -> fp8 scale format (``"fp32"``/``"ue8m0"``) the engines hold it in,
    or None until every engine worker has reported."""
    layouts = [json.loads(path.read_text()) for path in (broadcast_dir / ENGINE_LAYOUT).glob("*.json")]
    if not layouts or len(layouts) < layouts[0]["world_size"]:
        return None
    fp8 = {}
    for layout in layouts:
        for name, mode in layout["fp8"].items():
            if fp8.setdefault(name, mode) != mode:
                raise NotImplementedError(f"Engine workers hold {name} in different fp8 formats")
    return fp8


def read_manifest(step_dir: Path) -> dict:
    return json.loads((step_dir / DELTA_DIR / MANIFEST).read_text())


def read_delta(
    step_dir: Path, names: set[str] | None = None, num_threads: int = 16
) -> tuple[list[tuple[list[tuple[str, int]], torch.Tensor, torch.Tensor]], dict[str, int]]:
    """Read and decompress every rank file of a delta step dir into ``(entries,
    gaps, xor value bytes)`` in host memory; ``decode_gaps`` turns the
    gaps into flat indices on the device that applies them. With ``names``, keep
    only those tensors (frames without any are not read).
    Also returns the new weights' ``bits_checksum`` per HF tensor, summed over ranks."""
    manifest = read_manifest(step_dir)
    tensors = manifest["tensors"]
    checksums: dict[str, int] = {}
    with ExitStack() as stack, ThreadPoolExecutor(num_threads) as pool:
        files = [stack.enter_context(open(step_dir / DELTA_DIR / name, "rb")) for name in manifest["files"]]
        jobs = []
        for file in files:
            frames, partial = _rank_file_frames(file, tensors, names)
            jobs.extend((file.fileno(), frame) for frame in frames)
            for tensor_name, value in partial.items():
                checksums[tensor_name] = add_checksums(checksums.get(tensor_name, 0), value)
        kept = [entry for frame in pool.map(lambda job: _read_frame(*job, tensors, names), jobs) for entry in frame]
        for file in files:
            drop_page_cache(file.fileno())
    gaps = torch.from_numpy(np.concatenate([g for _, g, _ in kept]) if kept else np.empty(0, np.int32))
    values = torch.from_numpy(np.concatenate([v for _, _, v in kept]) if kept else np.empty(0, np.uint8))
    return [([(name, len(g)) for name, g, _ in kept], gaps, values)], checksums


def pin(num_bytes: int) -> torch.Tensor:
    """Page-locked host memory of exactly ``num_bytes``, outside torch's caching host allocator
    (which rounds sizes up and keeps freed blocks). Release it with ``unpin``."""
    tensor = torch.from_numpy(np.empty(num_bytes, np.uint8))
    if num_bytes:
        error = torch.cuda.cudart().cudaHostRegister(tensor.data_ptr(), num_bytes, 0)
        assert error == 0, f"cudaHostRegister failed: {error}"
    return tensor


def pin_in_background(num_bytes: int) -> tuple[torch.Tensor, threading.Thread]:
    """Host memory of ``num_bytes`` and a (not yet started) thread that page-locks it with one
    ``cudaHostRegister`` (a copy may not span two registrations). Page-locking a large buffer is slow
    on a loaded host. No CUDA copies may touch the memory while the thread runs."""
    tensor = torch.from_numpy(np.empty(num_bytes, np.uint8))

    def register() -> None:
        if num_bytes:
            error = torch.cuda.cudart().cudaHostRegister(tensor.data_ptr(), num_bytes, 0)
            assert error == 0, f"cudaHostRegister failed: {error}"

    return tensor, threading.Thread(target=register)


def unpin(tensor: torch.Tensor) -> None:
    if tensor.numel():
        torch.cuda.cudart().cudaHostUnregister(tensor.data_ptr())


def drop_page_cache(fd: int) -> None:
    """Deltas and anchors are read once; don't let them fill the inference host's page cache."""
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)


def bits_checksum(bits: torch.Tensor) -> int:
    """Wrapping int64 sum of raw bits: additive over any split of a tensor, so trainer
    ranks and engine workers can each sum the part they hold."""
    return int(bits.sum(dtype=torch.int64))


def add_checksums(a: int, b: int) -> int:
    return (a + b + 2**63) % 2**64 - 2**63
