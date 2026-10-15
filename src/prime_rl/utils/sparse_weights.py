"""Sparse weight deltas for the filesystem weight broadcast.

A delta lives in the ``delta/`` subdir of a broadcast step dir (next to the
full checkpoint on anchor steps): a ``manifest.json``, written first, and per
trainer rank part files ``rank_{r}_{p}.safetensors``, written as the encode
produces them, then a ``rank_{r}.parts`` marker. A part file stores zstd frames over its
entries ``(hf_name, count)``: gaps between sorted flat indices (restarted per
entry) and the differences of the new and the previous raw bits (wrapping, in
the bits dtype; mostly a few ulps), zigzag-encoded. Gaps and differences are
stored as one byte each, with values of 255 or more in an int32 escape stream.
A delta must be applied on top of exactly its ``base_step``; per-tensor
checksums of the new weights catch it if not.
"""

import json
import os
import re
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from itertools import groupby
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


def zigzag(delta: torch.Tensor) -> torch.Tensor:
    """Raw-bit differences (bits dtype, wrapping) -> unsigned values of the same width, small for
    small steps of either sign: 0, -1, 1, -2 -> 0, 1, 2, 3."""
    signed = delta.view(torch.int8) if delta.dtype == torch.uint8 else delta
    return ((signed << 1) ^ (signed >> (8 * signed.element_size() - 1))).view(delta.dtype)


def _unpack_values(data: np.ndarray, entries: list[tuple[str, int]], tensors: dict) -> np.ndarray:
    """The per-change unsigned values of ``entries`` from their raw bytes (each entry in its bits dtype)."""
    out, offset = [], 0
    for itemsize, group in groupby(entries, key=lambda entry: bits_dtype(tensors[entry[0]][1]).itemsize):
        num_bytes = sum(count for _, count in group) * itemsize
        out.append(data[offset : offset + num_bytes].view(f"<u{itemsize}").astype(np.int64))
        offset += num_bytes
    return np.concatenate(out) if out else np.empty(0, np.int64)


def _pack_values(values: np.ndarray, entries: list[tuple[str, int]], tensors: dict) -> np.ndarray:
    """Inverse of ``_unpack_values`` for signed values, wrapping to each entry's bits dtype."""
    out, offset = [], 0
    for itemsize, group in groupby(entries, key=lambda entry: bits_dtype(tensors[entry[0]][1]).itemsize):
        count = sum(count for _, count in group)
        out.append(values[offset : offset + count].astype(f"<i{itemsize}").view(np.uint8))
        offset += count
    return np.concatenate(out) if out else np.empty(0, np.uint8)


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
    readers can skip frames without entries they hold. ``values`` holds each entry's ``zigzag``
    differences in its bits dtype. Gaps and values are stored as uint8; those of 255 or more go to
    the int32 escape stream (gaps first). ``checksums`` (HF name -> ``bits_checksum`` of the new weights
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
        frame_values = _unpack_values(frame_values, entries[first : last + 1], tensors)
        escapes = np.concatenate([frame_gaps[frame_gaps >= 255], frame_values[frame_values >= 255]])
        return (
            pyzstd.compress(np.minimum(frame_gaps, 255).astype(np.uint8).tobytes(), options),
            pyzstd.compress(escapes.astype(np.int32).tobytes(), options),
            pyzstd.compress(np.minimum(frame_values, 255).astype(np.uint8).tobytes(), options),
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


def _read_frame(
    fd: int, frame, tensors: dict, names: set[str] | None
) -> tuple[list[tuple[str, int]], np.ndarray, np.ndarray]:
    """A frame's kept ``(entries, gaps, value bytes)``."""
    frame_entries, ((gaps_at, gaps_size), (escapes_at, escapes_size), (values_at, values_size)) = frame
    gaps = np.frombuffer(pyzstd.decompress(os.pread(fd, gaps_size, gaps_at)), np.uint8).astype(np.int32)
    escapes = np.frombuffer(pyzstd.decompress(os.pread(fd, escapes_size, escapes_at)), np.int32)
    gap_escapes = gaps == 255
    num_gap_escapes = int(gap_escapes.sum())
    gaps[gap_escapes] = escapes[:num_gap_escapes]
    values = np.frombuffer(pyzstd.decompress(os.pread(fd, values_size, values_at)), np.uint8).astype(np.int64)
    values[values == 255] = escapes[num_gap_escapes:].astype(np.int64) & 0xFFFFFFFF
    values = _pack_values((values >> 1) ^ -(values & 1), frame_entries, tensors)
    if names is None or all(name in names for name, _ in frame_entries):
        return frame_entries, gaps, values
    kept, kept_gaps, kept_values, gap_offset, value_offset = [], [], [], 0, 0
    for name, count in frame_entries:
        num_bytes = count * bits_dtype(tensors[name][1]).itemsize
        if name in names:
            kept.append((name, count))
            kept_gaps.append(gaps[gap_offset : gap_offset + count])
            kept_values.append(values[value_offset : value_offset + num_bytes])
        gap_offset += count
        value_offset += num_bytes
    return kept, np.concatenate(kept_gaps), np.concatenate(kept_values)


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


def rank_part(rank: int, part: int) -> str:
    return f"rank_{rank}_{part}.safetensors"


def rank_parts_marker(rank: int) -> str:
    """Written by a trainer rank after its last part file: the number of part files."""
    return f"rank_{rank}.parts"


def read_delta(
    step_dir: Path, names: set[str] | None = None, num_threads: int = 16, cpus: set[int] | None = None
) -> tuple[list[tuple[list[tuple[str, int]], np.ndarray, np.ndarray]], dict[str, int]]:
    """Read and decompress a delta step dir into ``(entries, gaps, value bytes)`` per frame in host
    memory (values: new minus previous raw bits, in each entry's bits dtype); ``decode_gaps`` turns
    the gaps into flat indices on the device that applies them. With ``names``, keep only those
    tensors (frames without any are not read). Part files are read as the trainer writes them, until
    every rank has written its parts marker. ``cpus``: CPUs for the reader threads (the device's NUMA
    node). Also returns the new weights' ``bits_checksum`` per HF tensor, summed over ranks."""
    manifest = read_manifest(step_dir)
    tensors = manifest["tensors"]
    delta_dir = step_dir / DELTA_DIR
    checksums: dict[str, int] = {}
    num_parts: list[int | None] = [None] * manifest["ranks"]
    next_part = [0] * manifest["ranks"]
    initializer = None if cpus is None else (lambda: os.sched_setaffinity(0, cpus))
    with ExitStack() as stack, ThreadPoolExecutor(num_threads, initializer=initializer) as pool:
        files, jobs = [], []
        while any(count is None or next_part[rank] < count for rank, count in enumerate(num_parts)):
            found = False
            for rank in range(len(num_parts)):
                if num_parts[rank] is None and (delta_dir / rank_parts_marker(rank)).exists():
                    num_parts[rank] = int((delta_dir / rank_parts_marker(rank)).read_text())
                path = delta_dir / rank_part(rank, next_part[rank])
                if (num_parts[rank] is None or next_part[rank] < num_parts[rank]) and path.exists():
                    file = stack.enter_context(open(path, "rb"))
                    files.append(file)
                    frames, partial = _rank_file_frames(file, tensors, names)
                    jobs.extend(pool.submit(_read_frame, file.fileno(), frame, tensors, names) for frame in frames)
                    for tensor_name, value in partial.items():
                        checksums[tensor_name] = add_checksums(checksums.get(tensor_name, 0), value)
                    next_part[rank] += 1
                    found = True
            if not found:
                time.sleep(0.02)
        frames = [frame for frame in (job.result() for job in jobs) if frame[0]]
        for file in files:
            drop_page_cache(file.fileno())
    return frames, checksums


def numa_cpus(device: int) -> set[int] | None:
    """The CPUs of the NUMA node the CUDA ``device`` hangs off, or None if unknown."""
    props = torch.cuda.get_device_properties(device)
    sysfs = Path(f"/sys/bus/pci/devices/{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0")
    node = int((sysfs / "numa_node").read_text()) if (sysfs / "numa_node").exists() else -1
    if node < 0:
        return None
    cpus = set()
    for part in Path(f"/sys/devices/system/node/node{node}/cpulist").read_text().strip().split(","):
        first, _, last = part.partition("-")
        cpus.update(range(int(first), int(last or first) + 1))
    return cpus


def pin(num_bytes: int) -> torch.Tensor:
    """Page-locked host memory of exactly ``num_bytes``, outside torch's caching host allocator
    (which rounds sizes up and keeps freed blocks). Release it with ``unpin``."""
    tensor = torch.from_numpy(np.empty(num_bytes, np.uint8))
    if num_bytes:
        error = torch.cuda.cudart().cudaHostRegister(tensor.data_ptr(), num_bytes, 0)
        assert error == 0, f"cudaHostRegister failed: {error}"
    return tensor


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
