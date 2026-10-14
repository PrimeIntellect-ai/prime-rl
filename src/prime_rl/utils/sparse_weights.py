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
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyzstd
import torch
from safetensors.numpy import save_file
from safetensors.torch import save_file as save_torch_file

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
) -> int:
    """Write a rank's delta as zstd frames of about ``frame_size`` changes each (whole entries), so
    readers can skip frames without entries they hold. ``checksums`` (HF name -> ``bits_checksum`` of
    the new weights this rank holds) let engines verify every step."""
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

    def compress(frame: list[int]) -> tuple[bytes, bytes]:
        first, last = frame[0], frame[-1]
        gap_start = gap_ends[first - 1] if first else 0
        value_start = value_ends[first - 1] if first else 0
        return (
            pyzstd.compress(gaps[gap_start : gap_ends[last]].tobytes(), options),
            pyzstd.compress(values[value_start : value_ends[last]].tobytes(), options),
        )

    with ThreadPoolExecutor(8) as pool:
        compressed = list(pool.map(compress, frames if entries else []))
    arrays = {
        "gaps": np.frombuffer(b"".join(g for g, _ in compressed), np.uint8),
        "values": np.frombuffer(b"".join(v for _, v in compressed), np.uint8),
    }
    metadata = {
        "entries": json.dumps(entries),
        "frames": json.dumps([[len(frame), len(g), len(v)] for frame, (g, v) in zip(frames, compressed)]),
        "checksums": json.dumps(checksums),
    }
    save_file(arrays, str(path), metadata=metadata)
    return sum(array.nbytes for array in arrays.values())


def decode_gaps(gaps: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Inverse of ``encode_gaps`` for consecutive entries of ``counts`` indices each."""
    indices = torch.cumsum(gaps.long() + 1, 0) - 1
    entry_starts = torch.cumsum(counts, 0) - counts
    restart = torch.where(entry_starts > 0, indices[(entry_starts - 1).clamp(min=0)] + 1, 0)
    return indices - torch.repeat_interleave(restart, counts)


def _read_rank_file(
    path: Path, tensors: dict, names: set[str] | None
) -> tuple[list[tuple[str, int]], torch.Tensor, torch.Tensor, dict[str, int]]:
    with open(path, "rb") as file:
        header_size = int.from_bytes(file.read(8), "little")
        header = json.loads(file.read(header_size))
        metadata = header["__metadata__"]
        gaps_start = 8 + header_size + header["gaps"]["data_offsets"][0]
        values_start = 8 + header_size + header["values"]["data_offsets"][0]
        entries = [tuple(entry) for entry in json.loads(metadata["entries"])]
        frames, start, gaps_offset, values_offset = [], 0, gaps_start, values_start
        for num_entries, num_gap_bytes, num_value_bytes in json.loads(metadata["frames"]):
            frame_entries = entries[start : start + num_entries]
            if names is None or any(name in names for name, _ in frame_entries):
                frames.append((frame_entries, (gaps_offset, num_gap_bytes), (values_offset, num_value_bytes)))
            start += num_entries
            gaps_offset += num_gap_bytes
            values_offset += num_value_bytes

        def read_frame(frame) -> list[tuple[str, np.ndarray, np.ndarray]]:
            frame_entries, (gap_at, gap_size), (value_at, value_size) = frame
            frame_gaps = np.frombuffer(pyzstd.decompress(os.pread(file.fileno(), gap_size, gap_at)), np.int32)
            frame_values = np.frombuffer(pyzstd.decompress(os.pread(file.fileno(), value_size, value_at)), np.uint8)
            out, gap_offset, value_offset = [], 0, 0
            for name, count in frame_entries:
                itemsize = bits_dtype(tensors[name][1]).itemsize
                entry_gaps = frame_gaps[gap_offset : gap_offset + count]
                entry_values = frame_values[value_offset : value_offset + count * itemsize]
                gap_offset += count
                value_offset += count * itemsize
                if names is not None and name not in names:
                    continue
                if len(entry_gaps):
                    out.append((name, entry_gaps, entry_values))
            return out

        with ThreadPoolExecutor(4) as pool:
            kept = [entry for frame in pool.map(read_frame, frames) for entry in frame]
        drop_page_cache(file.fileno())
    gaps = torch.from_numpy(np.concatenate([g for _, g, _ in kept]) if kept else np.empty(0, np.int32))
    values = torch.from_numpy(np.concatenate([v for _, _, v in kept]) if kept else np.empty(0, np.uint8))
    return [(name, len(g)) for name, g, _ in kept], gaps, values, json.loads(metadata["checksums"])


def write_anchor_shards(state_dict: dict[str, torch.Tensor], step_dir: Path, rank: int, world_size: int) -> None:
    """Write one trainer rank's share of a full checkpoint, without collectives; the
    ``anchor_rank{rank}.done`` marker (holding the world size) is written last."""
    shards, size = [[]], 0
    for name, tensor in state_dict.items():
        if size > 4 * 2**30:
            shards.append([])
            size = 0
        shards[-1].append(name)
        size += tensor.nbytes
    for index, names in enumerate(shards):
        path = step_dir / f"model-rank{rank:05d}-{index:05d}.safetensors"
        tmp = step_dir / f".{path.name}.tmp"
        save_torch_file({name: state_dict[name].contiguous() for name in names}, tmp, metadata={"format": "pt"})
        tmp.rename(path)
    (step_dir / f"anchor_rank{rank}.done").write_text(str(world_size))


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
    step_dir: Path, names: set[str] | None = None
) -> tuple[list[tuple[list[tuple[str, int]], torch.Tensor, torch.Tensor]], dict[str, int]]:
    """Read and decompress every rank file of a delta step dir into ``(entries,
    gaps, xor value bytes)`` in host memory; ``decode_gaps`` turns the
    gaps into flat indices on the device that applies them. With ``names``, keep
    only those tensors (frames without any are not read).
    Also returns the new weights' ``bits_checksum`` per HF tensor, summed over ranks."""
    manifest = read_manifest(step_dir)
    paths = [step_dir / DELTA_DIR / name for name in manifest["files"]]
    with ThreadPoolExecutor(len(paths)) as pool:
        files = list(pool.map(lambda path: _read_rank_file(path, manifest["tensors"], names), paths))
    checksums: dict[str, int] = {}
    for *_, partial in files:
        for name, value in partial.items():
            checksums[name] = add_checksums(checksums.get(name, 0), value)
    return [delta for *delta, _ in files], checksums


def drop_page_cache(fd: int) -> None:
    """Deltas and anchors are read once; don't let them fill the inference host's page cache."""
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)


def bits_checksum(bits: torch.Tensor) -> int:
    """Wrapping int64 sum of raw bits: additive over any split of a tensor, so trainer
    ranks and engine workers can each sum the part they hold."""
    return int(bits.sum(dtype=torch.int64))


def add_checksums(a: int, b: int) -> int:
    return (a + b + 2**63) % 2**64 - 2**63
