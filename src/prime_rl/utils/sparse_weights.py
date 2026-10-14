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
from safetensors.numpy import save_file

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
    path: Path, entries: list[tuple[str, int]], gaps: np.ndarray, values: np.ndarray, tensors: dict
) -> int:
    """Write every entry's gaps and values as separate zstd frames, so readers can skip entries."""
    options = {pyzstd.CParameter.compressionLevel: 1, pyzstd.CParameter.checksumFlag: 1}
    counts = [count for _, count in entries]
    value_counts = [count * bits_dtype(tensors[name][1]).itemsize for name, count in entries]
    gap_parts = np.split(gaps, np.cumsum(counts)[:-1])
    value_parts = np.split(values, np.cumsum(value_counts)[:-1])
    with ThreadPoolExecutor(8) as pool:
        gap_frames = list(pool.map(lambda part: pyzstd.compress(part.tobytes(), options), gap_parts))
        value_frames = list(pool.map(lambda part: pyzstd.compress(part.tobytes(), options), value_parts))
    arrays = {
        "gaps": np.frombuffer(b"".join(gap_frames), np.uint8),
        "values": np.frombuffer(b"".join(value_frames), np.uint8),
    }
    frames = [[name, count, len(g), len(v)] for (name, count), g, v in zip(entries, gap_frames, value_frames)]
    save_file(arrays, str(path), metadata={"entries": json.dumps(frames)})
    return sum(array.nbytes for array in arrays.values())


def decode_gaps(gaps: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Inverse of ``encode_gaps`` for consecutive entries of ``counts`` indices each."""
    indices = torch.cumsum(gaps.long() + 1, 0) - 1
    entry_starts = torch.cumsum(counts, 0) - counts
    restart = torch.where(entry_starts > 0, indices[(entry_starts - 1).clamp(min=0)] + 1, 0)
    return indices - torch.repeat_interleave(restart, counts)


def _read_rank_file(
    path: Path, tensors: dict, spans: dict[str, tuple[int, int]] | None
) -> tuple[list[tuple[str, int]], torch.Tensor, torch.Tensor]:
    with safe_open(str(path), framework="np") as f:
        frames = json.loads(f.metadata()["entries"])
        gaps_bytes, values_bytes = f.get_slice("gaps"), f.get_slice("values")
        entries, gaps, values = [], [], []
        gaps_offset = values_offset = 0
        for name, count, num_gap_bytes, num_value_bytes in frames:
            gaps_end, values_end = gaps_offset + num_gap_bytes, values_offset + num_value_bytes
            if spans is None or name in spans:
                entry_gaps = np.frombuffer(pyzstd.decompress(gaps_bytes[gaps_offset:gaps_end].tobytes()), np.int32)
                entry_values = np.frombuffer(
                    pyzstd.decompress(values_bytes[values_offset:values_end].tobytes()), np.uint8
                )
                if spans is not None:
                    indices = np.cumsum(entry_gaps, dtype=np.int64) + np.arange(1, count + 1) - 1
                    begin, end = np.searchsorted(indices, spans[name])
                    itemsize = bits_dtype(tensors[name][1]).itemsize
                    entry_values = entry_values[begin * itemsize : end * itemsize]
                    entry_gaps = entry_gaps[begin:end].copy()
                    if end > begin:
                        entry_gaps[0] = indices[begin]
                if len(entry_gaps):
                    entries.append((name, len(entry_gaps)))
                    gaps.append(entry_gaps)
                    values.append(entry_values)
            gaps_offset, values_offset = gaps_end, values_end
    gaps = torch.from_numpy(np.concatenate(gaps) if gaps else np.empty(0, np.int32))
    values = torch.from_numpy(np.concatenate(values) if values else np.empty(0, np.uint8))
    if torch.cuda.is_available():
        gaps, values = gaps.pin_memory(), values.pin_memory()
    return entries, gaps, values


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
    step_dir: Path, spans: dict[str, tuple[int, int]] | None = None
) -> list[tuple[list[tuple[str, int]], torch.Tensor, torch.Tensor]]:
    """Read and decompress every rank file of a delta step dir into ``(entries,
    gaps, xor value bytes)`` in (pinned) host memory; ``decode_gaps`` turns the
    gaps into flat indices on the device that applies them. With ``spans``, keep
    only the listed tensors and, within each, the flat indices in ``[start, end)``."""
    manifest = read_manifest(step_dir)
    paths = [step_dir / DELTA_DIR / name for name in manifest["files"]]
    with ThreadPoolExecutor(len(paths)) as pool:
        return list(pool.map(lambda path: _read_rank_file(path, manifest["tensors"], spans), paths))
