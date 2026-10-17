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
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from itertools import groupby
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pyzstd
import torch
from safetensors.numpy import save

DELTA_DIR = "delta"
MANIFEST = "manifest.json"
# Dir next to the step dirs where every engine worker writes which checkpoint tensors it holds in fp8.
ENGINE_LAYOUT = "engine_layout"
FP8_BLOCK = [128, 128]
# safetensors dtype -> torch dtype name
SAFETENSORS_DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}

_BITS_DTYPE = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def as_bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(_BITS_DTYPE[tensor.element_size()])


def bits_dtype(dtype: str) -> torch.dtype:
    """The integer dtype holding the raw bits of the dtype named ``dtype``."""
    return _BITS_DTYPE[getattr(torch, dtype).itemsize]


def quantize_fp8_blocks(weight: torch.Tensor, ue8m0: bool, compiled=None) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM's online ``fp8_per_block`` quantization of a 2-D weight: fp8 bits (uint8) and fp32 block
    scales. With ``ue8m0``, also DeepGEMM's requantization to power-of-two scales. Uses vLLM's own
    compiled kernels: a reimplementation rounds differently on a few elements. ``compiled``: the
    ahead-of-time compiled kernels from ``compile_fp8_quantizer`` (columns a multiple of 128), which,
    unlike vLLM's torch.compile'd function, may run on a thread next to a compiled training step."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import requant_weight_ue8m0_inplace
    from vllm.utils.deep_gemm import per_block_cast_to_fp8

    if compiled is None:
        quantized, scales = per_block_cast_to_fp8(weight, FP8_BLOCK, use_ue8m0=False)
        if ue8m0:
            requant_weight_ue8m0_inplace(quantized, scales, FP8_BLOCK)
        return quantized.view(torch.uint8), scales
    cast, cast_ue8m0 = compiled
    quantized, scales = cast(weight)
    if ue8m0:
        # requant_weight_ue8m0_inplace, step by step
        rows, cols = quantized.shape
        expanded = scales.repeat_interleave(FP8_BLOCK[0], 0).repeat_interleave(FP8_BLOCK[1], 1)[:rows, :cols]
        quantized, scales = cast_ue8m0(quantized.to(torch.float32) * expanded)
    return quantized.view(torch.uint8), scales


def compile_fp8_quantizer() -> tuple:
    """vLLM's ``per_block_cast_to_fp8`` compiled ahead of time with AOTInductor for any shape (bf16 input;
    and fp32 input with ue8m0 scales, for the requantization). Matches vLLM's compiled function bit for
    bit for columns a multiple of 128. Compile on the thread that trains: compiling next to a compiled
    training step breaks activation checkpointing there."""
    import tempfile

    from torch.export import Dim
    from vllm.utils.deep_gemm import per_block_cast_to_fp8

    cast = per_block_cast_to_fp8._torchdynamo_orig_callable

    class Cast(torch.nn.Module):
        def __init__(self, ue8m0: bool):
            super().__init__()
            self.ue8m0 = ue8m0

        def forward(self, weight: torch.Tensor):
            return cast(weight, FP8_BLOCK, use_ue8m0=self.ue8m0)

    compiled = []
    for ue8m0, dtype in ((False, torch.bfloat16), (True, torch.float32)):
        example = torch.zeros(384, 640, dtype=dtype, device="cuda")
        program = torch.export.export(Cast(ue8m0), (example,), dynamic_shapes={"weight": {0: Dim.AUTO, 1: Dim.AUTO}})
        path = str(Path(tempfile.mkdtemp()) / "quantize.pt2")
        compiled.append(
            torch._inductor.aoti_load_package(torch._inductor.aoti_compile_and_package(program, package_path=path))
        )
    return tuple(compiled)


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
) -> int:
    """Write a rank's delta as zstd frames of about ``frame_size`` changes each (whole entries), so
    readers can skip frames without entries they hold. ``values`` holds each entry's ``zigzag``
    differences in its bits dtype. Gaps and values are stored as uint8; those of 255 or more go to
    the int32 escape stream (gaps first). ``checksums`` (HF name -> ``bits_checksum`` of the new weights
    this rank holds) let engines verify every step. Returns the file size."""
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
    data = save(arrays, metadata=metadata)
    write_file(path, data)
    return len(data)


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


def write_file(path: Path, data: bytes | Iterable) -> None:
    """Write ``data`` (bytes, or buffers written back to back) under a temporary name, then rename, so
    readers never see a partial file. Plain file I/O releases the GIL during the (possibly slow) write,
    so background writes don't stall training."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with open(tmp, "wb") as file:
        for buffer in [data] if isinstance(data, bytes) else data:
            file.write(buffer)
        drop_page_cache(file.fileno())
    tmp.rename(path)


class AnchorWriter:
    """One trainer rank's share of a full checkpoint: ``(hf_name, shape, flat start, numel, dtype)`` slices,
    stored as tensors ``"{hf_name}@{start}"`` in ~1 GiB safetensors files whose headers are written first, so
    slices can be written in any order, from any thread (``write_all``: in parallel). The files are written next to ``step_dir`` and moved
    into it by ``close``, followed by the ``anchor_rank{rank}.done`` marker (the world size)."""

    def __init__(self, slices: list[tuple[str, list[int], int, int, torch.dtype]], step_dir: Path, rank: int):
        self.step_dir, self.rank = step_dir, rank
        self.tmp_dir = step_dir.with_name(f".{step_dir.name}.anchor{rank}")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.files: list[tuple[str, int]] = []
        self.offsets: dict[tuple[str, int], tuple[int, int]] = {}
        # Slices go round-robin to the files: a file is written at ~0.5 GiB/s, several at once in parallel
        num_files = max(1, -(-sum(numel * dtype.itemsize for *_, numel, dtype in slices) // 2**30))
        groups = [slices[index::num_files] for index in range(num_files)]
        self.pool = ThreadPoolExecutor(8)
        for index, group in enumerate(groups):
            header, offset = {"__metadata__": {"shapes": json.dumps({name: shape for name, shape, *_ in group})}}, 0
            for name, _, start, numel, dtype in group:
                safetensors_dtype = next(
                    key for key, value in SAFETENSORS_DTYPES.items() if getattr(torch, value) == dtype
                )
                header[f"{name}@{start}"] = {
                    "dtype": safetensors_dtype,
                    "shape": [numel],
                    "data_offsets": [offset, offset + numel * dtype.itemsize],
                }
                offset += numel * dtype.itemsize
            encoded = json.dumps(header).encode()
            encoded += b" " * (-len(encoded) % 8)
            file_name = f"model-rank{rank:05d}-{index:05d}.safetensors"
            fd = os.open(self.tmp_dir / file_name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            self._pwrite(fd, len(encoded).to_bytes(8, "little") + encoded, 0)
            for name, _, start, _, _ in group:
                self.offsets[(name, start)] = (fd, 8 + len(encoded) + header[f"{name}@{start}"]["data_offsets"][0])
            self.files.append((file_name, fd))

    @staticmethod
    def _pwrite(fd: int, data, offset: int) -> None:
        data = memoryview(data).cast("B")
        while data:
            written = os.pwrite(fd, data, offset)
            data, offset = data[written:], offset + written

    def write(self, name: str, start: int, data, at: int = 0) -> None:
        """Write ``data`` (bytes) at byte ``at`` of the slice ``"{name}@{start}"``."""
        fd, offset = self.offsets[(name, start)]
        self._pwrite(fd, data, offset + at)

    def write_all(self, items: list[tuple[str, int, Any]]) -> None:
        """``write`` each ``(name, start, data)``, in parallel."""
        for future in [self.pool.submit(self.write, *item) for item in items]:
            future.result()

    def close(self, world_size: int) -> None:
        self.pool.shutdown()
        for name, fd in self.files:
            drop_page_cache(fd)
            os.close(fd)
            (self.tmp_dir / name).rename(self.step_dir / name)
        self.tmp_dir.rmdir()
        write_file(self.step_dir / f"anchor_rank{self.rank}.done", str(world_size).encode())


def write_anchor_shards(
    parts: list[tuple[str, list[int], int, torch.Tensor]],
    step_dir: Path,
    rank: int,
    world_size: int,
    to_host: Callable[[torch.Tensor], Iterable] = lambda values: [values.view(torch.uint8).numpy()],
) -> None:
    """Write one trainer rank's share of a full checkpoint (``AnchorWriter``) from ``(hf_name, shape, flat
    start, flat values)`` slices. ``to_host`` yields each slice's bytes on the host, each written before the
    next is taken."""
    writer = AnchorWriter(
        [(name, shape, start, values.numel(), values.dtype) for name, shape, start, values in parts], step_dir, rank
    )
    for name, _, start, values in parts:
        written = 0
        for chunk in to_host(values):
            writer.write(name, start, chunk, written)
            written += chunk.nbytes
    writer.close(world_size)


def checkpoint_index(directory: Path) -> dict[str, tuple[list[int], torch.dtype, list[tuple[int, int, Path, int]]]]:
    """Tensor name -> ``(shape, dtype, slices)`` of the safetensors files in ``directory``, with slices
    ``(flat start, numel, file, data offset)``: a HF checkpoint (one slice at 0), or an anchor
    (``"{name}@{start}"`` slices, full shapes in the metadata)."""
    index = {}
    for path in sorted(directory.glob("*.safetensors")):
        with open(path, "rb") as file:
            header_size = int.from_bytes(file.read(8), "little")
            header = json.loads(file.read(header_size))
        shapes = json.loads(header.pop("__metadata__", {}).get("shapes", "{}"))
        for key, info in header.items():
            name, _, start = key.partition("@")
            dtype = getattr(torch, SAFETENSORS_DTYPES[info["dtype"]])
            _, _, slices = index.setdefault(name, (shapes.get(name, info["shape"]), dtype, []))
            numel = (info["data_offsets"][1] - info["data_offsets"][0]) // dtype.itemsize
            slices.append((int(start or 0), numel, path, 8 + header_size + info["data_offsets"][0]))
    return index


def read_checkpoint_range(entry: tuple, begin: int, end: int, out: np.ndarray) -> None:
    """Read flat elements ``[begin, end)`` of a ``checkpoint_index`` entry into the bytes ``out``, and drop
    them from the page cache (checkpoints are read once)."""
    _, dtype, slices = entry
    itemsize = dtype.itemsize
    for start, numel, path, data_offset in slices:
        low, high = max(begin, start), min(end, start + numel)
        if low >= high:
            continue
        with open(path, "rb") as file:
            done, size, offset = 0, (high - low) * itemsize, data_offset + (low - start) * itemsize
            target = out[(low - begin) * itemsize : (high - begin) * itemsize]
            while done < size:
                read = os.preadv(file.fileno(), [target[done:]], offset + done)
                assert read > 0, f"short read of {path}"
                done += read
            os.posix_fadvise(file.fileno(), offset, size, os.POSIX_FADV_DONTNEED)


def anchor_complete(step_dir: Path) -> bool:
    """Whether ``step_dir`` holds a complete full checkpoint written by ``write_anchor_shards``."""
    revalidate(step_dir)
    markers = list(step_dir.glob("anchor_rank*.done"))
    return bool(markers) and len(markers) == int(markers[0].read_text())


def read_engine_layout(broadcast_dir: Path) -> dict[str, str] | None:
    """Checkpoint tensor -> fp8 scale format (``"fp32"``/``"ue8m0"``) the engines hold it in,
    or None until every engine worker has reported."""
    revalidate(broadcast_dir / ENGINE_LAYOUT)
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
    """Written by a trainer rank after its last part file: the number of part files and their bytes."""
    return f"rank_{rank}.parts"


def read_delta(
    step_dir: Path, pool: ThreadPoolExecutor, names: set[str] | None = None
) -> tuple[list[tuple[list[tuple[str, int]], np.ndarray, np.ndarray]], dict[str, int]]:
    """Read and decompress a delta step dir into ``(entries, gaps, value bytes)`` per frame in host
    memory (values: new minus previous raw bits, in each entry's bits dtype) on ``pool``; ``decode_gaps``
    turns the gaps into flat indices on the device that applies them. With ``names``, keep only those
    tensors (frames without any are not read). Part files are read as the trainer writes them, until
    every rank has written its parts marker. Also returns the new weights' ``bits_checksum`` per HF
    tensor, summed over ranks."""
    manifest = read_manifest(step_dir)
    tensors = manifest["tensors"]
    delta_dir = step_dir / DELTA_DIR
    checksums: dict[str, int] = {}
    num_parts: list[int | None] = [None] * manifest["ranks"]
    next_part = [0] * manifest["ranks"]
    with ExitStack() as stack:
        files, jobs = [], []
        while any(count is None or next_part[rank] < count for rank, count in enumerate(num_parts)):
            found = False
            revalidate(delta_dir)
            for rank in range(len(num_parts)):
                marker = delta_dir / rank_parts_marker(rank)
                if num_parts[rank] is None and marker.exists():
                    num_parts[rank] = json.loads(marker.read_text())["parts"]
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


def pin(num_bytes: int) -> torch.Tensor:
    """Page-locked host memory of exactly ``num_bytes``, outside torch's caching host allocator
    (which rounds sizes up and keeps freed blocks). Release it with ``unpin``."""
    return page_lock(torch.from_numpy(np.empty(num_bytes, np.uint8)))


def page_lock(tensor: torch.Tensor) -> torch.Tensor:
    """Page-lock a host tensor in place with one ``cudaHostRegister`` (a copy may not span two)."""
    if tensor.numel():
        error = torch.cuda.cudart().cudaHostRegister(tensor.data_ptr(), tensor.nbytes, 0)
        assert error == 0, f"cudaHostRegister failed: {error}"
    return tensor


def unpin(tensor: torch.Tensor) -> None:
    if tensor.numel():
        torch.cuda.cudart().cudaHostUnregister(tensor.data_ptr())


def revalidate(directory: Path) -> None:
    """Make an NFS client see files another host just created in ``directory``: opening a dir (and
    the one above, which may hold ``directory`` itself) revalidates it, instead of answering from
    cached "not found" lookups for up to ~30 s."""
    for path in (directory.parent, directory):
        try:
            os.close(os.open(path, os.O_RDONLY | os.O_DIRECTORY))
        except FileNotFoundError:
            pass


def drop_page_cache(fd: int) -> None:
    """Deltas and anchors are read once; don't let them fill the inference host's page cache."""
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)


def bits_checksum(bits: torch.Tensor, start: int = 0) -> int:
    """Wrapping int64 sum of the raw bits of a flat tensor slice, each weighted by its flat index + 1
    in the full tensor (the slice starts at ``start``): additive over any split of a tensor, so
    trainer ranks and engine workers can each sum the part they hold, and a change written at a
    wrong position is caught."""
    positions = torch.arange(start + 1, start + 1 + bits.numel(), device=bits.device)
    return int((bits.reshape(-1).long() * positions).sum())


def add_checksums(a: int, b: int) -> int:
    return (a + b + 2**63) % 2**64 - 2**63
