import itertools
import json
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors import safe_open
from torch.nn import Module
from vllm.distributed.parallel_state import get_ep_group, get_tp_group, get_world_group
from vllm.logger import init_logger
from vllm.model_executor.model_loader import DefaultModelLoader, get_model_loader

from prime_rl.inference.vllm.worker.weight_transfer import (
    CheckpointIndexMap,
    load_weights_checkpoint_layerwise,
    update_mla_absorbed_weights,
    weight_checksums,
)
from prime_rl.utils.sparse_weights import (
    DELTA_DIR,
    ENGINE_LAYOUT,
    MANIFEST,
    anchor_complete,
    bits_dtype,
    decode_gaps,
    drop_page_cache,
    pin,
    read_delta,
    read_manifest,
    unpin,
)

# This is to get type hints for the Worker class but not actually extend it at runtime as this is required by vLLM worker extension
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object

logger = init_logger("vllm.inference.vllm.worker_filesystem")

_prefetch_pool = ThreadPoolExecutor(1)

_SAFETENSORS_DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}


class FileSystemWeightUpdateWorker(Worker):
    """vLLM worker extension for updating weights in-place using shared filesystem.

    A broadcast step dir holds a full HF checkpoint (anchor), a sparse delta
    against an earlier step, or both. The worker tracks which step it holds and
    applies a delta only on top of its base step; otherwise it reloads the
    chain's anchor and replays the deltas since."""

    weight_step: int | None = None
    prefetched: tuple[str, Future] | None = None
    index_map: CheckpointIndexMap | None = None
    wrote_layout = False
    host_buffer: torch.Tensor | None = None

    def liveness_probe(self) -> None:
        """No-op RPC used by the API server liveness endpoint."""
        return None

    def _model(self) -> Module:
        # When enforce_eager=True, model isn't wrapped by torch.compile so no .runnable attr
        model = self.model_runner.model
        if hasattr(model, "runnable"):
            model = model.runnable
        assert isinstance(model, Module)
        return model

    def prefetch_weights_from_path(self, weight_path: str) -> None:
        """Start decoding the sparse delta in ``weight_path`` into pinned host memory."""
        self.prefetched = (weight_path, _prefetch_pool.submit(self._read_delta, Path(weight_path)))

    def _read_delta(
        self, step_dir: Path
    ) -> tuple[list[tuple[list[tuple[str, int]], torch.Tensor, torch.Tensor]], dict[str, int]]:
        """Read this worker's part of a delta into a pinned host buffer that is reused across steps
        (allocating pinned memory every step is slow on a loaded host)."""
        start = time.perf_counter()
        delta, checksums = read_delta(step_dir, None if self.index_map is None else self.index_map.names)
        read_time = time.perf_counter() - start
        num_bytes = sum(-(-gaps.nbytes // 4) * 4 + -(-values.nbytes // 4) * 4 for _, gaps, values in delta)
        if self.host_buffer is None or self.host_buffer.numel() < num_bytes:
            if self.host_buffer is not None:
                unpin(self.host_buffer)
            self.host_buffer = pin(num_bytes * 5 // 4)
        pinned, offset = [], 0
        for entries, gaps, values in delta:
            parts = []
            for array in (gaps, values):
                part = self.host_buffer[offset : offset + array.nbytes].view(array.dtype)
                part.copy_(array)
                parts.append(part)
                offset += -(-array.nbytes // 4) * 4
            pinned.append((entries, *parts))
        logger.info(
            f"Read sparse weight delta {step_dir.name}: {num_bytes / 2**30:.2f} GiB, "
            f"read+decompress {read_time:.2f}s, pin {time.perf_counter() - start - read_time:.2f}s"
        )
        return pinned, checksums

    def prefetch_weights_done(self) -> bool:
        return self.prefetched is None or self.prefetched[1].done()

    def update_weights_from_path(self, weight_path: str) -> None:
        """Move to the broadcast step in ``weight_path`` (a full HF checkpoint and/or a sparse delta)."""
        step_dir = Path(weight_path)
        step = int(step_dir.name.removeprefix("step_"))
        if not (step_dir / DELTA_DIR / MANIFEST).exists():
            self._load_checkpoint(step_dir)
        else:
            base_step = read_manifest(step_dir)["base_step"]
            if base_step is None:
                # The run's first broadcast: the weights the engine booted with.
                if self.weight_step is not None:
                    self._load_checkpoint(self._boot_dir())
            elif base_step == self.weight_step:
                self._apply_delta(step_dir)
                if anchor_complete(step_dir):
                    # Anchor step: check that the delta chain reproduced the full checkpoint bit for bit.
                    chained = weight_checksums(self._model())
                    self._load_checkpoint(step_dir)
                    current = weight_checksums(self._model())
                    mismatched = [name for name, value in current.items() if chained[name] != value]
                    if mismatched:
                        raise RuntimeError(f"Sparse weight deltas diverged from the v{step} anchor: {mismatched[:8]}")
                    logger.info(f"Sparse weight delta chain matches the v{step} anchor")
            else:
                self._replay(step_dir)
        self.weight_step = step
        if (step_dir / DELTA_DIR).is_dir() and not self.wrote_layout:
            self._write_engine_layout(step_dir)
            self.wrote_layout = True

    def _boot_dir(self) -> Path:
        """The local dir of the checkpoint the engine booted from."""
        model_loader = get_model_loader(self.load_config)
        assert isinstance(model_loader, DefaultModelLoader)
        folder, *_ = model_loader._prepare_weights(
            self.model_config.model, None, self.model_config.revision, True, None
        )
        return Path(folder)

    def _apply_chunk(
        self, entries: list[tuple[str, int]], gaps: torch.Tensor, values: torch.Tensor, tensors: dict, device
    ) -> None:
        assert self.index_map is not None
        counts = torch.tensor([count for _, count in entries], device=device)
        indices = decode_gaps(gaps.to(device, non_blocking=True), counts)
        values = values.to(device, non_blocking=True)
        offset = num_bytes = 0
        for bits, group in itertools.groupby(entries, key=lambda entry: bits_dtype(tensors[entry[0]][1])):
            run = list(group)
            num_changes = sum(count for _, count in run)
            itemsize = bits.itemsize
            xor = values[num_bytes : num_bytes + num_changes * itemsize]
            # A run after an odd number of narrower values is misaligned for its dtype view.
            xor = xor.view(bits) if num_bytes % itemsize == 0 else xor.clone().view(bits)
            self.index_map.apply_xor(run, indices[offset : offset + num_changes], xor)
            offset += num_changes
            num_bytes += num_changes * itemsize

    def _write_engine_layout(self, step_dir: Path) -> None:
        """Tell the trainer which checkpoint tensors this engine holds in fp8 (and with which
        scales), so it can send their deltas in the fp8 domain. Reads one checkpoint tensor per fp8
        param from the step's full checkpoint, or from the boot checkpoint."""
        checkpoint_dir = step_dir if anchor_complete(step_dir) else self._boot_dir()
        files = {}
        tensors = {}
        for path in checkpoint_dir.glob("*.safetensors"):
            with safe_open(path, framework="pt") as f:
                for name in f.keys():
                    tensor_slice = f.get_slice(name)
                    files[name] = path
                    tensors[name] = (tensor_slice.get_shape(), _SAFETENSORS_DTYPES[tensor_slice.get_dtype()])
        if (step_dir / DELTA_DIR / MANIFEST).exists():
            tensors = read_manifest(step_dir)["tensors"]
        if self.index_map is None:
            self.index_map = CheckpointIndexMap(self._model(), tensors)

        def read_tensor(name: str) -> torch.Tensor:
            with safe_open(files[name], framework="pt") as f:
                return f.get_tensor(name)

        # Each worker reports the tensors it holds; the trainer waits for all of them.
        if self.vllm_config.parallel_config.enable_expert_parallel:
            group = get_ep_group()
        else:
            group = get_world_group()
        layout = {"world_size": group.world_size, "fp8": self.index_map.fp8_layout(read_tensor)}
        path = step_dir.parent / ENGINE_LAYOUT / f"rank_{group.rank_in_group}.json"
        path.parent.mkdir(exist_ok=True)
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(layout))
        os.replace(tmp, path)
        logger.info(f"Wrote the engine layout: {len(layout['fp8'])} checkpoint tensors held in fp8")

    def _replay(self, step_dir: Path) -> None:
        """Load the latest complete anchor at or before ``step_dir`` (or the boot checkpoint, for a
        chain from the run's first broadcast) and replay the deltas since."""
        chain = []
        while not anchor_complete(step_dir):
            base_step = read_manifest(step_dir)["base_step"]
            if base_step is None:
                break
            chain.append(step_dir)
            step_dir = step_dir.parent / f"step_{base_step}"
        logger.info(f"Replaying {len(chain)} sparse weight deltas on top of {step_dir.name}")
        self._load_checkpoint(step_dir if anchor_complete(step_dir) else self._boot_dir())
        for delta_dir in reversed(chain):
            self._apply_delta(delta_dir)

    def _apply_delta(self, step_dir: Path) -> None:
        start = time.perf_counter()
        prefetched, self.prefetched = self.prefetched, None
        # Always wait: a running prefetch writes into the same host buffer.
        delta = prefetched[1].result() if prefetched is not None else None
        if prefetched is None or prefetched[0] != step_dir.as_posix():
            delta = self._read_delta(step_dir)
        delta, checksums = delta
        read_time = time.perf_counter() - start
        tensors = read_manifest(step_dir)["tensors"]
        model = self._model()
        device = next(model.parameters()).device
        if self.index_map is None:
            self.index_map = CheckpointIndexMap(model, tensors)
            logger.info(f"Built the checkpoint index map in {time.perf_counter() - start - read_time:.2f}s")
        for entries, gaps, values in delta:
            # Decode in chunks of whole entries: the int64 indices take 8 B per change on the GPU.
            first = change_offset = byte_offset = num_changes = 0
            for last in range(len(entries)):
                num_changes += entries[last][1]
                if num_changes < 1 << 26 and last + 1 < len(entries):
                    continue
                chunk = entries[first : last + 1]
                num_bytes = sum(count * bits_dtype(tensors[name][1]).itemsize for name, count in chunk)
                self._apply_chunk(
                    chunk,
                    gaps[change_offset : change_offset + num_changes],
                    values[byte_offset : byte_offset + num_bytes],
                    tensors,
                    device,
                )
                first, change_offset, byte_offset = last + 1, change_offset + num_changes, byte_offset + num_bytes
                num_changes = 0
        update_mla_absorbed_weights(model)
        torch.cuda.synchronize()
        apply_time = time.perf_counter() - start - read_time
        self._verify(step_dir, checksums)
        logger.info(
            f"Applied sparse weight delta {step_dir.name}: {sum(len(gaps) for _, gaps, _ in delta)} changes, "
            f"read {read_time:.2f}s (0 if prefetched), apply {apply_time:.2f}s, "
            f"verify {time.perf_counter() - start - read_time - apply_time:.2f}s"
        )

    def _verify(self, step_dir: Path, checksums: dict[str, int]) -> None:
        """Check every tensor this worker holds against the trainer's checksum of the new weights.
        Tensors split across TP workers are summed over the TP group first."""
        assert self.index_map is not None
        names, totals, partial = self.index_map.held_checksums()
        tp_group = get_tp_group()
        if tp_group.world_size > 1 and partial.any():
            reduced = totals[partial]
            torch.distributed.all_reduce(reduced, group=tp_group.device_group)
            totals[partial] = reduced
        elif partial.any():
            totals[partial] = 0
            names = [name if not is_partial else None for name, is_partial in zip(names, partial.tolist())]
        mismatched = [
            name
            for name, value in zip(names, totals.tolist())
            if name is not None and name in checksums and value != checksums[name]
        ]
        if mismatched:
            raise RuntimeError(f"Sparse weight delta {step_dir.name} diverged from the trainer: {mismatched[:8]}")

    def _load_checkpoint(self, step_dir: Path) -> None:
        """Reload every weight from the HF-compatible checkpoint in ``step_dir``."""
        model = self._model()
        model_loader = get_model_loader(self.load_config)
        assert isinstance(model_loader, DefaultModelLoader)
        local_source = DefaultModelLoader.Source(
            step_dir.as_posix(),
            revision=None,  # TODO: Check that this is correct or if we should use the default (model_config.revision)
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
            allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
        )
        weights_iterator = model_loader._get_weights_iterator(local_source)
        load_weights_checkpoint_layerwise(
            model,
            weights_iterator,
            self.model_runner.model_config,
            self.vllm_config,
        )
        if anchor_complete(step_dir):
            for path in step_dir.glob("*.safetensors"):
                with open(path, "rb") as file:
                    drop_page_cache(file.fileno())
