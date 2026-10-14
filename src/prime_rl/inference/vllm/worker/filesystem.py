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
    read_delta,
    read_manifest,
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
        delta, checksums = read_delta(step_dir, self._spans())
        read_time = time.perf_counter() - start
        num_bytes = sum(-(-gaps.nbytes // 4) * 4 + -(-values.nbytes // 4) * 4 for _, gaps, values in delta)
        if self.host_buffer is None or self.host_buffer.numel() < num_bytes:
            self.host_buffer = torch.empty(num_bytes * 5 // 4, dtype=torch.uint8, pin_memory=True)
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

    def _spans(self) -> dict[str, tuple[int, int]] | None:
        """The checkpoint index ranges this worker's params hold, once the index map exists."""
        return None if self.index_map is None else self.index_map.spans

    def prefetch_weights_done(self) -> bool:
        return self.prefetched is None or self.prefetched[1].done()

    def update_weights_from_path(self, weight_path: str) -> None:
        """Move to the broadcast step in ``weight_path`` (a full HF checkpoint and/or a sparse delta)."""
        step_dir = Path(weight_path)
        step = int(step_dir.name.removeprefix("step_"))
        if not (step_dir / DELTA_DIR / MANIFEST).exists():
            self._load_checkpoint(step_dir)
        elif read_manifest(step_dir)["base_step"] == self.weight_step:
            self._apply_delta(step_dir)
            if anchor_complete(step_dir):
                # Anchor step: check that the delta chain reproduced the full checkpoint bit for bit.
                chained = weight_checksums(self._model())
                self._load_checkpoint(step_dir)
                mismatched = [name for name, value in weight_checksums(self._model()).items() if chained[name] != value]
                if mismatched:
                    raise RuntimeError(f"Sparse weight deltas diverged from the v{step} anchor: {mismatched[:8]}")
                logger.info(f"Sparse weight delta chain matches the v{step} anchor")
        else:
            self._replay_from_anchor(step_dir)
        self.weight_step = step
        if (step_dir / DELTA_DIR).is_dir() and not self.wrote_layout and anchor_complete(step_dir):
            self._write_engine_layout(step_dir)
            self.wrote_layout = True

    def _write_engine_layout(self, step_dir: Path) -> None:
        """Tell the trainer which checkpoint tensors this engine holds in fp8 (and with which
        scales), so it can send their deltas in the fp8 domain. Needs a full checkpoint."""
        files = {}
        tensors = {}
        for path in step_dir.glob("*.safetensors"):
            with safe_open(path, framework="pt") as f:
                for name in f.keys():
                    tensor_slice = f.get_slice(name)
                    files[name] = path
                    tensors[name] = (tensor_slice.get_shape(), _SAFETENSORS_DTYPES[tensor_slice.get_dtype()])
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

    def _replay_from_anchor(self, step_dir: Path) -> None:
        chain = [step_dir]
        manifest = read_manifest(step_dir)
        while manifest["base_step"] != manifest["anchor_step"]:
            chain.append(step_dir.parent / f"step_{manifest['base_step']}")
            manifest = read_manifest(chain[-1])
        logger.info(f"Replaying {len(chain)} sparse weight deltas on top of the v{manifest['anchor_step']} anchor")
        self._load_checkpoint(step_dir.parent / f"step_{manifest['anchor_step']}")
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
        sums, partial = self.index_map.held_checksums(checksums)
        tp_group = get_tp_group()
        partial_names = sorted(partial) if tp_group.world_size > 1 else []
        if partial_names:
            reduced = torch.stack([sums[name] for name in partial_names])
            torch.distributed.all_reduce(reduced, group=tp_group.device_group)
            sums.update(zip(partial_names, reduced))
        names = [name for name in sums if name not in partial or partial_names]
        values = torch.stack([sums[name] for name in names]).tolist() if names else []
        mismatched = [name for name, value in zip(names, values) if value != checksums[name]]
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
