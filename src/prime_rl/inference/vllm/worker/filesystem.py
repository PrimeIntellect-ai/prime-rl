import itertools
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.nn import Module
from vllm.logger import init_logger
from vllm.model_executor.model_loader import DefaultModelLoader, get_model_loader

from prime_rl.inference.vllm.worker.weight_transfer import (
    CheckpointIndexMap,
    load_weights_checkpoint_layerwise,
    update_mla_absorbed_weights,
    weight_checksums,
)
from prime_rl.utils.sparse_weights import DELTA_DIR, MANIFEST, bits_dtype, decode_gaps, read_delta, read_manifest

# This is to get type hints for the Worker class but not actually extend it at runtime as this is required by vLLM worker extension
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object

logger = init_logger("vllm.inference.vllm.worker_filesystem")

_prefetch_pool = ThreadPoolExecutor(1)


class FileSystemWeightUpdateWorker(Worker):
    """vLLM worker extension for updating weights in-place using shared filesystem.

    A broadcast step dir holds a full HF checkpoint (anchor), a sparse delta
    against an earlier step, or both. The worker tracks which step it holds and
    applies a delta only on top of its base step; otherwise it reloads the
    chain's anchor and replays the deltas since."""

    weight_step: int | None = None
    prefetched: tuple[str, Future] | None = None
    index_map: CheckpointIndexMap | None = None

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
        self.prefetched = (weight_path, _prefetch_pool.submit(read_delta, Path(weight_path), self._spans()))

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
            if any(step_dir.glob("*.safetensors")):
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
        if prefetched is not None and prefetched[0] == step_dir.as_posix():
            delta = prefetched[1].result()
        else:
            delta = read_delta(step_dir, self._spans())
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
        logger.info(
            f"Applied sparse weight delta {step_dir.name}: {sum(len(gaps) for _, gaps, _ in delta)} changes, "
            f"read {read_time:.2f}s (0 if prefetched), apply {time.perf_counter() - start - read_time:.2f}s"
        )

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
