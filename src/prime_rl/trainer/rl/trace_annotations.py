import atexit
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from prime_rl.configs.trainer import TrainerConfig

UPDATE_VERSION = 1


class DisabledTraceAnnotationWriter:
    def export(self, *args: Any, **kwargs: Any) -> None:
        return

    def mark_stable(self) -> None:
        return

    def close(self) -> None:
        return


class TraceAnnotationWriter:
    """Writes the trainer's per-token streams (recomputed logprobs, entropies) as
    verifiers ``TraceUpdate`` JSONL — one record per packed sample, keyed by
    ``(trace_id, branch_index)`` — under ``<output_dir>/trace_annotations/step_<N>/``.
    Streams are full-length over the sample's token prefix so readers can stamp them
    back onto rollout trace nodes without knowing the trainer's loss mask."""

    def __init__(self, output_dir: Path, rank: int) -> None:
        self.rank = rank
        self.output_dir = output_dir / "trace_annotations"
        self._closed = False
        self._initialized_files: set[tuple[int, int]] = set()
        self._pending_stable_dirs: set[Path] = set()
        atexit.register(self.close)

    def export(self, step: int, micro_batch: Mapping[str, Any], model_output: Mapping[str, Tensor]) -> None:
        trace_ids = micro_batch["trace_ids"]
        branch_indices = micro_batch["branch_indices"]
        if not trace_ids or not branch_indices:
            return
        sequence_lengths = micro_batch["sequence_lengths"]
        loss_mask = [bool(v) for v in micro_batch["loss_mask"].detach().cpu().reshape(-1).tolist()]
        env_names = micro_batch["env_names"]
        trainer_logprobs = _tensor_to_floats(model_output["logprobs"])
        entropies = _tensor_to_floats(model_output["entropy"])

        start = 0
        for trace_id, branch_index, length in zip(trace_ids, branch_indices, sequence_lengths):
            span_start, end = start, start + length
            start = end
            if not trace_id or branch_index < 0:
                continue
            # Trailing padding is appended to the last sample and folded into its length.
            while end > span_start and env_names[end - 1] == "" and not loss_mask[end - 1]:
                end -= 1
            if end <= span_start or not any(loss_mask[span_start:end]):
                continue
            logprob_span = trainer_logprobs[span_start:end]
            entropy_span = entropies[span_start:end]
            # After the right shift, a sample's first value crosses the packing boundary.
            logprob_span[0] = None
            entropy_span[0] = None
            self._write(
                step,
                {
                    "version": UPDATE_VERSION,
                    "trace_id": trace_id,
                    "branches": [{"index": branch_index, "trainer_logprobs": logprob_span, "entropies": entropy_span}],
                },
            )

    def close(self) -> None:
        self._closed = True

    def mark_stable(self) -> None:
        # The caller barriers first so a STABLE only lands after every rank flushed.
        while self._pending_stable_dirs:
            stable_dir = self._pending_stable_dirs.pop()
            (stable_dir / "STABLE").touch()

    def _write(self, step: int, record: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError(f"Trace annotation writer is closed for {self.output_dir}")
        step_dir = self.output_dir / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        annotation_file = step_dir / f"rank_{self.rank}.jsonl"
        file_key = (step, self.rank)
        # First write per (step, rank) truncates so a restarted step overwrites stale rows.
        mode = "a" if file_key in self._initialized_files else "w"
        with annotation_file.open(mode, encoding="utf-8") as file:
            file.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        self._initialized_files.add(file_key)
        self._pending_stable_dirs.add(step_dir)


def setup_trace_annotation_writer(
    config: TrainerConfig, parallel_dims: Any, world: Any, logger: Any
) -> TraceAnnotationWriter | DisabledTraceAnnotationWriter:
    if not config.enable_trace_annotations:
        return DisabledTraceAnnotationWriter()
    # CP ranks share the micro batch; only the first writes.
    if parallel_dims.cp_enabled and parallel_dims.world_mesh["cp"].get_local_rank() != 0:
        return DisabledTraceAnnotationWriter()

    writer = TraceAnnotationWriter(config.output_dir, world.rank)
    logger.info(f"Writing trace annotations under {writer.output_dir}")
    return writer


def _tensor_to_floats(tensor: Tensor) -> list[float | None]:
    values = tensor.detach().to(dtype=torch.float32, device="cpu").reshape(-1).tolist()
    return [float(value) if math.isfinite(value) else None for value in values]
