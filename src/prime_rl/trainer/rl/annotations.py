import asyncio
import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from prime_rl import monitors
from prime_rl.monitors.file.traces.update import make_update


class AnnotationWriter:
    """Collects the trainer's per-token streams (recomputed logprobs, entropies) during
    a step and logs them as trace updates — one record per trained sequence, keyed by
    ``(trace_id, branch_index)``. Streams are full-length over the sample's token prefix
    so readers can fold them onto trace nodes without knowing the trainer's loss mask.

    ``export`` accumulates locally per micro batch; ``flush`` gathers every rank's
    records to rank 0, the only rank running monitors. CP ranks past the first
    accumulate nothing since they share their micro batches."""

    def __init__(self, parallel_dims: Any, world: Any) -> None:
        self.world = world
        self.is_duplicate_rank = parallel_dims.cp_enabled and parallel_dims.world_mesh["cp"].get_local_rank() != 0
        self._pending: list[dict[str, Any]] = []

    def export(self, micro_batch: Mapping[str, Any], model_output: Mapping[str, Tensor]) -> None:
        if self.is_duplicate_rank:
            return
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
            self._pending.append(
                make_update(
                    trace_id,
                    branches={branch_index: {"trainer_logprobs": logprob_span, "entropies": entropy_span}},
                )
            )

    def flush(self) -> None:
        """Gather the step's records to rank 0 and log them; collective, so every rank
        must call it once per step."""
        records, self._pending = self._pending, []
        if dist.is_initialized() and self.world.world_size > 1:
            gathered: list[list[dict[str, Any]]] | None = None
            if self.world.rank == 0:
                gathered = [[] for _ in range(self.world.world_size)]
            dist.gather_object(records, gathered, dst=0)
            if gathered is None:
                return
            records = [record for rank_records in gathered for record in rank_records]
        asyncio.run(monitors.log_annotations(records))


def _tensor_to_floats(tensor: Tensor) -> list[float | None]:
    values = tensor.detach().to(dtype=torch.float32, device="cpu").reshape(-1).tolist()
    return [float(value) if math.isfinite(value) else None for value in values]
