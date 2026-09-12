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
    """Collects the trainer's per-token streams (recomputed logprobs, entropies, the
    trainer-vs-sampler KL, and loss decisions) during a step and logs them as trace
    updates — one record per trained sequence, keyed by
    ``(trace_id, branch_index)``. Streams are full-length over the sample's token prefix
    so readers can fold them onto trace nodes without knowing the trainer's loss mask;
    positions outside that mask hold null, since a fold keeps sampled tokens only and a
    run of nulls costs nothing once the stream is sealed.

    ``export`` accumulates locally per micro batch; ``flush`` gathers every rank's
    records to rank 0, the only rank running monitors. CP ranks past the first
    accumulate nothing since they share their micro batches."""

    def __init__(self, parallel_dims: Any, world: Any, float_decimals: int | None) -> None:
        self.world = world
        self.float_decimals = float_decimals
        self.is_duplicate_rank = parallel_dims.cp_enabled and parallel_dims.world_mesh["cp"].get_local_rank() != 0
        self._pending: list[dict[str, Any]] = []

    def export(
        self,
        micro_batch: Mapping[str, Any],
        model_output: Mapping[str, Tensor],
        loss_annotations: Mapping[str, Tensor],
    ) -> None:
        if self.is_duplicate_rank:
            return
        trace_ids = micro_batch["trace_ids"]
        branch_indices = micro_batch["branch_indices"]
        if not trace_ids or not branch_indices:
            return
        sequence_lengths = micro_batch["sequence_lengths"]
        loss_mask = [bool(v) for v in micro_batch["loss_mask"].detach().cpu().reshape(-1).tolist()]
        env_names = micro_batch["env_names"]
        trainer_logprobs = _tensor_to_floats(model_output["logprobs"], self.float_decimals)
        entropies = _tensor_to_floats(model_output["entropy"], self.float_decimals)
        is_masked = _tensor_to_optional_bools(loss_annotations.get("is_masked"))
        # The KL is a tiny difference of exponentials, so its stream rounds finer than
        # the logprobs — float_decimals digits would flatten most of them to zero.
        kl_decimals = None if self.float_decimals is None else max(self.float_decimals, 6)
        sampled = _sampled_positions(loss_mask, micro_batch["rl_weights"], micro_batch["ref_kl_weights"])
        mismatch_kl = _mismatch_kl(model_output["logprobs"], micro_batch["inference_logprobs"], sampled, kl_decimals)

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
            trained = loss_mask[span_start:end]
            logprob_span = [v if m else None for v, m in zip(trainer_logprobs[span_start:end], trained)]
            entropy_span = [v if m else None for v, m in zip(entropies[span_start:end], trained)]
            is_masked_span = is_masked[span_start:end] if is_masked is not None else None
            kl_span = [v if m else None for v, m in zip(mismatch_kl[span_start:end], sampled[span_start:end])]
            # After the right shift, a sample's first value crosses the packing boundary.
            logprob_span[0] = None
            entropy_span[0] = None
            kl_span[0] = None
            streams = {"trainer_logprobs": logprob_span, "entropies": entropy_span, "mismatch_kl": kl_span}
            if is_masked_span is not None:
                is_masked_span[0] = None
                streams["is_masked"] = is_masked_span
            self._pending.append(make_update(trace_id, branches={branch_index: streams}))

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


def _tensor_to_floats(tensor: Tensor, decimals: int | None) -> list[float | None]:
    """Rounded like the record's own logprobs: the streams only ever colour an overlay,
    and full-precision digits are the least compressible bytes a run writes."""
    values = tensor.detach().to(dtype=torch.float32, device="cpu").reshape(-1).tolist()
    if decimals is None:
        return [value if math.isfinite(value) else None for value in values]
    return [round(value, decimals) if math.isfinite(value) else None for value in values]


def _sampled_positions(loss_mask: list[bool], rl_weights: Tensor | None, ref_kl_weights: Tensor | None) -> list[bool]:
    """Loss-mask positions whose action the policy sampled, mirroring the trainer's
    own mismatch metric: a ce token is a frozen model's action, so the sampler logprob
    beside it is that model's and the trainer-vs-sampler KL is meaningless there."""
    if rl_weights is None and ref_kl_weights is None:
        return loss_mask
    sampled = [v != 0 for v in rl_weights.detach().cpu().reshape(-1).tolist()] if rl_weights is not None else loss_mask
    if ref_kl_weights is not None:
        for i, v in enumerate(ref_kl_weights.detach().cpu().reshape(-1).tolist()):
            sampled[i] = sampled[i] or v != 0
    return [m and s for m, s in zip(loss_mask, sampled)]


def _mismatch_kl(
    trainer_logprobs: Tensor, inference_logprobs: Tensor, sampled: list[bool], decimals: int | None
) -> list[float | None]:
    """The trainer-vs-sampler KL per token — the same estimate the trainer's own
    mismatch metric logs. Null where the policy did not sample the action, and where
    the estimate overflowed (non-finite values read as null downstream)."""
    log_ratio = trainer_logprobs.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
    log_ratio = log_ratio - inference_logprobs.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
    values = _tensor_to_floats(torch.exp(log_ratio) - log_ratio - 1, decimals)
    return [v if m else None for v, m in zip(values, sampled)]


def _tensor_to_optional_bools(tensor: Tensor | None) -> list[bool | None] | None:
    if tensor is None:
        return None
    return [None if value < 0 else bool(value) for value in tensor.detach().to(device="cpu").reshape(-1).tolist()]
