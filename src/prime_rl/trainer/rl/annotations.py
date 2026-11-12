import asyncio
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

from prime_rl import monitors
from prime_rl.monitors.file.traces.update import make_update


class AnnotationWriter:
    """Collects the trainer's per-token streams (recomputed logprobs, entropies) during
    a step and logs them as trace updates — one record per trained sequence, keyed by
    ``(trace_id, branch_index)``. Streams are full-length over the sample's token prefix
    so readers can fold them onto trace nodes without knowing the trainer's loss mask;
    positions outside that mask hold null, since a fold keeps sampled tokens only and a
    run of nulls costs nothing once the stream is sealed. Streams are float32 arrays with
    NaN for null.

    ``export`` accumulates locally per micro batch; ``flush`` gathers the records to
    rank 0, the only rank running monitors. CP ranks past the first accumulate nothing
    since they share their micro batches, so only the CP leaders take part in the
    gather: their data-parallel group is exactly the leaders, rank 0 among them."""

    def __init__(self, parallel_dims: Any, world: Any, float_decimals: int | None) -> None:
        self.world = world
        self.float_decimals = float_decimals
        self.is_duplicate_rank = parallel_dims.cp_enabled and parallel_dims.world_mesh["cp"].get_local_rank() != 0
        # With pipeline stages the dp group spans one stage only, so gather over the world.
        self.group = None if parallel_dims.pp_enabled else parallel_dims.get_mesh("dp").get_group()
        self._pending: list[tuple[str, int, np.ndarray, np.ndarray]] = []

    def export(self, micro_batch: Mapping[str, Any], model_output: Mapping[str, Tensor]) -> None:
        if self.is_duplicate_rank:
            return
        trace_ids = micro_batch["trace_ids"]
        branch_indices = micro_batch["branch_indices"]
        if not trace_ids or not branch_indices:
            return
        sequence_lengths = micro_batch["sequence_lengths"]
        loss_mask = micro_batch["loss_mask"].detach().cpu().reshape(-1).numpy()
        trainer_logprobs = _masked_floats(model_output["logprobs"], loss_mask, self.float_decimals)
        entropies = _masked_floats(model_output["entropy"], loss_mask, self.float_decimals)

        start = 0
        for index, (trace_id, branch_index, length) in enumerate(zip(trace_ids, branch_indices, sequence_lengths)):
            span_start, end = start, start + length
            start = end
            if not trace_id or branch_index < 0:
                continue
            # Trailing padding is appended to the last sample and folded into its length.
            if index == len(sequence_lengths) - 1:
                end -= micro_batch["padding"]
            if end <= span_start or not loss_mask[span_start:end].any():
                continue
            logprob_span = trainer_logprobs[span_start:end]
            entropy_span = entropies[span_start:end]
            # After the right shift, a sample's first value crosses the packing boundary.
            logprob_span[0] = np.nan
            entropy_span[0] = np.nan
            self._pending.append((trace_id, branch_index, logprob_span, entropy_span))

    def flush(self) -> None:
        """Gather the step's records to rank 0 and log them. Collective over the CP
        leaders (every rank with pipeline stages), so each of them must call it once per
        step; the other CP ranks return at once."""
        streams, self._pending = self._pending, []
        if self.is_duplicate_rank and self.group is not None:
            return
        if dist.is_initialized() and self.world.world_size > 1:
            gathered = gather_streams(streams, self.group)
            if gathered is None:
                return
            streams = gathered
        records = [
            make_update(trace_id, branches={branch_index: {"trainer_logprobs": logprobs, "entropies": entropies}})
            for trace_id, branch_index, logprobs, entropies in streams
        ]
        asyncio.run(monitors.log_annotations(records))


def gather_streams(
    streams: list[tuple[str, int, np.ndarray, np.ndarray]], group: dist.ProcessGroup | None
) -> list[tuple[str, int, np.ndarray, np.ndarray]] | None:
    """Gather ``(trace_id, branch_index, logprobs, entropies)`` streams to the group's
    first rank, in group-rank order; returns None on the other ranks.

    ``gather_object`` would pickle each member's arrays, pad every member to the largest
    and copy each received buffer twice into ``bytes`` on the receiver. Instead only the
    keys travel as objects; the floats travel as one ``(2, n)`` float32 tensor per
    member, which the receiver copies to host once and splits into views."""
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    device = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend(group) == "nccl" else None
    keys = [(trace_id, branch_index, len(logprobs)) for trace_id, branch_index, logprobs, _ in streams]
    flat = (
        np.concatenate([np.stack([logprobs, entropies]) for _, _, logprobs, entropies in streams], axis=1)
        if streams
        else np.empty((2, 0), dtype=np.float32)
    )

    lengths = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(size)]
    dist.all_gather(lengths, torch.tensor([flat.shape[1]], device=device), group=group)
    lengths = [int(length.item()) for length in lengths]
    gathered_keys: list[list[tuple[str, int, int]]] | None = [[] for _ in range(size)] if rank == 0 else None
    dist.gather_object(keys, gathered_keys, group_dst=0, group=group)

    width = max(lengths)
    received = None
    if width:
        # Every member sends the same shape; the padding past its own length is never read.
        send = torch.empty((2, width), dtype=torch.float32, device=device)
        send[:, : flat.shape[1]] = torch.from_numpy(flat)
        received = torch.empty((size, 2, width), dtype=torch.float32, device=device) if rank == 0 else None
        dist.gather(send, list(received.unbind(0)) if received is not None else None, group_dst=0, group=group)
    if rank != 0:
        return None
    if received is None:
        return []

    host = received.cpu().numpy()
    gathered: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    for member, member_keys in enumerate(gathered_keys):
        start = 0
        for trace_id, branch_index, length in member_keys:
            gathered.append(
                (
                    trace_id,
                    branch_index,
                    host[member, 0, start : start + length],
                    host[member, 1, start : start + length],
                )
            )
            start += length
    return gathered


def _masked_floats(tensor: Tensor, mask: np.ndarray, decimals: int | None) -> np.ndarray:
    """Rounded like the record's own logprobs: the streams only ever colour an overlay,
    and full-precision digits are the least compressible bytes a run writes. Masked and
    non-finite positions become NaN, which serializes as null."""
    values = tensor.detach().to(dtype=torch.float32, device="cpu").reshape(-1).numpy()
    if decimals is not None:
        values = values.round(decimals)
    return np.where(mask & np.isfinite(values), values, np.float32(np.nan))
