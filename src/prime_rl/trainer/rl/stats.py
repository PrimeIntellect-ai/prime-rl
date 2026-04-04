from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class LocalBatchStats:
    num_micro_batches: int
    num_tokens: int
    num_loss_tokens: int
    num_samples: int
    max_micro_batch_tokens: int


def get_local_batch_stats(micro_batches: list[Mapping[str, Any]]) -> LocalBatchStats:
    return LocalBatchStats(
        num_micro_batches=len(micro_batches),
        num_tokens=sum(int(micro_batch["input_ids"].numel()) for micro_batch in micro_batches),
        num_loss_tokens=sum(int(micro_batch["loss_mask"].sum().item()) for micro_batch in micro_batches),
        num_samples=sum(int(micro_batch["sample_count"]) for micro_batch in micro_batches),
        max_micro_batch_tokens=max(int(micro_batch["input_ids"].shape[1]) for micro_batch in micro_batches),
    )


def aggregate_dp_count(
    num_local_value: int,
    *,
    dp_world_size: int,
    dp_group,
    device: torch.device,
) -> int:
    if dp_world_size == 1:
        return num_local_value

    num_value = torch.tensor(num_local_value, device=device, dtype=torch.long)
    dist.all_reduce(num_value, op=dist.ReduceOp.SUM, group=dp_group)
    return int(num_value.item())
