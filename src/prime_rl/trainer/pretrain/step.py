"""The pretrain loop's forward and backward passes over one step's micro-batches, with the step's statistics
reduced once."""

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor, nn

from prime_rl.trainer.model import is_tt_moe_model, iter_moe_blocks
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.pretrain.metrics import expert_load_metrics, moe_layer_indices, reduce_stats, take_routing_counts
from prime_rl.trainer.utils import scale_gradients_


@dataclass
class StepResult:
    loss: float
    """Mean loss per unmasked token over the global batch."""

    nan_count: int
    """Micro-batches with a NaN loss over all ranks. They are left out of the backward pass."""

    num_tokens: int
    """Input tokens of the global batch."""

    tokens_per_expert: Tensor | None
    """Global ``[num_moe_layers, num_experts]`` token counts of the step (fp32, on the GPU). None for dense models."""

    metrics: dict[str, float]
    """Statistics of the step over all ranks: data wait (max and mean over ranks), and for MoE
    models the expert balance of the global counts (`expert_load_metrics`), the busiest expert of one rank's
    micro-batch over the mean (``expert_load/micro_max_mean``, mean and max over ranks and micro-batches) and the
    routing confidence."""

    sums: dict[str, float]
    """The caller's ``extra_sums``, summed over the ranks."""


def run_micro_batches(
    model: nn.Module,
    dataiter: Iterator[dict],
    compute_loss: Callable[[dict], tuple[Tensor, Tensor, Tensor]],
    grad_accum_steps: int,
    parallel_dims: ParallelDims,
    *,
    extra_sums: dict[str, Tensor] | None = None,
) -> StepResult:
    """Forward and backward passes over the ``grad_accum_steps`` micro-batches of one step, leaving the
    gradients of the summed token loss divided by the global token count (the mean loss per token, as on one
    GPU without accumulation). ``compute_loss`` returns the summed loss and the number of unmasked tokens of a
    micro-batch, and an auxiliary loss on the same scale as the summed loss that is trained but not logged.

    The step's statistics stay on the GPU during the micro-batches (no collective, no host sync) and are reduced
    with one SUM and one MAX all-reduce at the end, together with ``extra_sums`` (GPU tensors the caller adds to
    in ``compute_loss``)."""
    dp_cp_group = parallel_dims.get_mesh("dp_cp").get_group()
    is_moe_model = is_tt_moe_model(model)

    zero = torch.zeros((), device="cuda", dtype=torch.float64)
    sums = {"loss": zero.clone(), "loss_tokens": zero.clone(), "nan_count": zero.clone(), "tokens": zero.clone()}
    maxes = {}
    data_time = 0.0
    for micro_step in range(grad_accum_steps):
        data_start_time = time.perf_counter()
        micro_batch = next(dataiter)
        data_time += time.perf_counter() - data_start_time
        sums["tokens"] += micro_batch["input_ids"].shape[1]
        micro_loss_sum, micro_token_count, aux_loss_sum = compute_loss(micro_batch)
        sums["loss_tokens"] += micro_token_count
        # A NaN micro-batch is left out of the loss and the backward, without a host sync.
        is_nan = torch.isnan(micro_loss_sum.detach())
        sums["nan_count"] += is_nan
        sums["loss"] += torch.where(is_nan, 0.0, micro_loss_sum.detach())
        scaled_loss = torch.where(is_nan, torch.nan_to_num(micro_loss_sum, nan=0.0), micro_loss_sum + aux_loss_sum)
        is_last_micro_step = micro_step == grad_accum_steps - 1
        # Reduce-scatter (and reshard) once per step, not per micro-batch: the expert gradients cross nodes.
        model.set_requires_gradient_sync(is_last_micro_step)
        model.set_reshard_after_backward(is_last_micro_step)
        (scaled_loss / grad_accum_steps).backward()

        if is_moe_model:
            counts, confidence = take_routing_counts(model)
            sums["expert_counts"] = sums.get("expert_counts", zero) + counts
            sums["routing_confidence"] = sums.get("routing_confidence", zero) + confidence
            micro_max_mean = counts.amax(dim=1) / counts.mean(dim=1)
            sums["micro_max_mean"] = sums.get("micro_max_mean", zero) + micro_max_mean.mean()
            maxes["micro_max_mean_max"] = torch.maximum(maxes.get("micro_max_mean_max", zero), micro_max_mean.amax())

    sums["data_time"] = zero + data_time
    maxes["data_time_max"] = zero + data_time
    extra_sums = extra_sums or {}
    stats = reduce_stats(sums | {f"extra/{name}": value for name, value in extra_sums.items()}, maxes, dp_cp_group)

    global_token_count = int(stats["loss_tokens"])
    scale_gradients_(None, model, parallel_dims.fsdp_gradient_divide_factor * grad_accum_steps / global_token_count)

    num_ranks = dist.get_world_size(dp_cp_group)
    metrics = {
        "time/data/max": stats["data_time_max"].item(),
        "time/data/mean": stats["data_time"].item() / num_ranks,
    }
    tokens_per_expert = None
    if is_moe_model:
        counts = stats["expert_counts"]
        routed_tokens = counts.sum(dim=1) / next(iter_moe_blocks(model)).router.top_k
        metrics |= expert_load_metrics(counts, moe_layer_indices(model)) | {
            "expert_load/micro_max_mean/mean": stats["micro_max_mean"].item() / (grad_accum_steps * num_ranks),
            "expert_load/micro_max_mean/max": stats["micro_max_mean_max"].item(),
            "router/confidence": (stats["routing_confidence"] / routed_tokens).mean().item(),
        }
        tokens_per_expert = counts.to("cuda", torch.float32)
    return StepResult(
        loss=stats["loss"].item() / global_token_count,
        nan_count=int(stats["nan_count"]),
        # Every CP rank counts its rows in full.
        num_tokens=int(stats["tokens"]) // parallel_dims.cp,
        tokens_per_expert=tokens_per_expert,
        metrics=metrics,
        sums={name: stats[f"extra/{name}"].item() for name in extra_sums},
    )
