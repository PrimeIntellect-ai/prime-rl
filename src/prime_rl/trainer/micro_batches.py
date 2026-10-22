"""Forward and backward passes over one optimizer step's micro-batches. The step's statistics stay on the GPU until
the last micro-batch and are then reduced with one SUM and one MAX all-reduce: no host sync per micro-batch."""

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.profiler import record_function

from prime_rl.trainer.metrics import expert_load_metrics, moe_layer_indices, reduce_stats, take_routing_counts
from prime_rl.trainer.model import compute_max_vio_and_routing_confidence, iter_moe_blocks
from prime_rl.trainer.optim.offload import GradientOffloadManager
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.perf import PerfCounter
from prime_rl.trainer.utils import begin_backward, finish_backward, scale_gradients_


@dataclass
class StepResult:
    loss: float
    """Mean loss per unmasked token over the global batch."""

    nan_count: int
    """Micro-batches with a NaN loss over all ranks. They are left out of the loss and the backward pass."""

    num_tokens: int
    """Input tokens of the global batch."""

    num_flops: float
    """Training FLOPs of the global batch."""

    tokens_per_expert: Tensor | None
    """Global ``[num_moe_layers, num_experts]`` token counts of the step. None for dense models."""

    metrics: dict[str, float]
    """The data wait (max and mean over ranks) and, for MoE models, the routing statistics, to log."""


def run_micro_batches(
    model: nn.Module,
    micro_batches: Iterator[dict],
    compute_loss: Callable[[dict], tuple[Tensor, Tensor]],
    grad_accum_steps: int,
    parallel_dims: ParallelDims,
    perf_counter: PerfCounter,
    *,
    gradient_manager: GradientOffloadManager | None = None,
    defer_grad_sync: bool = False,
) -> StepResult:
    """Forward and backward passes over the ``grad_accum_steps`` micro-batches of one step, leaving the gradients of
    the summed token loss divided by the global token count. ``compute_loss`` returns the summed loss and the number
    of unmasked tokens of a micro-batch. With a ``gradient_manager``, the caller has set the gradient scale up front.
    With ``defer_grad_sync``, FSDP reduce-scatters the gradients only at the last micro-batch."""
    dp_cp_group = parallel_dims.get_mesh("dp_cp").get_group()
    ep_group = parallel_dims.get_mesh("ep").get_group() if parallel_dims.ep_enabled else None
    moes = list(iter_moe_blocks(model))

    zero = torch.zeros((), device="cuda", dtype=torch.float64)
    loss_sum, loss_tokens, nan_count = zero.clone(), zero.clone(), zero.clone()
    num_tokens, num_flops, data_time = 0, 0, 0.0
    expert_counts, routing_confidence_sums = [], []
    for micro_step in range(grad_accum_steps):
        data_start_time = time.perf_counter()
        micro_batch = next(micro_batches)
        data_time += time.perf_counter() - data_start_time
        num_tokens += micro_batch["input_ids"].shape[1]
        num_flops += perf_counter.flop_counter(micro_batch["seq_lens"].tolist())

        with record_function("forward"):
            micro_loss_sum, micro_token_count = compute_loss(micro_batch)
        loss_tokens += micro_token_count
        is_nan = torch.isnan(micro_loss_sum.detach())
        nan_count += is_nan
        loss_sum += torch.where(is_nan, 0.0, micro_loss_sum.detach())
        scaled_loss = torch.where(is_nan, torch.nan_to_num(micro_loss_sum, nan=0.0), micro_loss_sum) / grad_accum_steps

        is_last_micro_step = micro_step == grad_accum_steps - 1
        if defer_grad_sync:
            model.set_requires_gradient_sync(is_last_micro_step)
            model.set_reshard_after_backward(is_last_micro_step)
        with record_function("backward"):
            begin_backward(gradient_manager, final_backward=is_last_micro_step)
            scaled_loss.backward()
            finish_backward(gradient_manager)

        if moes:
            counts, confidence_sums = take_routing_counts(moes)
            expert_counts.append(counts)
            routing_confidence_sums.append(confidence_sums)

    # Every CP rank holds the same rows.
    sums = {
        "loss": loss_sum,
        "loss_tokens": loss_tokens,
        "nan_count": nan_count,
        "tokens": zero + num_tokens / parallel_dims.cp,
        "flops": zero + num_flops / parallel_dims.cp,
        "data_time": zero + data_time,
    }
    maxes = {"data_time_max": zero + data_time}
    if moes:
        counts, confidence_sums = torch.stack(expert_counts), torch.stack(routing_confidence_sums)
        sums["expert_counts"] = counts.sum(dim=0)
        # Max violation and routing confidence of every micro-batch, over the EP group's tokens.
        if ep_group is not None:
            packed = torch.cat([counts, confidence_sums.unsqueeze(-1)], dim=-1)
            dist.all_reduce(packed, group=ep_group)
            counts, confidence_sums = packed[..., :-1], packed[..., -1]
        max_vio, routing_confidence = compute_max_vio_and_routing_confidence(
            counts, confidence_sums, moes[0].router.top_k
        )
        maxes["max_vio"] = max_vio.amax(dim=-1)
        sums["routing_confidence"] = routing_confidence.mean(dim=-1)
    stats = reduce_stats(sums, maxes, dp_cp_group)

    global_token_count = int(stats["loss_tokens"])
    if gradient_manager is None and global_token_count > 0:
        scale_gradients_(None, model, parallel_dims.fsdp_gradient_divide_factor * grad_accum_steps / global_token_count)

    num_ranks = dist.get_world_size(dp_cp_group)
    metrics = {"time/data/max": stats["data_time_max"].item(), "time/data/mean": stats["data_time"].item() / num_ranks}
    tokens_per_expert = None
    if moes:
        tokens_per_expert = stats["expert_counts"]
        metrics |= expert_load_metrics(tokens_per_expert, moe_layer_indices(model)) | {
            "max_vio/mean": stats["max_vio"].mean().item(),
            "max_vio/max": stats["max_vio"].max().item(),
            "routing_confidence/mean": stats["routing_confidence"].mean().item() / num_ranks,
        }
    return StepResult(
        loss=stats["loss"].item() / global_token_count if global_token_count > 0 else 0.0,
        nan_count=int(stats["nan_count"]),
        num_tokens=round(stats["tokens"].item()),
        num_flops=stats["flops"].item(),
        tokens_per_expert=tokens_per_expert,
        metrics=metrics,
    )
