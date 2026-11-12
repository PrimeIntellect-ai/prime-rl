"""Forward and backward passes over one optimizer step's micro-batches. The step's statistics stay on the GPU: after
the last micro-batch they are reduced with one SUM and one MAX all-reduce, and the step copies them to the host once,
with its other statistics (`to_host`, `step_result`)."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.profiler import record_function

from prime_rl.trainer.model import (
    add_routing_stats,
    compute_expert_load_stats,
    iter_moe_blocks,
    take_moe_routing_stats,
)
from prime_rl.trainer.optim.offload import GradientOffloadManager
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.perf import PerfCounter
from prime_rl.trainer.utils import begin_backward, finish_backward, scale_gradients_


@dataclass
class StepResult:
    loss: float
    """Mean loss per unmasked token over the global batch."""

    nan_count: int
    """Micro-batches with a NaN loss over all ranks. Their loss is left out of the logged loss; their gradients may
    still be NaN."""

    num_tokens: int
    num_flops: float

    metrics: dict[str, float]
    """The routing and expert-load statistics of MoE models, to log. Empty for dense models."""


def run_micro_batches(
    model: nn.Module,
    micro_batches: Iterator[dict],
    compute_loss: Callable[[dict], tuple[Tensor, Tensor]],
    grad_accum_steps: int,
    parallel_dims: ParallelDims,
    perf_counter: PerfCounter,
    gradient_manager: GradientOffloadManager | None = None,
) -> dict[str, Tensor]:
    """Forward and backward passes over the ``grad_accum_steps`` micro-batches of one step, leaving the gradients of
    the summed token loss divided by the global token count. ``compute_loss`` returns the summed loss and the number
    of unmasked tokens of a micro-batch. With a ``gradient_manager``, the caller has set the gradient scale up front.
    Returns the step's statistics reduced over the ranks, still on the device."""
    dp_cp_group = parallel_dims.get_mesh("dp_cp").get_group()
    ep_group = parallel_dims.get_mesh("ep").get_group() if parallel_dims.ep_enabled else None
    moes = list(iter_moe_blocks(model))

    zero = torch.zeros((), device="cuda", dtype=torch.float64)
    loss_sum, loss_tokens, nan_count = zero.clone(), zero.clone(), zero.clone()
    num_tokens, num_flops = 0, 0
    routing_stats = []
    for micro_step in range(grad_accum_steps):
        micro_batch = next(micro_batches)
        num_tokens += micro_batch["input_ids"].shape[1]
        num_flops += perf_counter.flop_counter(micro_batch["seq_lens"].tolist())

        with record_function("forward"):
            micro_loss_sum, micro_token_count = compute_loss(micro_batch)
        loss_tokens += micro_token_count
        is_nan = torch.isnan(micro_loss_sum.detach())
        nan_count += is_nan
        loss_sum += torch.where(is_nan, 0.0, micro_loss_sum.detach())
        # The summed loss: the gradients are divided by the step's loss tokens once, after the last backward.
        scaled_loss = torch.where(is_nan, torch.nan_to_num(micro_loss_sum, nan=0.0), micro_loss_sum)

        is_last_micro_step = micro_step == grad_accum_steps - 1
        if parallel_dims.dp_replicate_enabled and gradient_manager is None:
            # HSDP keeps the reduce-scattered gradients across micro batches and all-reduces them over the replicas
            # once, after the last backward
            model.set_requires_all_reduce(is_last_micro_step)
        with record_function("backward"):
            begin_backward(gradient_manager, final_backward=is_last_micro_step)
            scaled_loss.backward()
            finish_backward(gradient_manager)

        if moes:
            routing_stats.append(take_moe_routing_stats(model))

    # Every CP rank holds the same rows: `step_result` divides the token and FLOP counts by the CP degree.
    sums = {
        "loss": loss_sum,
        "loss_tokens": loss_tokens,
        "nan_count": nan_count,
        "tokens": zero + num_tokens,
        "flops": zero + num_flops,
    }
    maxes = {}
    if moes:
        add_routing_stats(sums, maxes, routing_stats, moes[0].router.top_k, ep_group)
    stats = reduce_stats(sums, maxes, dp_cp_group)
    if gradient_manager is None:
        tokens = stats["loss_tokens"]
        factor = parallel_dims.fsdp_gradient_divide_factor / tokens.clamp(min=1)
        scale_gradients_(None, model, torch.where(tokens > 0, factor, 1.0).float())
    return stats


def reduce_stats(sums: dict[str, Tensor], maxes: dict[str, Tensor], group: dist.ProcessGroup) -> dict[str, Tensor]:
    """Sum or max per-rank statistics over ``group`` with one all-reduce per kind. The results stay on the device as
    float64."""
    reduced = {}
    for values, op in ((sums, dist.ReduceOp.SUM), (maxes, dist.ReduceOp.MAX)):
        if not values:
            continue
        flat = torch.cat([value.reshape(-1).to(torch.float64) for value in values.values()])
        dist.all_reduce(flat, op=op, group=group)
        for name, value in zip(values, flat.split([value.numel() for value in values.values()])):
            reduced[name] = value.reshape(values[name].shape)
    return reduced


def to_host(*tensor_dicts: dict[str, Tensor]) -> list[dict[str, Tensor]]:
    """CPU float64 copies of the tensors in ``tensor_dicts``, in the same structure, through one device-to-host copy."""
    tensors = [tensor for tensor_dict in tensor_dicts for tensor in tensor_dict.values()]
    flat = torch.cat([tensor.detach().reshape(-1).to(torch.float64) for tensor in tensors]).cpu()
    values = iter(flat.split([tensor.numel() for tensor in tensors]))
    return [
        {name: next(values).reshape(tensor.shape) for name, tensor in tensor_dict.items()}
        for tensor_dict in tensor_dicts
    ]


def step_result(stats: dict[str, Tensor], parallel_dims: ParallelDims) -> StepResult:
    """The step's values from the host copy of the statistics of `run_micro_batches`."""
    global_token_count = int(stats["loss_tokens"])
    metrics = {}
    if "expert_counts" in stats:
        num_ranks = dist.get_world_size(parallel_dims.get_mesh("dp_cp").get_group())
        metrics = {name: value.item() for name, value in compute_expert_load_stats(stats["expert_counts"]).items()}
        metrics |= {
            "max_vio/mean": stats["max_vio"].mean().item(),
            "max_vio/max": stats["max_vio"].max().item(),
            "routing_confidence/mean": stats["routing_confidence"].mean().item() / num_ranks,
        }
        if "ep_imbalance/rank" in stats:
            dp_mean = stats["ep_imbalance/rank_dp_mean"] / num_ranks
            metrics |= {
                "ep_imbalance/rank/mean": stats["ep_imbalance/rank"].mean().item(),
                "ep_imbalance/rank/max": stats["ep_imbalance/rank"].max().item(),
                "ep_imbalance/rank_dp_mean/mean": dp_mean.mean().item(),
                "ep_imbalance/rank_dp_mean/max": dp_mean.max().item(),
            }
    return StepResult(
        loss=stats["loss"].item() / global_token_count if global_token_count > 0 else 0.0,
        nan_count=int(stats["nan_count"]),
        num_tokens=round(stats["tokens"].item() / parallel_dims.cp),
        num_flops=stats["flops"].item() / parallel_dims.cp,
        metrics=metrics,
    )
