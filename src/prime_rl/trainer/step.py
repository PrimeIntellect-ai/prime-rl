from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor, nn

from prime_rl.trainer.model import get_global_moe_stats, is_tt_moe_model
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.utils import begin_backward, finish_backward, prepare_gradient_offload, scale_gradients_
from prime_rl.utils.logger import get_logger


@dataclass
class StepResult:
    loss: float
    """Mean loss per unmasked token over the global batch."""

    nan_count: int
    """Micro-batches with a NaN loss over all ranks. They are left out of the backward pass."""

    num_tokens: int
    """Input tokens of the global batch."""

    moe_stats: dict[str, float]
    """Routing statistics averaged over the micro-batches (``max_vio/mean``, ``max_vio/max``,
    ``routing_confidence/mean``). Empty for dense models."""

    tokens_per_expert: Tensor | None
    """Global ``[num_moe_layers, num_experts]`` token counts of the step. None for dense models."""


def run_micro_batches(
    model: nn.Module,
    dataiter: Iterator[dict],
    compute_loss: Callable[[dict], tuple[Tensor, Tensor, Tensor | None]],
    grad_accum_steps: int,
    parallel_dims: ParallelDims,
    gradient_manager=None,
    *,
    overlap_optimizer: bool = True,
    record_function: Callable[[str], AbstractContextManager] = nullcontext,
) -> StepResult:
    """Forward and backward passes over the ``grad_accum_steps`` micro-batches of one step, leaving the
    gradients of the summed token loss divided by the global token count (the mean loss per token, as on one
    GPU without accumulation). ``compute_loss`` returns the summed loss and the number of unmasked tokens of a
    micro-batch, and an optional auxiliary loss on the same scale as the summed loss that is trained but not logged."""
    logger = get_logger()
    dp_cp_group = parallel_dims.get_mesh("dp_cp").get_group()
    ep_group = parallel_dims.get_mesh("ep").get_group() if parallel_dims.ep_enabled else None
    is_moe_model = is_tt_moe_model(model)

    if gradient_manager is None:
        micro_batches = (next(dataiter) for _ in range(grad_accum_steps))
        local_token_count = torch.tensor(0, dtype=torch.int64, device="cuda")
    else:
        # The offloaded gradients are scaled as they stream out, so the token count is needed up front.
        micro_batches = [next(dataiter) for _ in range(grad_accum_steps)]
        global_token_count = torch.tensor(
            sum(int(micro_batch["loss_mask"].sum()) for micro_batch in micro_batches), device="cuda"
        )
        dist.all_reduce(global_token_count, op=dist.ReduceOp.SUM, group=dp_cp_group)
        global_token_count = global_token_count.item() // parallel_dims.cp
        grad_scale = (
            parallel_dims.fsdp_gradient_divide_factor * grad_accum_steps / global_token_count
            if global_token_count > 0
            else 1.0
        )
        prepare_gradient_offload(gradient_manager, grad_scale, overlap_optimizer=overlap_optimizer)

    loss_sum = torch.tensor(0.0, device="cuda")
    nan_count = torch.tensor(0, device="cuda")
    moe_stats = {name: torch.tensor(0.0) for name in ("max_vio/mean", "max_vio/max", "routing_confidence/mean")}
    tokens_per_expert = None
    local_num_tokens = 0
    for micro_step, micro_batch in enumerate(micro_batches):
        local_num_tokens += micro_batch["input_ids"].shape[1]
        micro_loss_sum, micro_token_count, aux_loss_sum = compute_loss(micro_batch)
        if gradient_manager is None:
            local_token_count += micro_token_count
        if torch.isnan(micro_loss_sum.detach()):
            nan_count += 1
            logger.warning("Local loss is nan, excluding this micro step from backward")
            scaled_loss = torch.nan_to_num(micro_loss_sum, nan=0.0) / grad_accum_steps
        else:
            loss_sum += micro_loss_sum.detach()
            scaled_loss = (micro_loss_sum if aux_loss_sum is None else micro_loss_sum + aux_loss_sum) / grad_accum_steps
        with record_function("backward"):
            begin_backward(gradient_manager, final_backward=micro_step == grad_accum_steps - 1)
            scaled_loss.backward()
            finish_backward(gradient_manager)

        if is_moe_model:
            micro_moe_stats, micro_tokens_per_expert = get_global_moe_stats(model, ep_group, dp_cp_group)
            for name, value in micro_moe_stats.items():
                moe_stats[f"{name}/mean"] += value / grad_accum_steps
                if name == "max_vio":
                    moe_stats["max_vio/max"] = torch.maximum(moe_stats["max_vio/max"], value)
            tokens_per_expert = micro_tokens_per_expert + (0 if tokens_per_expert is None else tokens_per_expert)

    if gradient_manager is None:
        dist.all_reduce(local_token_count, op=dist.ReduceOp.SUM, group=dp_cp_group)
        global_token_count = local_token_count.item()
        if global_token_count > 0:
            scale_gradients_(
                None, model, parallel_dims.fsdp_gradient_divide_factor * grad_accum_steps / global_token_count
            )

    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM, group=dp_cp_group)
    dist.all_reduce(nan_count, op=dist.ReduceOp.SUM)
    # The dp mesh excludes cp, whose ranks hold the same rows.
    num_tokens = torch.tensor(local_num_tokens, dtype=torch.int64, device="cuda")
    dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM, group=parallel_dims.get_mesh("dp").get_group())
    return StepResult(
        loss=(loss_sum / global_token_count).item() if global_token_count > 0 else 0.0,
        nan_count=int(nan_count.item()),
        num_tokens=int(num_tokens.item()),
        moe_stats={name: value.item() for name, value in moe_stats.items()} if is_moe_model else {},
        tokens_per_expert=tokens_per_expert,
    )
