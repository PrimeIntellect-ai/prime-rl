from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from deep_ep_v2 import EPBuffer
from torch.distributed import ProcessGroup

from prime_rl.trainer.distributed.handles import get_handle, store_handle
from prime_rl.trainer.distributed.token_dispatcher import TokenDispatcherBase
from prime_rl.trainer.world import get_world

_buffer: EPBuffer | None = None


@torch.library.custom_op("deepep_v2::dispatch", mutates_args=())
def _dispatch(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    expert_alignment: int,
    num_sms: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Send each token to the ranks of its experts.

    Returns one row per (token, local expert) pair, grouped by expert and padded per expert to
    `expert_alignment`, the routing weight of each row, the padded row count per local expert,
    and the handle ID.
    """
    recv_x, _, recv_weights, handle, _ = _buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=num_experts,
        expert_alignment=expert_alignment,
        num_sms=num_sms,
        do_expand=True,
        do_zero_padding=True,
    )
    num_tokens_per_expert = torch.tensor(handle.num_recv_tokens_per_expert_list, pin_memory=True).to(
        x.device, non_blocking=True
    )
    # Padding rows are zeroed, their weights are not.
    expert = torch.repeat_interleave(num_tokens_per_expert, output_size=recv_x.shape[0])
    expert_start = num_tokens_per_expert.cumsum(0) - num_tokens_per_expert
    row_in_expert = torch.arange(recv_x.shape[0], device=x.device) - expert_start[expert]
    recv_weights = recv_weights.masked_fill(row_in_expert >= handle.num_unaligned_recv_tokens_per_expert[expert], 0)
    return recv_x, recv_weights, num_tokens_per_expert, store_handle(handle)


@_dispatch.register_fake
def _(x, topk_idx, topk_weights, num_experts, expert_alignment, num_sms):
    num_recv_rows = torch.library.get_ctx().new_dynamic_size()
    return (
        x.new_empty((num_recv_rows, x.shape[1])),
        topk_weights.new_empty((num_recv_rows,)),
        topk_idx.new_empty((num_experts // _buffer.num_ranks,)),
        torch.empty(1, dtype=torch.int64),
    )


@torch.library.custom_op("deepep_v2::combine", mutates_args=())
def _combine(x: torch.Tensor, handle_id: torch.Tensor, num_tokens: int) -> torch.Tensor:
    combined, _, _ = _buffer.combine(x, get_handle(handle_id))
    return combined


@_combine.register_fake
def _(x, handle_id, num_tokens):
    return x.new_empty((num_tokens, x.shape[1]))


@torch.library.custom_op("deepep_v2::dispatch_backward", mutates_args=())
def _dispatch_backward_op(
    grad_x: torch.Tensor, grad_weights: torch.Tensor, handle_id: torch.Tensor, num_tokens: int, num_topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_x, grad_topk_weights, _ = _buffer.combine(grad_x, get_handle(handle_id), topk_weights=grad_weights)
    return grad_x, grad_topk_weights


@_dispatch_backward_op.register_fake
def _(grad_x, grad_weights, handle_id, num_tokens, num_topk):
    return grad_x.new_empty((num_tokens, grad_x.shape[1])), grad_weights.new_empty((num_tokens, num_topk))


@torch.library.custom_op("deepep_v2::combine_backward", mutates_args=())
def _combine_backward_op(grad: torch.Tensor, handle_id: torch.Tensor, num_recv_rows: int) -> torch.Tensor:
    handle = get_handle(handle_id)
    grad_x, _, _, _, _ = _buffer.dispatch(
        grad, handle=handle, num_sms=handle.num_sms, do_expand=True, do_zero_padding=True
    )
    return grad_x


@_combine_backward_op.register_fake
def _(grad, handle_id, num_recv_rows):
    return grad.new_empty((num_recv_rows, grad.shape[1]))


def _dispatch_setup_context(ctx, inputs, output) -> None:
    x, topk_idx, *_ = inputs
    *_, handle_id = output
    ctx.save_for_backward(handle_id)
    ctx.num_tokens, ctx.num_topk = topk_idx.shape


def _dispatch_backward(ctx, grad_recv_x, grad_recv_weights, *_):
    (handle_id,) = ctx.saved_tensors
    grad_x, grad_topk_weights = _dispatch_backward_op(
        grad_recv_x.contiguous(), grad_recv_weights.contiguous(), handle_id, ctx.num_tokens, ctx.num_topk
    )
    return grad_x, None, grad_topk_weights, None, None, None


def _combine_setup_context(ctx, inputs, output) -> None:
    x, handle_id, _ = inputs
    ctx.save_for_backward(handle_id)
    ctx.num_recv_rows = x.shape[0]


def _combine_backward(ctx, grad_combined):
    (handle_id,) = ctx.saved_tensors
    return _combine_backward_op(grad_combined.contiguous(), handle_id, ctx.num_recv_rows), None, None


_dispatch.register_autograd(_dispatch_backward, setup_context=_dispatch_setup_context)
_combine.register_autograd(_combine_backward, setup_context=_combine_setup_context)


def _create_buffer(group: ProcessGroup, num_max_tokens_per_rank: int, hidden_size: int, num_topk: int) -> None:
    global _buffer
    if _buffer is not None and _buffer.group == group and _buffer.num_max_tokens_per_rank >= num_max_tokens_per_rank:
        return
    local_world_size = get_world().local_world_size
    if len({rank // local_world_size for rank in dist.get_process_group_ranks(group)}) > 1:
        raise ValueError(
            "DeepEP V2 dispatch (type = 'deepep_v2') needs the expert-parallel group within one node. "
            "Use type = 'deepep' when expert parallelism spans nodes."
        )
    if _buffer is not None:
        _buffer.destroy()
    _buffer = EPBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden_size,
        num_topk=num_topk,
        explicitly_destroy=True,
    )


@dataclass(frozen=True)
class DeepEPV2DispatchState:
    num_input_tokens: int
    handle_id: torch.Tensor
    scores_after_experts: torch.Tensor | None


class DeepEPV2TokenDispatcher(TokenDispatcherBase[DeepEPV2DispatchState]):
    """DeepEP V2 dispatch in the expanded layout: rows arrive grouped and aligned per local expert."""

    def __init__(
        self,
        *,
        num_experts: int,
        top_k: int,
        token_group_alignment: int,
        group: ProcessGroup,
        num_sms: int,
        hidden_size: int,
        num_max_tokens_per_rank: int,
    ) -> None:
        super().__init__(num_experts, token_group_alignment)
        self.num_sms = num_sms
        # Create the buffer before any forward: its constructor runs collectives that dynamo cannot
        # trace and that would shift the op order activation checkpointing replays in the recompute.
        _create_buffer(group, num_max_tokens_per_rank, hidden_size, top_k)

    def dispatch(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        *,
        score_before_experts: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, DeepEPV2DispatchState]:
        top_scores = top_scores.float().contiguous()
        selected_experts_indices = selected_experts_indices.masked_fill(top_scores == 0, -1).contiguous()
        recv_x, recv_scores, num_tokens_per_expert, handle_id = torch.ops.deepep_v2.dispatch(
            x, selected_experts_indices, top_scores, self.num_experts, self.token_group_alignment, self.num_sms
        )
        scores_after_experts = None
        if score_before_experts:
            recv_x = (recv_x.float() * recv_scores.reshape(-1, 1)).to(x.dtype)
        else:
            scores_after_experts = recv_scores
        return recv_x, num_tokens_per_expert, DeepEPV2DispatchState(x.shape[0], handle_id, scores_after_experts)

    def combine(self, routed_output: torch.Tensor, state: DeepEPV2DispatchState) -> torch.Tensor:
        if state.scores_after_experts is not None:
            routed_output = (routed_output.float() * state.scores_after_experts.reshape(-1, 1)).to(routed_output.dtype)
        return torch.ops.deepep_v2.combine(routed_output, state.handle_id, state.num_input_tokens)
