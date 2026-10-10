from __future__ import annotations

from dataclasses import dataclass

import torch
from deep_ep import Buffer
from torch.distributed import ProcessGroup

from prime_rl.trainer.distributed.handles import get_handle, store_handle
from prime_rl.trainer.distributed.token_dispatcher import (
    LocalDispatchState,
    TokenDispatcherBase,
    _scatter_routed_output,
    permute_for_grouped_gemm,
    unpermute_from_grouped_gemm,
)

_buffer: Buffer | None = None


@torch.library.custom_op("deepep::dispatch", mutates_args=())
def _dispatch(
    x: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Send each token to the ranks of its experts.

    Returns the received tokens and their routing weights, the order that groups the received
    (token, local expert) pairs by expert, the pair count per local expert, and the handle ID.
    """
    num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, _ = (
        _buffer.get_dispatch_layout(topk_idx, num_experts)
    )
    recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert, handle, _ = _buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
    )
    # DeepEP already returns the counts on the host, so the order is sliced without a device sync.
    num_local_experts = len(recv_num_tokens_per_expert)
    expert_keys = recv_topk_idx.flatten().masked_fill(recv_topk_idx.flatten() < 0, num_local_experts)
    expert_order = torch.argsort(expert_keys, stable=True)[: sum(recv_num_tokens_per_expert)]
    recv_num_tokens_per_expert = torch.tensor(recv_num_tokens_per_expert, pin_memory=True).to(
        x.device, non_blocking=True
    )
    return recv_x, recv_topk_weights, expert_order, recv_num_tokens_per_expert, store_handle(handle)


@_dispatch.register_fake
def _(x, topk_idx, topk_weights, num_experts):
    ctx = torch.library.get_ctx()
    num_recv_tokens = ctx.new_dynamic_size()
    return (
        x.new_empty((num_recv_tokens, x.shape[1])),
        topk_weights.new_empty((num_recv_tokens, topk_weights.shape[1])),
        topk_idx.new_empty((ctx.new_dynamic_size(),)),
        topk_idx.new_empty((num_experts // _buffer.group_size,)),
        torch.empty(1, dtype=torch.int64),
    )


@torch.library.custom_op("deepep::combine", mutates_args=())
def _combine(x: torch.Tensor, handle_id: torch.Tensor, num_tokens: int) -> torch.Tensor:
    combined, _, _ = _buffer.combine(x, get_handle(handle_id))
    return combined


@_combine.register_fake
def _(x, handle_id, num_tokens):
    return x.new_empty((num_tokens, x.shape[1]))


@torch.library.custom_op("deepep::dispatch_backward", mutates_args=())
def _dispatch_backward_op(
    grad_x: torch.Tensor, grad_topk_weights: torch.Tensor, handle_id: torch.Tensor, num_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_x, grad_topk_weights, _ = _buffer.combine(grad_x, get_handle(handle_id), topk_weights=grad_topk_weights)
    return grad_x, grad_topk_weights


@_dispatch_backward_op.register_fake
def _(grad_x, grad_topk_weights, handle_id, num_tokens):
    return grad_x.new_empty((num_tokens, grad_x.shape[1])), grad_topk_weights.new_empty(
        (num_tokens, grad_topk_weights.shape[1])
    )


@torch.library.custom_op("deepep::combine_backward", mutates_args=())
def _combine_backward_op(grad: torch.Tensor, handle_id: torch.Tensor, num_recv_tokens: int) -> torch.Tensor:
    grad_x, _, _, _, _, _ = _buffer.dispatch(grad, handle=get_handle(handle_id))
    return grad_x


@_combine_backward_op.register_fake
def _(grad, handle_id, num_recv_tokens):
    return grad.new_empty((num_recv_tokens, grad.shape[1]))


def _dispatch_setup_context(ctx, inputs, output) -> None:
    x, *_ = inputs
    *_, handle_id = output
    ctx.save_for_backward(handle_id)
    ctx.num_tokens = x.shape[0]


def _dispatch_backward(ctx, grad_recv_x, grad_recv_topk_weights, *_):
    (handle_id,) = ctx.saved_tensors
    grad_x, grad_topk_weights = _dispatch_backward_op(
        grad_recv_x, grad_recv_topk_weights.contiguous(), handle_id, ctx.num_tokens
    )
    return grad_x, None, grad_topk_weights, None


def _combine_setup_context(ctx, inputs, output) -> None:
    x, handle_id, _ = inputs
    ctx.save_for_backward(handle_id)
    ctx.num_recv_tokens = x.shape[0]


def _combine_backward(ctx, grad_combined):
    (handle_id,) = ctx.saved_tensors
    return _combine_backward_op(grad_combined.contiguous(), handle_id, ctx.num_recv_tokens), None, None


_dispatch.register_autograd(_dispatch_backward, setup_context=_dispatch_setup_context)
_combine.register_autograd(_combine_backward, setup_context=_combine_setup_context)


def _create_buffer(group: ProcessGroup, hidden_bytes: int) -> None:
    global _buffer

    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (Buffer.get_dispatch_config(group.size()), Buffer.get_combine_config(group.size())):
        num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes)
        num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes)

    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        # Internode kernels need at least one RDMA queue pair per SM.
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes, num_qps_per_rank=Buffer.num_sms)


@dataclass(frozen=True)
class DeepEPDispatchState(LocalDispatchState):
    num_input_tokens: int
    handle_id: torch.Tensor


class DeepEPTokenDispatcher(TokenDispatcherBase[DeepEPDispatchState]):
    def __init__(
        self,
        *,
        num_experts: int,
        token_group_alignment: int,
        group: ProcessGroup,
        num_sms: int,
        hidden_size: int,
    ) -> None:
        super().__init__(num_experts, token_group_alignment)
        self.num_local_experts = num_experts // group.size()
        Buffer.set_num_sms(num_sms)
        # Create the buffer before any forward: its constructor runs collectives that dynamo cannot
        # trace and that would shift the op order activation checkpointing replays in the recompute.
        _create_buffer(group, hidden_size * 2)

    def dispatch(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        *,
        score_before_experts: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, DeepEPDispatchState]:
        top_scores = top_scores.float().contiguous()
        selected_experts_indices = selected_experts_indices.masked_fill(top_scores == 0, -1).contiguous()
        recv_x, recv_scores, expert_order, num_tokens_per_expert, handle_id = torch.ops.deepep.dispatch(
            x, selected_experts_indices, top_scores, self.num_experts
        )
        token_indices = expert_order // selected_experts_indices.shape[1]
        routed_input = recv_x[token_indices]
        sorted_scores = recv_scores.flatten()[expert_order]
        scores_after_experts = None
        if score_before_experts:
            routed_input = (routed_input.float() * sorted_scores.reshape(-1, 1)).to(x.dtype)
        else:
            scores_after_experts = sorted_scores

        routed_input, num_tokens_per_expert, permutation = permute_for_grouped_gemm(
            routed_input,
            num_tokens_per_expert,
            experts_per_rank=self.num_local_experts,
            num_ranks=1,
            alignment=self.token_group_alignment,
        )
        state = DeepEPDispatchState(
            num_tokens=recv_x.shape[0],
            token_indices_experts_sorted=token_indices,
            scores_after_experts=scores_after_experts,
            permutation=permutation,
            num_input_tokens=x.shape[0],
            handle_id=handle_id,
        )
        return routed_input, num_tokens_per_expert, state

    def combine(self, routed_output: torch.Tensor, state: DeepEPDispatchState) -> torch.Tensor:
        routed_output = unpermute_from_grouped_gemm(routed_output, state.permutation)
        routed_output = _scatter_routed_output(
            routed_output,
            num_tokens=state.num_tokens,
            token_indices_experts_sorted=state.token_indices_experts_sorted,
            scores_after_experts=state.scores_after_experts,
        )
        return torch.ops.deepep.combine(routed_output, state.handle_id, state.num_input_tokens)
