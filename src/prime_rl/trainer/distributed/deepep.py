from dataclasses import dataclass

import torch
from torch.distributed import ProcessGroup

try:
    from deep_ep import Buffer
    from deep_ep.utils import EventHandle, EventOverlap
except ImportError as e:
    raise ImportError("DeepEP is required for this backend. Install from https://github.com/deepseek-ai/DeepEP.") from e


_buffer: Buffer | None = None
_handle_cache: dict[int, object] = {}
_handle_counter = 0
_pending_combine_event: EventOverlap | None = None


def _get_next_handle_id() -> torch.Tensor:
    global _handle_counter
    _handle_counter += 1
    return torch.tensor([_handle_counter], dtype=torch.int64, device="cpu")


_lib = torch.library.Library("deepep", "DEF")
_lib.define(
    "dispatch(Tensor x, Tensor topk_idx, Tensor topk_weights, "
    "Tensor num_tokens_per_rank, Tensor num_tokens_per_rdma_rank, "
    "Tensor is_token_in_rank, Tensor num_tokens_per_expert) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor)"
)
_lib.define("combine(Tensor x, Tensor handle_id) -> Tensor")


@torch.library.impl(_lib, "dispatch", "CUDA")
def _dispatch_op_impl(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_tokens_per_rank: torch.Tensor,
    num_tokens_per_rdma_rank: torch.Tensor,
    is_token_in_rank: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert _buffer is not None, "DeepEP buffer must be initialized before dispatch."

    previous_event = EventOverlap(EventHandle())
    recv_x, recv_indices, recv_scores, recv_num_tokens_per_expert_list, handle, after_event = _buffer.dispatch(
        x=x,
        topk_idx=topk_idx,
        topk_weights=topk_weights.to(torch.float32),
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        previous_event=previous_event,
        async_finish=True,
        allocate_on_comm_stream=True,
    )
    after_event.current_stream_wait()

    handle_id = _get_next_handle_id()
    _handle_cache[handle_id.item()] = handle
    recv_num_tokens_per_expert = torch.tensor(recv_num_tokens_per_expert_list, dtype=torch.int32, device="cpu")
    return recv_x, recv_indices, recv_scores, recv_num_tokens_per_expert, handle_id


def _dispatch_setup_context(ctx, inputs, output) -> None:
    x, *_ = inputs
    *_, handle_id = output
    ctx.input_dtype = x.dtype
    ctx.saved_handle = _handle_cache.get(handle_id.item())


def _dispatch_backward(
    ctx,
    grad_recv_x,
    grad_recv_indices,
    grad_recv_scores,
    grad_recv_num_tokens_per_expert,
    grad_handle_id,
):
    if grad_recv_x is None:
        return None, None, None, None, None, None, None

    handle = ctx.saved_handle
    assert handle is not None

    previous_event = EventOverlap(EventHandle())
    grad_x, grad_scores, after_event = _buffer.combine(
        x=grad_recv_x,
        handle=handle,
        topk_weights=grad_recv_scores.float() if grad_recv_scores is not None else None,
        previous_event=previous_event,
        async_finish=True,
        allocate_on_comm_stream=True,
    )
    after_event.current_stream_wait()

    grad_x = grad_x.to(ctx.input_dtype)
    grad_topk_weights = grad_scores.to(ctx.input_dtype) if grad_scores is not None else None
    return grad_x, None, grad_topk_weights, None, None, None, None


@torch.library.impl(_lib, "combine", "CUDA")
def _combine_op_impl(x: torch.Tensor, handle_id: torch.Tensor) -> torch.Tensor:
    global _pending_combine_event

    assert _buffer is not None, "DeepEP buffer must be initialized before combine."
    if torch.is_inference_mode_enabled():
        handle = _handle_cache.pop(handle_id.item(), None)
    else:
        handle = _handle_cache.get(handle_id.item())
    assert handle is not None, f"Handle not found for handle_id={handle_id.item()}"

    previous_event = EventOverlap(EventHandle())
    combined, _, after_event = _buffer.combine(
        x=x,
        handle=handle,
        previous_event=previous_event,
        async_finish=True,
        allocate_on_comm_stream=True,
    )
    _pending_combine_event = after_event
    return combined


def _combine_setup_context(ctx, inputs, output) -> None:
    _, handle_id = inputs
    ctx.saved_handle = _handle_cache.pop(handle_id.item(), None)


def _combine_backward(ctx, grad_combined):
    handle = ctx.saved_handle
    assert handle is not None, "Handle not found in DeepEP combine backward."

    previous_event = EventOverlap(EventHandle())
    grad_x, _, _, _, _, after_event = _buffer.dispatch(
        x=grad_combined,
        topk_idx=None,
        topk_weights=None,
        num_tokens_per_rank=None,
        num_tokens_per_rdma_rank=None,
        is_token_in_rank=None,
        num_tokens_per_expert=None,
        handle=handle,
        previous_event=previous_event,
        async_finish=True,
        allocate_on_comm_stream=True,
    )
    after_event.current_stream_wait()
    return grad_x, None


torch.library.register_autograd("deepep::dispatch", _dispatch_backward, setup_context=_dispatch_setup_context)
torch.library.register_autograd("deepep::combine", _combine_backward, setup_context=_combine_setup_context)


@torch.compiler.disable()
def sync_combine() -> None:
    global _pending_combine_event

    if _pending_combine_event is not None:
        _pending_combine_event.current_stream_wait()
        _pending_combine_event = None


def get_hidden_bytes(x: torch.Tensor) -> int:
    return x.size(1) * max(x.element_size(), 2)


def get_buffer(group: ProcessGroup, hidden_bytes: int) -> Buffer:
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
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)

    return _buffer


def _permute_tokens(
    hidden_states: torch.Tensor,
    dispatched_indices: torch.Tensor,
    dispatched_scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = dispatched_indices != -1
    valid_expert_ids = dispatched_indices[mask]
    valid_scores = dispatched_scores[mask]

    sort_order = torch.argsort(valid_expert_ids, stable=True)
    permuted_indices = torch.arange(len(hidden_states), device=hidden_states.device).repeat_interleave(mask.sum(dim=1))[
        sort_order
    ]
    permuted_hidden_states = hidden_states.index_select(0, permuted_indices)
    permuted_scores = valid_scores[sort_order]
    return permuted_hidden_states, permuted_scores, permuted_indices


def _unpermute_tokens(
    permuted_hidden_states: torch.Tensor,
    permuted_indices: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    hidden_dim = permuted_hidden_states.shape[1]
    output_hidden_states = permuted_hidden_states.new_zeros((num_tokens, hidden_dim))
    output_hidden_states.scatter_add_(0, permuted_indices.unsqueeze(1).expand(-1, hidden_dim), permuted_hidden_states)
    return output_hidden_states


@dataclass
class DispatchState:
    handle_id: torch.Tensor
    permuted_indices: torch.Tensor
    num_recv_tokens: int
    permuted_scores: torch.Tensor | None = None


def dispatch_tokens(
    hidden_states: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    top_scores: torch.Tensor,
    num_experts: int,
    group: ProcessGroup,
    *,
    score_before_experts: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, DispatchState]:
    selected_experts_indices = selected_experts_indices.contiguous()
    top_scores = top_scores.contiguous()
    selected_experts_indices = selected_experts_indices.masked_fill(top_scores == 0, -1)
    if top_scores.dtype != torch.float32:
        top_scores = top_scores.float()

    buffer = get_buffer(group, get_hidden_bytes(hidden_states))
    num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert_dispatch, is_token_in_rank, _ = (
        buffer.get_dispatch_layout(topk_idx=selected_experts_indices, num_experts=num_experts)
    )

    hidden_states, dispatched_indices, dispatched_expert_scores, num_tokens_per_expert, handle_id = (
        torch.ops.deepep.dispatch(
            hidden_states,
            selected_experts_indices,
            top_scores,
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            is_token_in_rank,
            num_tokens_per_expert_dispatch,
        )
    )

    num_recv_tokens = hidden_states.shape[0]
    hidden_states, permuted_scores, permuted_indices = _permute_tokens(
        hidden_states, dispatched_indices, dispatched_expert_scores
    )
    num_tokens_per_expert = num_tokens_per_expert.to(hidden_states.device)

    if score_before_experts and permuted_scores is not None:
        hidden_states = hidden_states * permuted_scores.to(hidden_states.dtype).reshape(-1, 1)
        permuted_scores_for_state = None
    else:
        permuted_scores_for_state = permuted_scores

    state = DispatchState(
        handle_id=handle_id,
        permuted_indices=permuted_indices,
        num_recv_tokens=num_recv_tokens,
        permuted_scores=permuted_scores_for_state,
    )
    return hidden_states, num_tokens_per_expert, state


def combine_tokens(hidden_states: torch.Tensor, state: DispatchState) -> torch.Tensor:
    if state.permuted_scores is not None:
        hidden_states = hidden_states * state.permuted_scores.to(hidden_states.dtype).reshape(-1, 1)
    hidden_states = _unpermute_tokens(hidden_states, state.permuted_indices, state.num_recv_tokens)
    return torch.ops.deepep.combine(hidden_states, state.handle_id)


__all__ = ["DispatchState", "combine_tokens", "dispatch_tokens", "sync_combine"]
