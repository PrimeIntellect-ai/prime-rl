from dataclasses import dataclass
from typing import cast
from weakref import WeakValueDictionary

import torch
from deep_ep import Buffer
from deep_ep.utils import EventHandle, EventOverlap
from torch.distributed import ProcessGroup

from prime_rl.trainer.distributed.token_dispatcher import (
    ExpertFunction,
    PermutationState,
    TokenDispatcherBase,
    permute_for_grouped_gemm,
    unpermute_from_grouped_gemm,
)

_buffer: Buffer | None = None
_handle_cache: dict[int, object] = {}
_combine_backward_handles: dict[int, object] = {}
_combine_dispatchers: WeakValueDictionary[int, object] = WeakValueDictionary()
_pending_dispatch_events: dict[int, EventOverlap] = {}
_handle_counter = 0
_deepep_cuda_ops_registered = False
_deepep_cuda_lib: torch.library.Library | None = None


def _get_next_handle_id() -> torch.Tensor:
    global _handle_counter
    _handle_counter += 1
    return torch.tensor([_handle_counter], dtype=torch.int64, device="cpu")


def _new_event_overlap() -> EventOverlap:
    return EventOverlap(EventHandle())


def register_deepep_cuda_ops() -> None:
    global _deepep_cuda_lib, _deepep_cuda_ops_registered
    if _deepep_cuda_ops_registered:
        return

    # Keep the Library alive so PyTorch does not deregister the custom ops.
    _deepep_cuda_lib = torch.library.Library("deepep", "DEF")
    _deepep_cuda_lib.define(
        "dispatch(Tensor x, Tensor topk_idx, Tensor topk_weights, "
        "Tensor num_tokens_per_rank, Tensor num_tokens_per_rdma_rank, "
        "Tensor is_token_in_rank, Tensor num_tokens_per_expert) "
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor)"
    )
    _deepep_cuda_lib.define("combine(Tensor x, Tensor handle_id, int dispatcher_id) -> Tensor")

    torch.library.impl(_deepep_cuda_lib, "dispatch", "CUDA")(_dispatch_op_impl)
    torch.library.impl(_deepep_cuda_lib, "combine", "CUDA")(_combine_op_impl)

    torch.library.register_autograd("deepep::dispatch", _dispatch_backward, setup_context=_dispatch_setup_context)
    torch.library.register_autograd("deepep::combine", _combine_backward, setup_context=_combine_setup_context)

    _deepep_cuda_ops_registered = True


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

    previous_event = _new_event_overlap()
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
    handle_id = _get_next_handle_id()
    _handle_cache[handle_id.item()] = handle
    _pending_dispatch_events[handle_id.item()] = after_event
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

    previous_event = _new_event_overlap()
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


def _combine_op_impl(x: torch.Tensor, handle_id: torch.Tensor, dispatcher_id: int) -> torch.Tensor:
    assert _buffer is not None, "DeepEP buffer must be initialized before combine."
    handle_key = handle_id.item()
    handle = _handle_cache.pop(handle_key, None)
    assert handle is not None, f"Handle not found for handle_id={handle_key}"

    previous_event = _new_event_overlap()
    combined, _, after_event = _buffer.combine(
        x=x,
        handle=handle,
        previous_event=previous_event,
        async_finish=True,
        allocate_on_comm_stream=True,
    )
    dispatcher = cast("DeepEPTokenDispatcher | None", _combine_dispatchers.get(dispatcher_id))
    assert dispatcher is not None, "DeepEP dispatcher was released before combine completed."
    dispatcher._pending_combine_events.append(after_event)
    if x.requires_grad:
        _combine_backward_handles[handle_key] = handle
    return combined


def _combine_setup_context(ctx, inputs, output) -> None:
    _, handle_id, _ = inputs
    ctx.handle = _combine_backward_handles.pop(handle_id.item(), None)


def _combine_backward(ctx, grad_combined: torch.Tensor) -> tuple[torch.Tensor, None, None]:
    handle = ctx.handle
    assert handle is not None, "Handle not found in DeepEP combine backward."

    previous_event = _new_event_overlap()
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
    return grad_x, None, None


@torch.compiler.disable()
def _sync_dispatch(handle_id: torch.Tensor | int) -> None:
    handle_key = handle_id if isinstance(handle_id, int) else handle_id.item()
    pending_event = _pending_dispatch_events.pop(handle_key, None)
    if pending_event is not None:
        pending_event.current_stream_wait()


def configure_num_sms(num_sms: int) -> None:
    """Set the number of SMs for DeepEP intranode dispatch/combine kernels.

    Must be called before the first dispatch/combine. Also determines
    internode RDMA channel count (num_channels = num_sms / 2).
    """
    Buffer.set_num_sms(num_sms)


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
class _PendingDispatchState:
    hidden_states: torch.Tensor
    dispatched_indices: torch.Tensor
    dispatched_scores: torch.Tensor
    num_tokens_per_expert: torch.Tensor
    handle_id: torch.Tensor
    score_before_experts: bool


def dispatch_tokens_async(
    hidden_states: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    top_scores: torch.Tensor,
    num_experts: int,
    group: ProcessGroup,
    *,
    score_before_experts: bool = True,
) -> _PendingDispatchState:
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

    return _PendingDispatchState(
        hidden_states=hidden_states,
        dispatched_indices=dispatched_indices,
        dispatched_scores=dispatched_expert_scores,
        num_tokens_per_expert=num_tokens_per_expert,
        handle_id=handle_id,
        score_before_experts=score_before_experts,
    )


@dataclass(frozen=True)
class DeepEPDispatchState:
    handle_id: torch.Tensor
    deep_ep_permuted_indices: torch.Tensor
    num_received_tokens: int
    scores_after_experts: torch.Tensor | None
    grouped_gemm_permutation: PermutationState


class DeepEPTokenDispatcher(TokenDispatcherBase[DeepEPDispatchState]):
    def __init__(
        self,
        *,
        num_experts: int,
        token_group_alignment: int,
        group: ProcessGroup,
        num_sms: int,
        token_chunk_size: int | None,
    ) -> None:
        super().__init__(num_experts, token_group_alignment)
        self.num_local_experts = num_experts // group.size()
        self.group = group
        self.token_chunk_size = token_chunk_size
        self._pending_combine_events: list[EventOverlap] = []
        self._dispatcher_id = id(self)
        _combine_dispatchers[self._dispatcher_id] = self
        self._concatenate_stream: torch.cuda.Stream | None = None
        self._output_event: torch.cuda.Event | None = None
        configure_num_sms(num_sms)

    def _finalize_dispatch(
        self, pending_state: _PendingDispatchState
    ) -> tuple[torch.Tensor, torch.Tensor, DeepEPDispatchState]:
        _sync_dispatch(pending_state.handle_id)

        hidden_states = pending_state.hidden_states
        num_received_tokens = hidden_states.shape[0]
        hidden_states, permuted_scores, deep_ep_permuted_indices = _permute_tokens(
            hidden_states,
            pending_state.dispatched_indices,
            pending_state.dispatched_scores,
        )
        num_tokens_per_expert = pending_state.num_tokens_per_expert.to(hidden_states.device)

        if pending_state.score_before_experts:
            hidden_states = (hidden_states.float() * permuted_scores.float().reshape(-1, 1)).to(hidden_states.dtype)
            scores_after_experts = None
        else:
            scores_after_experts = permuted_scores

        hidden_states, num_tokens_per_expert, grouped_gemm_permutation = permute_for_grouped_gemm(
            hidden_states,
            num_tokens_per_expert,
            experts_per_rank=self.num_local_experts,
            num_ranks=1,
            alignment=self.token_group_alignment,
        )
        state = DeepEPDispatchState(
            handle_id=pending_state.handle_id,
            deep_ep_permuted_indices=deep_ep_permuted_indices,
            num_received_tokens=num_received_tokens,
            scores_after_experts=scores_after_experts,
            grouped_gemm_permutation=grouped_gemm_permutation,
        )
        return hidden_states, num_tokens_per_expert, state

    def dispatch(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        *,
        score_before_experts: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, DeepEPDispatchState]:
        pending_state = dispatch_tokens_async(
            x,
            selected_experts_indices,
            top_scores,
            num_experts=self.num_experts,
            group=self.group,
            score_before_experts=score_before_experts,
        )
        return self._finalize_dispatch(pending_state)

    def combine(self, routed_output: torch.Tensor, state: DeepEPDispatchState) -> torch.Tensor:
        routed_output = unpermute_from_grouped_gemm(routed_output, state.grouped_gemm_permutation)
        if state.scores_after_experts is not None:
            routed_output = (routed_output.float() * state.scores_after_experts.float().reshape(-1, 1)).to(
                routed_output.dtype
            )
        routed_output = _unpermute_tokens(
            routed_output,
            state.deep_ep_permuted_indices,
            state.num_received_tokens,
        )
        combined = torch.ops.deepep.combine(routed_output, state.handle_id, self._dispatcher_id)
        return combined

    def _run_dispatched_chunk(
        self,
        pending_state: _PendingDispatchState,
        experts: ExpertFunction,
    ) -> torch.Tensor:
        hidden_states, num_tokens_per_expert, dispatch_state = self._finalize_dispatch(pending_state)
        routed_output = experts(hidden_states, num_tokens_per_expert)
        return self.combine(routed_output, dispatch_state)

    @torch.compiler.disable()
    def _synchronize_combines(self) -> None:
        for event in self._pending_combine_events:
            event.current_stream_wait()
        self._pending_combine_events.clear()

    def run(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        experts: ExpertFunction,
        *,
        score_before_experts: bool,
    ) -> torch.Tensor:
        if self.token_chunk_size is None:
            chunk_ranges = [(0, x.shape[0])]
        else:
            chunk_ranges = [
                (start, min(start + self.token_chunk_size, x.shape[0]))
                for start in range(0, x.shape[0], self.token_chunk_size)
            ]

        pending_states = (
            dispatch_tokens_async(
                x[start:end],
                selected_experts_indices[start:end],
                top_scores[start:end],
                num_experts=self.num_experts,
                group=self.group,
                score_before_experts=score_before_experts,
            )
            for start, end in chunk_ranges
        )

        pending_state = next(pending_states)
        routed_outputs: list[torch.Tensor] = []
        for next_pending_state in pending_states:
            routed_outputs.append(self._run_dispatched_chunk(pending_state, experts))
            pending_state = next_pending_state
        routed_outputs.append(self._run_dispatched_chunk(pending_state, experts))
        if len(routed_outputs) == 1:
            return routed_outputs[0]

        if self._concatenate_stream is None:
            self._concatenate_stream = torch.cuda.Stream()
        with torch.cuda.stream(self._concatenate_stream):
            self._synchronize_combines()
            output = torch.cat(routed_outputs, dim=0)
            self._output_event = self._concatenate_stream.record_event()
        return output

    def synchronize(self) -> None:
        if self._output_event is not None:
            torch.cuda.current_stream().wait_event(self._output_event)
            self._output_event = None
            return
        self._synchronize_combines()


register_deepep_cuda_ops()

__all__ = [
    "configure_num_sms",
    "DeepEPTokenDispatcher",
    "dispatch_tokens_async",
]
