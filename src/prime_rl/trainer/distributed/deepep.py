from dataclasses import dataclass

import torch
from deep_ep import Buffer
from deep_ep.utils import EventHandle, EventOverlap
from torch.distributed import ProcessGroup

from prime_rl.trainer.distributed.token_dispatcher import (
    ExpertFunction,
    PermutationState,
    permute_for_grouped_gemm,
    unpermute_from_grouped_gemm,
)

_buffer: Buffer | None = None
_handle_cache: dict[int, object] = {}
_pending_dispatch_events: dict[int, EventOverlap] = {}
_handle_counter = 0
_pending_combine_events: list[EventOverlap] = []
_concatenate_stream: torch.cuda.Stream | None = None
_deepep_cuda_ops_registered = False
_deepep_cuda_lib: torch.library.Library | None = None


def _get_next_handle_id() -> torch.Tensor:
    global _handle_counter
    _handle_counter += 1
    return torch.tensor([_handle_counter], dtype=torch.int64, device="cpu")


def _new_event_overlap() -> EventOverlap:
    return EventOverlap(EventHandle())


def _get_concatenate_stream() -> torch.cuda.Stream:
    global _concatenate_stream
    if _concatenate_stream is None:
        _concatenate_stream = torch.cuda.Stream()
    return _concatenate_stream


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

    torch.library.impl(_deepep_cuda_lib, "dispatch", "CUDA")(_dispatch_op_impl)

    torch.library.register_autograd("deepep::dispatch", _dispatch_backward, setup_context=_dispatch_setup_context)

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


class _DeepEPCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, handle_id: torch.Tensor) -> torch.Tensor:
        assert _buffer is not None, "DeepEP buffer must be initialized before combine."
        handle = _handle_cache.pop(handle_id.item(), None)
        assert handle is not None, f"Handle not found for handle_id={handle_id.item()}"

        previous_event = _new_event_overlap()
        combined, _, after_event = _buffer.combine(
            x=x,
            handle=handle,
            previous_event=previous_event,
            async_finish=True,
            allocate_on_comm_stream=True,
        )
        _pending_combine_events.append(after_event)
        ctx.handle = handle
        return combined

    @staticmethod
    def backward(ctx, grad_combined: torch.Tensor) -> tuple[torch.Tensor, None]:
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
        return grad_x, None


@torch.compiler.disable()
def sync_combine() -> None:
    for event in _pending_combine_events:
        event.current_stream_wait()
    _pending_combine_events.clear()


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
class _DispatchState:
    handle_id: torch.Tensor
    permuted_indices: torch.Tensor
    num_recv_tokens: int
    num_input_tokens: int
    permuted_scores: torch.Tensor | None = None


@dataclass
class _PendingDispatchState:
    hidden_states: torch.Tensor
    dispatched_indices: torch.Tensor
    dispatched_scores: torch.Tensor
    num_tokens_per_expert: torch.Tensor
    handle_id: torch.Tensor
    num_input_tokens: int
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
    num_input_tokens = hidden_states.shape[0]
    if num_input_tokens == 0:
        hidden_states = hidden_states.sum(dim=0, keepdim=True)
        selected_experts_indices = torch.arange(
            selected_experts_indices.shape[1],
            dtype=selected_experts_indices.dtype,
            device=selected_experts_indices.device,
        ).unsqueeze(0)
        top_scores = top_scores.new_ones((1, top_scores.shape[1]))

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
        num_input_tokens=num_input_tokens,
        score_before_experts=score_before_experts,
    )


def finalize_dispatch_tokens(pending_state: _PendingDispatchState) -> tuple[torch.Tensor, torch.Tensor, _DispatchState]:
    _sync_dispatch(pending_state.handle_id)

    hidden_states = pending_state.hidden_states
    num_recv_tokens = hidden_states.shape[0]
    hidden_states, permuted_scores, permuted_indices = _permute_tokens(
        hidden_states,
        pending_state.dispatched_indices,
        pending_state.dispatched_scores,
    )
    num_tokens_per_expert = pending_state.num_tokens_per_expert.to(hidden_states.device)

    if pending_state.score_before_experts and permuted_scores is not None:
        hidden_states = (hidden_states.to(torch.float32) * permuted_scores.to(torch.float32).reshape(-1, 1)).to(
            hidden_states.dtype
        )
        permuted_scores_for_state = None
    else:
        permuted_scores_for_state = permuted_scores

    state = _DispatchState(
        handle_id=pending_state.handle_id,
        permuted_indices=permuted_indices,
        num_recv_tokens=num_recv_tokens,
        num_input_tokens=pending_state.num_input_tokens,
        permuted_scores=permuted_scores_for_state,
    )
    return hidden_states, num_tokens_per_expert, state


def combine_tokens(hidden_states: torch.Tensor, state: _DispatchState) -> torch.Tensor:
    if state.permuted_scores is not None:
        hidden_states = (hidden_states.to(torch.float32) * state.permuted_scores.to(torch.float32).reshape(-1, 1)).to(
            hidden_states.dtype
        )
    hidden_states = _unpermute_tokens(hidden_states, state.permuted_indices, state.num_recv_tokens)
    combined = _DeepEPCombine.apply(hidden_states, state.handle_id)
    return combined[: state.num_input_tokens]


@dataclass(frozen=True)
class DeepEPDispatchState:
    backend: _DispatchState
    permutation: PermutationState


class DeepEPTokenDispatcher:
    def __init__(
        self,
        *,
        num_experts: int,
        token_group_alignment: int,
        group: ProcessGroup,
        num_sms: int,
        token_chunk_size: int | None,
    ) -> None:
        self.num_experts = num_experts
        self.num_local_experts = num_experts // group.size()
        self.token_group_alignment = token_group_alignment
        self.group = group
        self.token_chunk_size = token_chunk_size
        self._output_event: torch.cuda.Event | None = None
        configure_num_sms(num_sms)

    def _finalize_dispatch(
        self, pending_state: _PendingDispatchState
    ) -> tuple[torch.Tensor, torch.Tensor, DeepEPDispatchState]:
        hidden_states, num_tokens_per_expert, backend_state = finalize_dispatch_tokens(pending_state)
        hidden_states, num_tokens_per_expert, permutation = permute_for_grouped_gemm(
            hidden_states,
            num_tokens_per_expert,
            experts_per_rank=self.num_local_experts,
            num_ranks=1,
            alignment=self.token_group_alignment,
        )
        return hidden_states, num_tokens_per_expert, DeepEPDispatchState(backend_state, permutation)

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
        routed_output = unpermute_from_grouped_gemm(routed_output, state.permutation)
        return combine_tokens(routed_output, state.backend)

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
            max_tokens = torch.tensor(x.shape[0], dtype=torch.int64, device=x.device)
            torch.distributed.all_reduce(max_tokens, op=torch.distributed.ReduceOp.MAX, group=self.group)
            max_tokens_per_rank = max_tokens.item()
            chunk_ranges = [
                (min(start, x.shape[0]), min(start + self.token_chunk_size, x.shape[0]))
                for start in range(0, max_tokens_per_rank, self.token_chunk_size)
            ]
            if not chunk_ranges:
                chunk_ranges = [(0, 0)]

        def dispatch_chunk(start: int, end: int) -> _PendingDispatchState:
            return dispatch_tokens_async(
                x[start:end],
                selected_experts_indices[start:end],
                top_scores[start:end],
                num_experts=self.num_experts,
                group=self.group,
                score_before_experts=score_before_experts,
            )

        def run_pending_chunk(pending_state: _PendingDispatchState) -> torch.Tensor:
            hidden_states, num_tokens_per_expert, dispatch_state = self._finalize_dispatch(pending_state)
            routed_output = experts(hidden_states, num_tokens_per_expert)
            return self.combine(routed_output, dispatch_state)

        pending_state = dispatch_chunk(*chunk_ranges[0])
        routed_outputs: list[torch.Tensor] = []
        for chunk_start, chunk_end in chunk_ranges[1:]:
            next_pending_state = dispatch_chunk(chunk_start, chunk_end)
            routed_outputs.append(run_pending_chunk(pending_state))
            pending_state = next_pending_state
        routed_outputs.append(run_pending_chunk(pending_state))
        if len(routed_outputs) == 1:
            return routed_outputs[0]

        concatenate_stream = _get_concatenate_stream()
        with torch.cuda.stream(concatenate_stream):
            sync_combine()
            output = torch.cat(routed_outputs, dim=0)
            self._output_event = concatenate_stream.record_event()
        return output

    def synchronize(self) -> None:
        if self._output_event is not None:
            torch.cuda.current_stream().wait_event(self._output_event)
            self._output_event = None
            return
        sync_combine()


register_deepep_cuda_ops()

__all__ = [
    "combine_tokens",
    "configure_num_sms",
    "DeepEPTokenDispatcher",
    "dispatch_tokens_async",
    "finalize_dispatch_tokens",
    "sync_combine",
]
