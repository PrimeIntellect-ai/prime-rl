from __future__ import annotations

from dataclasses import dataclass
from weakref import WeakValueDictionary

import torch
from deep_ep import Buffer
from deep_ep.utils import EventHandle, EventOverlap
from torch.distributed import ProcessGroup

from prime_rl.trainer.distributed.token_dispatcher import ExpertFunction, TokenDispatcherBase
from prime_rl.trainer.models.kernels.fp8_utils import per_token_cast_to_fp8_triton, per_token_dequant_fp8_triton
from prime_rl.trainer.models.kernels.moe_permute import PairLayout, build_pair_layout, gather_pairs, reduce_pairs

_buffer: Buffer | None = None
_handle_cache: dict[int, object] = {}
_combine_backward_handles: dict[int, object] = {}
# The custom-op schema can carry only the dispatcher ID; weak values avoid extending dispatcher lifetimes.
_combine_dispatchers: WeakValueDictionary[int, DeepEPTokenDispatcher] = WeakValueDictionary()
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
    dispatcher = _combine_dispatchers.get(dispatcher_id)
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
        # Internode kernels need at least one RDMA queue pair per SM (DeepEP's default is 24).
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes, num_qps_per_rank=max(24, Buffer.num_sms))

    return _buffer


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
    layout: PairLayout
    scores_after_experts: torch.Tensor | None


class DeepEPTokenDispatcher(TokenDispatcherBase[DeepEPDispatchState]):
    def __init__(
        self,
        *,
        num_experts: int,
        token_group_alignment: int,
        group: ProcessGroup,
        num_sms: int,
        token_chunk_size: int | None,
        hidden_size: int,
        fp8: bool = False,
    ) -> None:
        super().__init__(num_experts, token_group_alignment)
        self.num_local_experts = num_experts // group.size()
        self.group = group
        self.token_chunk_size = token_chunk_size
        self.fp8 = fp8
        self._pending_combine_events: list[EventOverlap] = []
        self._dispatcher_id = id(self)
        _combine_dispatchers[self._dispatcher_id] = self
        self._experts: ExpertFunction | None = None
        configure_num_sms(num_sms)
        # Create the communication buffer now: created lazily, it would be built inside the first
        # activation-checkpointed forward, whose recompute then replays every later cached op
        # off by the collectives and tensors the creation ran.
        get_buffer(group, hidden_size * 2)

    def _finalize_dispatch(
        self, pending_state: _PendingDispatchState
    ) -> tuple[torch.Tensor, torch.Tensor, DeepEPDispatchState]:
        _sync_dispatch(pending_state.handle_id)

        layout, num_tokens_per_expert = build_pair_layout(
            pending_state.dispatched_indices,
            pending_state.num_tokens_per_expert.tolist(),
            self.token_group_alignment,
        )
        scores = pending_state.dispatched_scores
        if pending_state.score_before_experts:
            hidden_states = gather_pairs(pending_state.hidden_states, layout, scores)
            scores_after_experts = None
        else:
            hidden_states = gather_pairs(pending_state.hidden_states, layout)
            scores_after_experts = scores
        state = DeepEPDispatchState(
            handle_id=pending_state.handle_id, layout=layout, scores_after_experts=scores_after_experts
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
        routed_output = reduce_pairs(routed_output, state.layout, state.scores_after_experts)
        return torch.ops.deepep.combine(routed_output, state.handle_id, self._dispatcher_id)

    @torch.compiler.disable()
    def _synchronize_combines(self) -> None:
        for event in self._pending_combine_events:
            event.current_stream_wait()
        self._pending_combine_events.clear()

    @torch.compiler.disable()
    def run(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        experts: ExpertFunction,
        *,
        score_before_experts: bool,
    ) -> torch.Tensor:
        self._experts = experts
        outputs = torch.ops.prime_rl.deepep_moe(
            x, selected_experts_indices, top_scores, self._dispatcher_id, score_before_experts, torch.is_grad_enabled()
        )
        return outputs[0]

    def synchronize(self) -> None:
        self._synchronize_combines()

    def chunk_ranges(self, num_tokens: int) -> list[tuple[int, int]]:
        size = self.token_chunk_size or max(num_tokens, 1)
        return [(start, min(start + size, num_tokens)) for start in range(0, max(num_tokens, 1), size)]

    def routed_experts(
        self,
        recv_x: torch.Tensor,
        recv_indices: torch.Tensor,
        recv_scores: torch.Tensor,
        pairs_per_expert: list[int],
        score_before_experts: bool,
    ) -> torch.Tensor:
        """One rank's received tokens through its local experts, reduced back to one row per token."""
        layout, num_tokens_per_expert = build_pair_layout(recv_indices, pairs_per_expert, self.token_group_alignment)
        if score_before_experts:
            routed = self._experts(gather_pairs(recv_x, layout, recv_scores), num_tokens_per_expert)
            return reduce_pairs(routed, layout)
        routed = self._experts(gather_pairs(recv_x, layout), num_tokens_per_expert)
        return reduce_pairs(routed, layout, recv_scores)

    def received_tokens(self, recv_x: torch.Tensor, recv_sf: torch.Tensor) -> torch.Tensor:
        """Received tokens in bf16; FP8 dispatch keeps them quantized until the experts read them."""
        return per_token_dequant_fp8_triton(recv_x, recv_sf) if self.fp8 else recv_x

    def moe_forward(
        self, x: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor, score_before_experts: bool
    ) -> tuple[torch.Tensor, list[object], list[torch.Tensor]]:
        """Dispatch every chunk up front, then run each chunk's experts while later chunks are in flight."""
        buffer = get_buffer(self.group, get_hidden_bytes(x))
        topk_idx = topk_idx.contiguous().masked_fill(topk_weights == 0, -1)
        topk_weights = topk_weights.float().contiguous()
        in_flight = []
        for start, end in self.chunk_ranges(x.shape[0]):
            chunk_idx = topk_idx[start:end]
            per_rank, per_rdma_rank, per_expert, in_rank, _ = buffer.get_dispatch_layout(
                topk_idx=chunk_idx, num_experts=self.num_experts
            )
            chunk = x[start:end]
            if self.fp8:
                chunk_q, chunk_sf = per_token_cast_to_fp8_triton(chunk, use_ue8m0=True)
                chunk = (chunk_q, chunk_sf.contiguous())
            recv_x, recv_idx, recv_scores, counts, handle, event = buffer.dispatch(
                x=chunk,
                topk_idx=chunk_idx,
                topk_weights=topk_weights[start:end],
                num_tokens_per_rank=per_rank,
                num_tokens_per_rdma_rank=per_rdma_rank,
                is_token_in_rank=in_rank,
                num_tokens_per_expert=per_expert,
                previous_event=_new_event_overlap(),
                async_finish=True,
                allocate_on_comm_stream=True,
            )
            in_flight.append((recv_x, recv_idx, recv_scores, counts, handle, event))

        combined, handles, saved, events = [], [], [], []
        for recv_x, recv_idx, recv_scores, counts, handle, event in in_flight:
            event.current_stream_wait()
            recv_x, recv_sf = recv_x if self.fp8 else (recv_x, recv_x.new_empty(0))
            routed = self.routed_experts(
                self.received_tokens(recv_x, recv_sf), recv_idx, recv_scores, counts, score_before_experts
            )
            out, _, event = buffer.combine(
                x=routed,
                handle=handle,
                previous_event=_new_event_overlap(),
                async_finish=True,
                allocate_on_comm_stream=True,
            )
            combined.append(out)
            events.append(event)
            handles.append(handle)
            saved += [recv_x, recv_sf, recv_idx, recv_scores, torch.tensor(counts, dtype=torch.int32)]
        for event in events:
            event.current_stream_wait()
        return torch.cat(combined) if len(combined) > 1 else combined[0], handles, saved

    def moe_backward(
        self,
        grad_out: torch.Tensor,
        handles: list[object],
        saved: tuple[torch.Tensor, ...],
        score_before_experts: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Send every chunk's output gradient back up front, then run each chunk's expert backward
        (recomputing its forward) while the other chunks' gradients are in flight."""
        buffer = get_buffer(self.group, get_hidden_bytes(grad_out))
        grad_out = grad_out.contiguous()
        ranges = self.chunk_ranges(grad_out.shape[0])
        in_flight = []
        for i, (start, end) in enumerate(ranges):
            grad_recv, _, _, _, _, event = buffer.dispatch(
                x=grad_out[start:end],
                handle=handles[i],
                previous_event=_new_event_overlap(),
                async_finish=True,
                allocate_on_comm_stream=True,
            )
            in_flight.append((grad_recv, event))

        grads, events = [], []
        for i, (grad_recv, event) in enumerate(in_flight):
            recv_x, recv_sf, recv_idx, recv_scores, counts = saved[5 * i : 5 * i + 5]
            with torch.enable_grad():
                recv_x = self.received_tokens(recv_x, recv_sf).detach().requires_grad_()
                recv_scores = recv_scores.detach().requires_grad_()
                routed = self.routed_experts(recv_x, recv_idx, recv_scores, counts.tolist(), score_before_experts)
            event.current_stream_wait()
            # Accumulates the experts' weight gradients like any other backward.
            torch.autograd.backward(routed, grad_recv)
            grad_x, grad_scores, event = buffer.combine(
                x=recv_x.grad,
                handle=handles[i],
                topk_weights=recv_scores.grad,
                previous_event=_new_event_overlap(),
                async_finish=True,
                allocate_on_comm_stream=True,
            )
            grads.append((grad_x, grad_scores))
            events.append(event)
        for event in events:
            event.current_stream_wait()
        grad_x = torch.cat([g for g, _ in grads]) if len(grads) > 1 else grads[0][0]
        grad_scores = torch.cat([g for _, g in grads]) if len(grads) > 1 else grads[0][1]
        return grad_x, grad_scores


# Forward communication handles, kept from the forward to its backward.
_moe_handles: dict[int, list[object]] = {}


@torch.library.custom_op("prime_rl::deepep_moe", mutates_args=())
def deepep_moe(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    dispatcher_id: int,
    score_before_experts: bool,
    keep_for_backward: bool,
) -> list[torch.Tensor]:
    """Routed experts of one MoE layer over DeepEP, pipelined over token chunks.

    Returns the combined output, the key of the communication handles its backward reuses, and
    each chunk's received tokens, their FP8 scales (empty without FP8 dispatch), expert ids, scores
    and per-expert pair counts. Activation checkpointing saves all of them, so recompute neither
    resends tokens nor reruns the experts; the backward recomputes each chunk's experts itself,
    overlapped with communication.
    """
    dispatcher = _combine_dispatchers[dispatcher_id]
    out, handles, saved = dispatcher.moe_forward(x, topk_idx, topk_weights, score_before_experts)
    key = _get_next_handle_id()
    if keep_for_backward:
        _moe_handles[key.item()] = handles
    return [out, key, *saved]


@deepep_moe.register_fake
def _deepep_moe_fake(x, topk_idx, topk_weights, dispatcher_id, score_before_experts, keep_for_backward):
    dispatcher = _combine_dispatchers[dispatcher_id]
    ctx = torch.library.get_ctx()
    outputs = [torch.empty_like(x), torch.empty(1, dtype=torch.int64)]
    for _ in dispatcher.chunk_ranges(x.shape[0]):
        rows = ctx.new_dynamic_size()
        if dispatcher.fp8:
            tokens = x.new_empty(rows, x.shape[1], dtype=torch.float8_e4m3fn)
            scales = x.new_empty(rows, x.shape[1] // 128, dtype=torch.float32)
        else:
            tokens, scales = x.new_empty(rows, x.shape[1]), x.new_empty(0)
        outputs += [
            tokens,
            scales,
            topk_idx.new_empty(rows, topk_idx.shape[1]),
            x.new_empty(rows, topk_idx.shape[1], dtype=torch.float32),
            torch.empty(dispatcher.num_local_experts, dtype=torch.int32),
        ]
    return outputs


def _deepep_moe_setup_context(ctx, inputs, output) -> None:
    x, _, topk_weights, dispatcher_id, score_before_experts, _ = inputs
    ctx.dispatcher_id, ctx.score_before_experts = dispatcher_id, score_before_experts
    ctx.handle_key = output[1].item()
    ctx.x_dtype, ctx.weights_dtype = x.dtype, topk_weights.dtype
    ctx.save_for_backward(*output[2:])


def _deepep_moe_backward(ctx, grads):
    dispatcher = _combine_dispatchers[ctx.dispatcher_id]
    handles = _moe_handles.pop(ctx.handle_key)
    grad_x, grad_scores = dispatcher.moe_backward(grads[0], handles, ctx.saved_tensors, ctx.score_before_experts)
    return grad_x.to(ctx.x_dtype), None, grad_scores.to(ctx.weights_dtype), None, None, None


deepep_moe.register_autograd(_deepep_moe_backward, setup_context=_deepep_moe_setup_context)


register_deepep_cuda_ops()

__all__ = [
    "configure_num_sms",
    "DeepEPTokenDispatcher",
    "dispatch_tokens_async",
]
