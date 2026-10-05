from __future__ import annotations

from dataclasses import dataclass, replace
from weakref import WeakValueDictionary

import torch
from deep_ep import Buffer
from deep_ep.utils import EventHandle, EventOverlap
from torch.distributed import ProcessGroup

from prime_rl.trainer.distributed.token_dispatcher import ExpertFunction, TokenDispatcherBase
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
        experts: ExpertFunction,
    ) -> None:
        super().__init__(num_experts, token_group_alignment)
        self.num_local_experts = num_experts // group.size()
        self.group = group
        self.token_chunk_size = token_chunk_size
        self._pending_combine_events: list[EventOverlap] = []
        self._dispatcher_id = id(self)
        _combine_dispatchers[self._dispatcher_id] = self
        self._experts = experts
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
        # Every chunk is sent before any is computed, so later chunks travel while earlier ones compute.
        chunks = [
            self.launch(x[start:end], top_scores[start:end], selected_experts_indices[start:end])
            for start, end in self.chunk_ranges(x.shape[0])
        ]
        chunks = [self.compute(chunk, score_before_experts) for chunk in chunks]
        outputs = [self.wait(chunk) for chunk in chunks]
        return torch.cat(outputs) if len(outputs) > 1 else outputs[0]

    def synchronize(self) -> None:
        self._synchronize_combines()

    def chunk_ranges(self, num_tokens: int) -> list[tuple[int, int]]:
        size = self.token_chunk_size or max(num_tokens, 1)
        return [(start, min(start + size, num_tokens)) for start in range(0, max(num_tokens, 1), size)]

    @torch.compiler.disable()
    def launch(self, x: torch.Tensor, top_scores: torch.Tensor, selected_experts_indices: torch.Tensor) -> InFlightMoE:
        """Start sending tokens to their experts' ranks; `compute` and `wait` finish the layer."""
        key, recv_x, recv_idx, recv_scores, counts = torch.ops.prime_rl.deepep_dispatch_launch(
            x, selected_experts_indices, top_scores, self._dispatcher_id
        )
        return InFlightMoE(key, recv_x, recv_idx, recv_scores, counts, x.shape[0])

    @torch.compiler.disable()
    def compute(self, chunk: InFlightMoE, score_before_experts: bool) -> InFlightMoE:
        """Wait for the tokens, run the local experts and start sending the results back."""
        out = torch.ops.prime_rl.deepep_experts(
            chunk.recv_x,
            chunk.recv_idx,
            chunk.recv_scores,
            chunk.counts,
            chunk.key,
            chunk.num_tokens,
            self._dispatcher_id,
            score_before_experts,
        )
        return replace(chunk, out=out)

    @torch.compiler.disable()
    def wait(self, chunk: InFlightMoE) -> torch.Tensor:
        """The combined expert output, once it has arrived."""
        return torch.ops.prime_rl.deepep_combine_wait(chunk.out, chunk.key)

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


@dataclass(frozen=True)
class InFlightMoE:
    """One MoE layer's tokens between `launch`, `compute` and `wait`."""

    key: torch.Tensor
    recv_x: torch.Tensor
    recv_idx: torch.Tensor
    recv_scores: torch.Tensor
    counts: torch.Tensor
    num_tokens: int
    out: torch.Tensor | None = None


@dataclass
class _Transfers:
    """DeepEP state one MoE layer's three ops share, forward and backward.

    Forward: `deepep_dispatch_launch` sends the tokens (`dispatched`), `deepep_experts` waits for
    them and sends the results back (`combined`), `deepep_combine_wait` waits for those. Backward
    mirrors it: `deepep_combine_wait` sends the output gradient (`grad_received`), `deepep_experts`
    waits for it and sends the input gradients back (`grad_returned`), and `deepep_dispatch_launch`
    waits for those. Between a send and its wait, the other micro-batch's work runs.
    """

    handle: object
    dispatched: EventOverlap
    combined: EventOverlap | None = None
    grad_received: tuple[torch.Tensor, EventOverlap] | None = None
    grad_returned: tuple[torch.Tensor, torch.Tensor, EventOverlap] | None = None


_transfers: dict[int, _Transfers] = {}


@torch.library.custom_op("prime_rl::deepep_dispatch_launch", mutates_args=())
def deepep_dispatch_launch(
    x: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor, dispatcher_id: int
) -> list[torch.Tensor]:
    """Start sending tokens to their experts' ranks. Returns the transfer key and the received
    tokens, expert ids, scores and per-expert pair counts, still in flight until `deepep_experts`."""
    dispatcher = _combine_dispatchers[dispatcher_id]
    buffer = get_buffer(dispatcher.group, get_hidden_bytes(x))
    topk_idx = topk_idx.contiguous().masked_fill(topk_weights == 0, -1)
    per_rank, per_rdma_rank, per_expert, in_rank, _ = buffer.get_dispatch_layout(
        topk_idx=topk_idx, num_experts=dispatcher.num_experts
    )
    recv_x, recv_idx, recv_scores, counts, handle, event = buffer.dispatch(
        x=x,
        topk_idx=topk_idx,
        topk_weights=topk_weights.float().contiguous(),
        num_tokens_per_rank=per_rank,
        num_tokens_per_rdma_rank=per_rdma_rank,
        is_token_in_rank=in_rank,
        num_tokens_per_expert=per_expert,
        previous_event=_new_event_overlap(),
        async_finish=True,
        allocate_on_comm_stream=True,
    )
    key = _get_next_handle_id()
    _transfers[key.item()] = _Transfers(handle=handle, dispatched=event)
    return [key, recv_x, recv_idx, recv_scores, torch.tensor(counts, dtype=torch.int32)]


@deepep_dispatch_launch.register_fake
def _deepep_dispatch_launch_fake(x, topk_idx, topk_weights, dispatcher_id):
    dispatcher = _combine_dispatchers[dispatcher_id]
    rows = torch.library.get_ctx().new_dynamic_size()
    return [
        torch.empty(1, dtype=torch.int64),
        x.new_empty(rows, x.shape[1]),
        topk_idx.new_empty(rows, topk_idx.shape[1]),
        x.new_empty(rows, topk_idx.shape[1], dtype=torch.float32),
        torch.empty(dispatcher.num_local_experts, dtype=torch.int32),
    ]


def _deepep_dispatch_launch_setup_context(ctx, inputs, output) -> None:
    x, _, topk_weights, _ = inputs
    ctx.key, ctx.x_dtype, ctx.weights_dtype = output[0].item(), x.dtype, topk_weights.dtype


def _deepep_dispatch_launch_backward(ctx, *grads):
    grad_x, grad_scores, event = _transfers.pop(ctx.key).grad_returned
    event.current_stream_wait()
    return grad_x.to(ctx.x_dtype), None, grad_scores.to(ctx.weights_dtype), None


deepep_dispatch_launch.register_autograd(
    _deepep_dispatch_launch_backward, setup_context=_deepep_dispatch_launch_setup_context
)


@torch.library.custom_op("prime_rl::deepep_experts", mutates_args=())
def deepep_experts(
    recv_x: torch.Tensor,
    recv_idx: torch.Tensor,
    recv_scores: torch.Tensor,
    counts: torch.Tensor,
    key: torch.Tensor,
    num_tokens: int,
    dispatcher_id: int,
    score_before_experts: bool,
) -> torch.Tensor:
    """Wait for the dispatched tokens, run the local experts and start sending the results back.
    The `(num_tokens, hidden)` output is in flight until `deepep_combine_wait`."""
    dispatcher = _combine_dispatchers[dispatcher_id]
    transfers = _transfers[key.item()]
    transfers.dispatched.current_stream_wait()
    routed = dispatcher.routed_experts(recv_x, recv_idx, recv_scores, counts.tolist(), score_before_experts)
    out, _, transfers.combined = _buffer.combine(
        x=routed,
        handle=transfers.handle,
        previous_event=_new_event_overlap(),
        async_finish=True,
        allocate_on_comm_stream=True,
    )
    return out


@deepep_experts.register_fake
def _deepep_experts_fake(recv_x, recv_idx, recv_scores, counts, key, num_tokens, dispatcher_id, score_before_experts):
    return recv_x.new_empty(num_tokens, recv_x.shape[1])


def _deepep_experts_setup_context(ctx, inputs, output) -> None:
    recv_x, recv_idx, recv_scores, counts, key, _, dispatcher_id, score_before_experts = inputs
    ctx.key, ctx.dispatcher_id, ctx.score_before_experts = key.item(), dispatcher_id, score_before_experts
    ctx.save_for_backward(recv_x, recv_idx, recv_scores, counts)


def _deepep_experts_backward(ctx, grad_out):
    dispatcher = _combine_dispatchers[ctx.dispatcher_id]
    transfers = _transfers[ctx.key]
    recv_x, recv_idx, recv_scores, counts = ctx.saved_tensors
    # Recompute the experts while the output gradient is still arriving.
    with torch.enable_grad():
        recv_x = recv_x.detach().requires_grad_()
        recv_scores = recv_scores.detach().requires_grad_()
        routed = dispatcher.routed_experts(recv_x, recv_idx, recv_scores, counts.tolist(), ctx.score_before_experts)
    grad_routed, event = transfers.grad_received
    event.current_stream_wait()
    # Accumulates the experts' weight gradients like any other backward.
    torch.autograd.backward(routed, grad_routed)
    grad_x, grad_scores, event = _buffer.combine(
        x=recv_x.grad,
        handle=transfers.handle,
        topk_weights=recv_scores.grad,
        previous_event=_new_event_overlap(),
        async_finish=True,
        allocate_on_comm_stream=True,
    )
    transfers.grad_returned = (grad_x, grad_scores, event)
    # `deepep_dispatch_launch`'s backward returns `grad_returned`; these only order it after this one.
    return recv_x.grad, None, recv_scores.grad, None, None, None, None, None


deepep_experts.register_autograd(_deepep_experts_backward, setup_context=_deepep_experts_setup_context)


@torch.library.custom_op("prime_rl::deepep_combine_wait", mutates_args=())
def deepep_combine_wait(out: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """`deepep_experts`' output once it has arrived."""
    _transfers[key.item()].combined.current_stream_wait()
    return out.clone()


@deepep_combine_wait.register_fake
def _deepep_combine_wait_fake(out, key):
    return torch.empty_like(out)


def _deepep_combine_wait_setup_context(ctx, inputs, output) -> None:
    ctx.key = inputs[1].item()


def _deepep_combine_wait_backward(ctx, grad):
    transfers = _transfers[ctx.key]
    grad = grad.contiguous()
    grad_routed, _, _, _, _, event = _buffer.dispatch(
        x=grad,
        handle=transfers.handle,
        previous_event=_new_event_overlap(),
        async_finish=True,
        allocate_on_comm_stream=True,
    )
    transfers.grad_received = (grad_routed, event)
    return grad, None


deepep_combine_wait.register_autograd(_deepep_combine_wait_backward, setup_context=_deepep_combine_wait_setup_context)


register_deepep_cuda_ops()

__all__ = [
    "configure_num_sms",
    "DeepEPTokenDispatcher",
    "dispatch_tokens_async",
]
