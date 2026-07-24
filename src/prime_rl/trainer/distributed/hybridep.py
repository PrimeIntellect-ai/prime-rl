"""HybridEP: Expert Parallel Communication for GB200 NVLink72 Systems.

Ported from torchtitan.distributed.deepep.hybridep. Provides efficient token
dispatch/combine for MoE training via TMA-optimized all-to-all on GB200.
Uses a handle_id + cache pattern (like deepep.py) instead of CustomClassBase
for compatibility with the installed torch version.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

_buffer: Any = None

# Handle cache: maps int handle_id -> opaque deep_ep dispatch handle
_handle_cache: dict[int, Any] = {}
_handle_counter = 0


def _get_next_handle_id() -> torch.Tensor:
    global _handle_counter
    _handle_counter += 1
    return torch.tensor([_handle_counter], dtype=torch.int64, device="cpu")


@dataclass
class DispatchState:
    """State from dispatch needed for combine.

    Attributes:
        handle_id: CPU tensor used to retrieve the cached dispatch handle.
        permuted_scores: Routing scores applied to expert outputs in combine.
        num_tokens: Original input token count.
    """

    handle_id: torch.Tensor
    permuted_scores: torch.Tensor | None = None
    num_tokens: int = 0


def _num_permuted_tokens_for_non_blocking(
    num_tokens: int,
    ep_size: int,
    num_local_experts: int,
    top_k: int,
    moe_expert_capacity_factor: float,
    pad_multiple: int | None = None,
) -> int:
    n = int(num_tokens * ep_size * min(num_local_experts, top_k) * moe_expert_capacity_factor)
    if pad_multiple is not None:
        n = ((n + pad_multiple - 1) // pad_multiple) * pad_multiple
    return n


# ============================================================================
# Custom op registration
# ============================================================================

_lib = torch.library.Library("hybridep", "DEF")

_lib.define(
    "dispatch(Tensor x, Tensor topk_idx, Tensor topk_weights, "
    "int num_experts, int ep_size, str group_name, "
    "bool non_blocking, float? moe_expert_capacity_factor, int? pad_multiple) "
    "-> (Tensor, Tensor, Tensor, Tensor)"
)
_lib.define("combine(Tensor x, Tensor handle_id, int num_tokens, int? pad_multiple) -> Tensor")
_lib.define(
    "dispatch_bwd(Tensor grad_combined, Tensor handle_id, int num_permuted_tokens, int? pad_multiple) -> Tensor"
)
_lib.define(
    "combine_bwd(Tensor grad_hidden, Tensor grad_scores, Tensor handle_id, int num_tokens, int num_experts) -> (Tensor, Tensor)"
)


_DEEPEP_MULTINODE_NUM_SMS = 20


@torch.library.impl(_lib, "dispatch", "CUDA")
def _dispatch_op_impl(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    ep_size: int,
    group_name: str,
    non_blocking: bool,
    moe_expert_capacity_factor: float | None,
    pad_multiple: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """HybridEP dispatch with permute."""
    global _buffer
    num_local_experts = num_experts // ep_size

    group = dist.distributed_c10d._resolve_process_group(group_name)
    get_buffer(
        group=group,
        hidden_dim=x.shape[1],
        num_tokens=x.shape[0],
        num_local_experts=num_local_experts,
    )

    from deep_ep.hybrid_ep_buffer import indices_to_map

    routing_map, probs = indices_to_map(topk_idx, topk_weights.float(), x.shape[0], num_experts)

    num_permuted_tokens = None
    if non_blocking:
        assert moe_expert_capacity_factor is not None
        num_permuted_tokens = _num_permuted_tokens_for_non_blocking(
            x.shape[0],
            ep_size,
            num_local_experts,
            topk_idx.shape[1],
            moe_expert_capacity_factor,
            pad_multiple=pad_multiple,
        )

    hidden, scores, _, tokens_per_expert, handle = _buffer.dispatch_with_permute(
        hidden=x,
        routing_map=routing_map,
        probs=probs,
        scaling_factor=None,
        num_of_experts_per_rank=num_local_experts,
        pad_multiple=pad_multiple,
        num_permuted_tokens=num_permuted_tokens,
        non_blocking=non_blocking,
    )

    if scores is None:
        scores = torch.empty(0, device=x.device, dtype=torch.float32)
    if tokens_per_expert.device != x.device:
        tokens_per_expert = tokens_per_expert.to(x.device)

    handle_id = _get_next_handle_id()
    _handle_cache[handle_id.item()] = handle

    return hidden, scores, tokens_per_expert, handle_id


def _dispatch_setup_context(ctx, inputs, output):
    x, topk_idx, _, num_experts, *_ = inputs
    _, _, _, handle_id = output
    ctx.input_dtype = x.dtype
    ctx.num_experts = num_experts
    ctx.saved_handle_id = handle_id
    ctx.save_for_backward(topk_idx)


def _dispatch_backward(
    ctx,
    grad_hidden,
    grad_scores,
    grad_tpe,
    grad_handle_id,
):
    global _buffer
    if grad_hidden is None:
        return None, None, None, None, None, None, None, None, None

    handle = _handle_cache.pop(ctx.saved_handle_id.item(), None)
    assert handle is not None

    (topk_idx,) = ctx.saved_tensors

    grad_x, grad_probs_dense = _buffer.combine_with_unpermute(
        hidden=grad_hidden,
        probs=grad_scores if grad_scores.numel() > 0 else None,
        handle=handle,
    )
    grad_x = grad_x.to(ctx.input_dtype)

    if grad_probs_dense is None:
        grad_probs_dense = torch.empty(0, device=grad_hidden.device, dtype=torch.float32)

    grad_weights = grad_probs_dense.gather(dim=1, index=topk_idx) if grad_probs_dense.numel() > 0 else None
    return grad_x, None, grad_weights, None, None, None, None, None, None


@torch.library.impl(_lib, "combine", "CUDA")
def _combine_op_impl(
    x: torch.Tensor,
    handle_id: torch.Tensor,
    num_tokens: int,
    pad_multiple: int | None,
) -> torch.Tensor:
    global _buffer
    if _buffer is None:
        raise RuntimeError("HybridEP buffer not initialized.")

    handle = _handle_cache.get(handle_id.item())
    assert handle is not None, f"Handle not found for handle_id={handle_id.item()}"

    combined, _ = _buffer.combine_with_unpermute(hidden=x, handle=handle)
    return combined


def _combine_setup_context(ctx, inputs, output):
    x, handle_id, _num_tokens, pad_multiple = inputs
    ctx.saved_handle_id = handle_id
    ctx.num_permuted_tokens = x.shape[0]
    ctx.pad_multiple = pad_multiple


def _combine_backward(ctx, grad_combined):
    global _buffer
    if _buffer is None:
        raise RuntimeError("HybridEP buffer not initialized.")

    handle = _handle_cache.pop(ctx.saved_handle_id.item(), None)
    assert handle is not None

    grad_x, _, _, _, _ = _buffer.dispatch_with_permute(
        hidden=grad_combined,
        scaling_factor=None,
        handle=handle,
        num_permuted_tokens=ctx.num_permuted_tokens,
        pad_multiple=ctx.pad_multiple,
    )
    return grad_x, None, None, None


torch.library.register_autograd("hybridep::dispatch", _dispatch_backward, setup_context=_dispatch_setup_context)
torch.library.register_autograd("hybridep::combine", _combine_backward, setup_context=_combine_setup_context)


_NUM_SMS_DISPATCH = 16
_NUM_SMS_COMBINE = 16


def get_buffer(
    group: dist.ProcessGroup,
    hidden_dim: int,
    num_tokens: int,
    num_local_experts: int,
    fp8_dispatch: bool = False,
) -> None:
    """Ensure the global HybridEP buffer is initialized."""
    global _buffer

    if fp8_dispatch:
        raise AssertionError("HybridEP FP8 dispatch not yet supported")

    from deep_ep import HybridEPBuffer

    max_tokens_per_rank = num_tokens

    needs_reinit = (
        _buffer is None
        or _buffer.group != group
        or _buffer.configurer.buffer_config.hidden_dim < hidden_dim
        or _buffer.configurer.buffer_config.max_num_of_tokens_per_rank < max_tokens_per_rank
        or _buffer.configurer.buffer_config.num_of_experts_per_rank < num_local_experts
    )

    if needs_reinit:
        _buffer = HybridEPBuffer(
            group=group,
            hidden_dim=hidden_dim,
            max_num_of_tokens_per_rank=max_tokens_per_rank,
            num_local_experts=num_local_experts,
            use_fp8=fp8_dispatch,
            num_sms_dispatch_api=_NUM_SMS_DISPATCH,
            num_sms_combine_api=_NUM_SMS_COMBINE,
            load_cached_kernels=True,
            use_shared_buffer=True,
            enable_custom_allgather=True,
        )


def dispatch_tokens(
    hidden_states: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    top_scores: torch.Tensor,
    num_local_experts: int,
    num_experts: int,
    group: dist.ProcessGroup,
    non_blocking_expert_capacity_factor: float | None = None,
    pad_multiple: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, DispatchState]:
    """Dispatch tokens to experts via HybridEP all-to-all.

    Routing scores are applied to the expert outputs in combine_tokens,
    after expert computation.
    """
    non_blocking = non_blocking_expert_capacity_factor is not None

    selected_experts_indices = selected_experts_indices.contiguous()
    top_scores = top_scores.contiguous()

    ep_size = group.size()
    group_name = group.group_name

    (
        hidden,
        permuted_scores,
        tokens_per_expert,
        handle_id,
    ) = torch.ops.hybridep.dispatch(
        hidden_states,
        selected_experts_indices,
        top_scores,
        num_experts,
        ep_size,
        group_name,
        non_blocking,
        non_blocking_expert_capacity_factor,
        pad_multiple,
    )

    if permuted_scores is not None and permuted_scores.dtype != hidden.dtype:
        permuted_scores = permuted_scores.to(hidden.dtype)

    state = DispatchState(
        handle_id=handle_id,
        permuted_scores=permuted_scores,
        num_tokens=hidden_states.shape[0],
    )
    return hidden, tokens_per_expert, state


def combine_tokens(
    hidden_states: torch.Tensor,
    state: DispatchState,
    pad_multiple: int | None = None,
) -> torch.Tensor:
    """Combine expert outputs back to original token order."""
    if state.permuted_scores is not None:
        hidden_states = hidden_states * state.permuted_scores.reshape(-1, 1)

    return torch.ops.hybridep.combine(hidden_states, state.handle_id, state.num_tokens, pad_multiple)


__all__ = [
    "dispatch_tokens",
    "combine_tokens",
    "DispatchState",
]
