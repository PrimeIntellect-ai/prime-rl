from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DTensor

from prime_rl.trainer.models.layers.activations import Silu

if TYPE_CHECKING:
    from prime_rl.trainer.models.layers.moe import GroupedExperts


def mega_moe_available() -> bool:
    try:
        import deep_gemm
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    return (
        torch.cuda.get_device_capability() >= (10, 0)
        and hasattr(deep_gemm, "bf16_mega_moe")
        and hasattr(deep_gemm, "bf16_mega_moe_backward")
    )


def check_mega_moe_dims(hidden: int, intermediate_hidden: int) -> None:
    if hidden % 256 or intermediate_hidden % 128:
        raise ValueError(
            f"Mega MoE requires hidden ({hidden}) to be a multiple of 256 and intermediate_hidden "
            f"({intermediate_hidden}) a multiple of 128."
        )


@dataclass
class MegaMoeExpertWeights:
    l1: torch.Tensor
    l2: torch.Tensor


def prepare_mega_moe_weights(gate_up_proj: torch.Tensor, down_proj: torch.Tensor) -> MegaMoeExpertWeights:
    import deep_gemm

    l1, l2 = deep_gemm.transform_weights_for_mega_moe(
        gate_up_proj.to(torch.bfloat16).contiguous(), down_proj.to(torch.bfloat16).contiguous()
    )
    return MegaMoeExpertWeights(l1=l1, l2=l2)


# Mega MoE's BF16 L1 layout interleaves gate/up rows in groups of 8 ([g0..7, u0..7, g8..15, ...])
# instead of [gate | up]; L2 is used as is. Storing the fused `gate_up_proj` parameter in that
# layout makes the weight transform a no-op, so the kernels read the (FSDP-unsharded) parameter
# directly instead of a per-layer ~1.5 GiB copy, and the backward writes dW1 in the same layout.
MEGA_MOE_INTERLEAVE_GRAN = 8


def _interleave_gate_up(t: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    num_experts, n, *rest = t.shape
    half = n // 2
    grouped = t.view(num_experts, half // MEGA_MOE_INTERLEAVE_GRAN, 2, MEGA_MOE_INTERLEAVE_GRAN, *rest)
    natural = t.view(num_experts, 2, half // MEGA_MOE_INTERLEAVE_GRAN, MEGA_MOE_INTERLEAVE_GRAN, *rest)
    return (natural.transpose(1, 2) if not inverse else grouped.transpose(1, 2)).reshape(t.shape)


def _reorder_gate_up_(tensor: torch.Tensor, inverse: bool) -> None:
    from torch.distributed.tensor import Shard

    if isinstance(tensor, DTensor):
        if any(isinstance(p, Shard) and p.dim == 1 for p in tensor.placements):
            raise ValueError(
                "Mega MoE's interleaved gate_up layout requires expert weights sharded on dim 0 "
                "(disable `model.fusions.shard_fused_on_dim1`)."
            )
        tensor = tensor.to_local()
    if tensor.numel() == 0:
        return
    tensor.copy_(_interleave_gate_up(tensor, inverse=inverse))


def _mega_moe_experts(model: torch.nn.Module):
    from prime_rl.trainer.models.layers.moe import GroupedExperts

    for module in model.modules():
        if isinstance(module, GroupedExperts) and isinstance(module.compute, MegaMoEExpertCompute):
            if module.gate_up_proj is not None:
                yield module


@torch.no_grad()
def set_mega_moe_weight_layout(
    model: torch.nn.Module, optimizers: list[torch.optim.Optimizer], interleaved: bool
) -> None:
    """Reorder every Mega MoE layer's fused `gate_up_proj` (and its same-shaped optimizer state)
    between the natural [gate | up] layout used by checkpoints/weight broadcasts and the kernel's
    interleaved layout used during training. Idempotent per layer."""
    for experts in _mega_moe_experts(model):
        if getattr(experts, "mega_moe_interleaved", False) == interleaved:
            continue
        param = experts.gate_up_proj
        _reorder_gate_up_(param, inverse=not interleaved)
        for optimizer in optimizers:
            for value in getattr(optimizer, "state", {}).get(param, {}).values():
                if isinstance(value, torch.Tensor) and value.shape == param.shape:
                    _reorder_gate_up_(value, inverse=not interleaved)
        experts.mega_moe_interleaved = interleaved


@contextmanager
def natural_mega_moe_weight_layout(model: torch.nn.Module, optimizers: list[torch.optim.Optimizer]):
    """Temporarily restore the natural [gate | up] layout (for checkpoint saves and weight broadcasts)."""
    set_mega_moe_weight_layout(model, optimizers, interleaved=False)
    try:
        yield
    finally:
        set_mega_moe_weight_layout(model, optimizers, interleaved=True)


def reserve_sms_for_comm(num_reserved_sms: int) -> None:
    import deep_gemm

    total = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    deep_gemm.set_num_sms(max(total - num_reserved_sms, 1))


_BUFFER_CACHE: dict[tuple, object] = {}
_BUFFER_REGISTRY: dict[int, object] = {}


def register_mega_moe_buffer(buffer) -> int:
    key = id(buffer)
    _BUFFER_REGISTRY[key] = buffer
    return key


def build_mega_moe_buffer(
    group: ProcessGroup,
    num_experts: int,
    num_max_tokens_per_rank: int,
    top_k: int,
    hidden: int,
    intermediate_hidden: int,
):
    import deep_gemm

    check_mega_moe_dims(hidden, intermediate_hidden)
    key = (id(group), num_experts, num_max_tokens_per_rank, top_k, hidden, intermediate_hidden)
    buffer = _BUFFER_CACHE.get(key)
    if buffer is None:
        buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            group, num_experts, num_max_tokens_per_rank, top_k, hidden, intermediate_hidden, mma_type="bf16xbf16"
        )
        _BUFFER_CACHE[key] = buffer
    return buffer


def _stage_inputs(buffer, x: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor) -> None:
    num_tokens = x.shape[0]
    buffer.x[:num_tokens].copy_(x)
    buffer.topk_idx[:num_tokens].copy_(topk_idx.view(num_tokens, buffer.num_topk))
    buffer.topk_weights[:num_tokens].copy_(topk_weights.view(num_tokens, buffer.num_topk))


def mega_moe_forward(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: MegaMoeExpertWeights,
    buffer,
    activation_clamp: float | None = None,
) -> torch.Tensor:
    """Fused dispatch + SwiGLU MLP + combine for this rank's raw (pre-dispatch) bf16 tokens ``x``.
    Router weights are applied at combine time. ``activation_clamp`` clamps gate to ``<= clamp`` and
    up to ``[-clamp, clamp]`` before the SwiGLU. Returns bf16 ``(num_tokens, hidden)``."""
    import deep_gemm

    num_tokens, hidden = x.shape
    _stage_inputs(buffer, x, topk_idx, topk_weights)
    y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device=x.device)
    deep_gemm.bf16_mega_moe(y, weights.l1, weights.l2, buffer, activation_clamp=activation_clamp)
    return y


# The forward as a custom op, so selective activation checkpointing can save its output and skip the fused dispatch + expert compute + combine during recompute.
@torch.library.custom_op("prime_rl::mega_moe_forward", mutates_args=())
def mega_moe_forward_op(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    l1: torch.Tensor,
    l2: torch.Tensor,
    buffer_key: int,
    activation_clamp: float | None = None,
) -> torch.Tensor:
    return mega_moe_forward(
        x, topk_idx, topk_weights, MegaMoeExpertWeights(l1=l1, l2=l2), _BUFFER_REGISTRY[buffer_key], activation_clamp
    )


@mega_moe_forward_op.register_fake
def _mega_moe_forward_fake(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    l1: torch.Tensor,
    l2: torch.Tensor,
    buffer_key: int,
    activation_clamp: float | None = None,
) -> torch.Tensor:
    return torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)


def mega_moe_backward(
    dy: torch.Tensor,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: MegaMoeExpertWeights,
    buffer,
    dw_dtype: torch.dtype,
    dw_natural_layout: bool = True,
    activation_clamp: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    import deep_gemm

    num_tokens, hidden = x.shape
    _stage_inputs(buffer, x, topk_idx, topk_weights)
    dx = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device=x.device)
    dw1 = torch.empty(weights.l1.shape, dtype=dw_dtype, device=x.device)
    dw2 = torch.empty(weights.l2.shape, dtype=dw_dtype, device=x.device)
    dtopk = torch.empty((num_tokens, buffer.num_topk), dtype=torch.float32, device=x.device)
    deep_gemm.bf16_mega_moe_backward(
        dx,
        dw1,
        dw2,
        dtopk,
        dy,
        weights.l1,
        weights.l2,
        buffer,
        dw_natural_layout=dw_natural_layout,
        activation_clamp=activation_clamp,
    )
    return dx, dw1, dw2, dtopk


def _to_local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _activation_clamp(activation) -> float | None:
    """The kernel's SwiGLU clamp for a supported expert activation: None for plain SwiGLU, the
    limit for DeepSeek V4's clamped SwiGLU (gate <= limit, up in [-limit, limit])."""
    from prime_rl.trainer.models.deepseek_v4.moe import ClampedSwiglu

    if activation is Silu:
        return None
    if isinstance(activation, ClampedSwiglu):
        return float(activation.limit)
    raise ValueError("Mega MoE requires a SwiGLU (`silu` or DeepSeek V4 clamped) expert activation.")


class _MegaMoeRoutedExperts(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        gate_up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        buffer,
        interleaved: bool,
        activation_clamp: float | None,
    ) -> torch.Tensor:
        x_bf16 = x.to(torch.bfloat16).contiguous()
        topk_idx = selected_experts_indices.to(torch.int64)
        topk_weights = top_scores.to(torch.float32)
        if interleaved:
            weights = MegaMoeExpertWeights(
                l1=gate_up_proj.to(torch.bfloat16).contiguous(), l2=down_proj.to(torch.bfloat16).contiguous()
            )
        else:
            weights = prepare_mega_moe_weights(gate_up_proj, down_proj)
        y = torch.ops.prime_rl.mega_moe_forward(
            x_bf16, topk_idx, topk_weights, weights.l1, weights.l2, register_mega_moe_buffer(buffer), activation_clamp
        )
        ctx.save_for_backward(x_bf16, topk_idx, topk_weights, weights.l1, weights.l2)
        ctx.buffer = buffer
        ctx.interleaved = interleaved
        ctx.activation_clamp = activation_clamp
        ctx.dw_dtype = gate_up_proj.dtype if gate_up_proj.dtype in (torch.bfloat16, torch.float32) else torch.float32
        ctx.x_dtype, ctx.scores_dtype, ctx.scores_shape = x.dtype, top_scores.dtype, top_scores.shape
        return y.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        x_bf16, topk_idx, topk_weights, l1, l2 = ctx.saved_tensors
        dx, dl1, dl2, dtopk = mega_moe_backward(
            grad_y.to(torch.bfloat16).contiguous(),
            x_bf16,
            topk_idx,
            topk_weights,
            MegaMoeExpertWeights(l1=l1, l2=l2),
            ctx.buffer,
            ctx.dw_dtype,
            dw_natural_layout=not ctx.interleaved,
            activation_clamp=ctx.activation_clamp,
        )
        return (
            dx.to(ctx.x_dtype),
            dtopk.reshape(ctx.scores_shape).to(ctx.scores_dtype),
            None,
            dl1,
            dl2,
            None,
            None,
            None,
        )


class MegaMoEExpertCompute:
    """Fused Mega MoE dispatch + SwiGLU expert MLP + combine, forward and backward.

    It runs inside the experts' forward, so FSDP has already unsharded their weights. Pair it with
    a ``FusedTokenDispatcher``. ``num_experts`` counts the experts across the whole expert-parallel group.
    """

    def __init__(
        self,
        experts: "GroupedExperts",
        num_experts: int,
        top_k: int,
        group: ProcessGroup,
        max_tokens_per_rank: int,
        num_reserved_sms: int = 16,
    ) -> None:
        if not mega_moe_available():
            raise RuntimeError(
                "Mega MoE requires DeepGEMM's Mega MoE kernels (SM100+/Blackwell and a "
                "deep_gemm build with `bf16_mega_moe` and `bf16_mega_moe_backward`)."
            )
        if experts.gate_proj is None and experts.gate_up_proj is None:
            raise ValueError("Mega MoE requires gated experts (SwiGLU gate+up), got non-gated experts.")
        if any(bias is not None for bias in (experts.gate_proj_bias, experts.up_proj_bias, experts.down_proj_bias)):
            raise ValueError("Mega MoE does not support expert biases.")
        hidden = experts.down_proj.shape[1]
        check_mega_moe_dims(hidden, experts.hidden_dim)
        reserve_sms_for_comm(num_reserved_sms)

        self.activation_clamp = _activation_clamp(experts.activation)
        self.max_tokens_per_rank = max_tokens_per_rank
        self.buffer = build_mega_moe_buffer(group, num_experts, max_tokens_per_rank, top_k, hidden, experts.hidden_dim)

    def __call__(
        self,
        experts: "GroupedExperts",
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = x.shape[0]
        if num_tokens > self.max_tokens_per_rank:
            raise RuntimeError(
                f"Mega MoE buffer is sized for {self.max_tokens_per_rank} tokens/rank, got {num_tokens}. "
                "Raise `model.moe.dispatch.max_tokens_per_rank`."
            )
        if experts.gate_up_proj is not None:
            gate_up_proj = _to_local(experts.gate_up_proj)
            interleaved = getattr(experts, "mega_moe_interleaved", False)
        else:
            gate_up_proj = torch.cat([_to_local(experts.gate_proj), _to_local(experts.up_proj)], dim=1)
            interleaved = False
        return _MegaMoeRoutedExperts.apply(
            x,
            top_scores,
            selected_experts_indices,
            gate_up_proj,
            _to_local(experts.down_proj),
            self.buffer,
            interleaved,
            self.activation_clamp,
        )
