from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

import torch
from torch.distributed import ProcessGroup

from prime_rl.trainer.models.kernels.mega_moe_mxfp8 import (
    cast_weight_along_rows,
    mx_cast,
    mx_weight_sf,
    pack_ue8m0_k_major,
)
from prime_rl.trainer.models.layers.activations import Activation, ClampedSilu, Silu
from prime_rl.trainer.models.layers.expert_compute import to_local

if TYPE_CHECKING:
    from prime_rl.trainer.models.layers.moe import GroupedExperts


MegaMoePrecision = Literal["bf16", "mxfp8", "fp8", "fp8_bwd"]
# `deep_gemm` symbols each precision needs, named `<precision>_mega_moe[_backward]`; only the prime-mega-moe
# build has the training kernels
_REQUIRED_KERNELS: dict[str, tuple[str, ...]] = {
    "bf16": ("bf16_mega_moe", "bf16_mega_moe_backward"),
    "mxfp8": ("mxfp8_mega_moe", "mxfp8_mega_moe_backward"),
    "fp8": (
        "fp8_mega_moe",
        "fp8_mega_moe_backward",
        "quantize_weight_blockwise_fp8",
        "quantize_l1_weight_for_fp8_mega_moe",
    ),
    "fp8_bwd": ("bf16_mega_moe", "fp8_mega_moe_backward", "quantize_weight_blockwise_fp8"),
}
_MIN_CAPABILITY: dict[str, tuple[int, int]] = {
    "bf16": (9, 0),
    "mxfp8": (10, 0),
    "fp8": (9, 0),
    "fp8_bwd": (9, 0),
}


def mega_moe_available(precision: MegaMoePrecision = "bf16") -> bool:
    try:
        import deep_gemm
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability() < _MIN_CAPABILITY[precision]:
        return False
    return all(hasattr(deep_gemm, name) for name in _REQUIRED_KERNELS[precision])


def check_mega_moe_dims(hidden: int, intermediate_hidden: int) -> None:
    if hidden % 256 or intermediate_hidden % 128:
        raise ValueError(
            f"Mega MoE requires hidden ({hidden}) to be a multiple of 256 and intermediate_hidden "
            f"({intermediate_hidden}) a multiple of 128."
        )


def reserve_sms_for_comm(num_reserved_sms: int) -> int:
    """SMs the Mega MoE grids run on: ``total - num_reserved_sms``"""
    nccl_max_ctas = os.environ.get("NCCL_MAX_CTAS")
    if nccl_max_ctas is None or int(nccl_max_ctas) > num_reserved_sms:
        raise ValueError(
            f"Mega MoE reserves {num_reserved_sms} SMs for NCCL, so `NCCL_MAX_CTAS` must be set to at most "
            f"{num_reserved_sms} (got {nccl_max_ctas!r}); set it in `env_vars` or raise `num_reserved_sms`."
        )
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return max(total - num_reserved_sms, 1)


@contextmanager
def _deep_gemm_num_sms(num_sms: int):
    """``deep_gemm.set_num_sms`` is process-wide and read at launch"""
    import deep_gemm

    previous = deep_gemm.get_num_sms()
    deep_gemm.set_num_sms(num_sms)
    try:
        yield
    finally:
        deep_gemm.set_num_sms(previous)


@dataclass(frozen=True)
class MegaMoeBuffers:
    """Symmetric buffers the forward and backward kernels stage their inputs into."""

    forward: object
    backward: object


# Staging layout of each direction's symmetric buffer
_STAGING_MMA_TYPES: dict[str, tuple[str, str]] = {  # (forward, backward)
    "bf16": ("bf16xbf16", "bf16xbf16"),
    "mxfp8": ("fp8xfp8", "fp8xfp8"),
    "fp8": ("fp8xfp8", "bf16xbf16"),
    "fp8_bwd": ("bf16xbf16", "bf16xbf16"),
}


def build_mega_moe_buffers(
    group: ProcessGroup,
    num_experts: int,
    num_max_tokens_per_rank: int,
    top_k: int,
    hidden: int,
    intermediate_hidden: int,
    precision: MegaMoePrecision,
) -> MegaMoeBuffers:
    import deep_gemm

    check_mega_moe_dims(hidden, intermediate_hidden)
    args = (group, num_experts, num_max_tokens_per_rank, top_k, hidden, intermediate_hidden)
    forward_mma_type, backward_mma_type = _STAGING_MMA_TYPES[precision]
    backward = deep_gemm.SymmBuffer(*args, mma_type=backward_mma_type)
    if forward_mma_type == backward_mma_type:
        return MegaMoeBuffers(forward=backward, backward=backward)
    # Both directions restage their inputs on every call, so the forward layout can view the backward's allocation instead of allocating a second symmetric buffer
    forward = deep_gemm.SymmBuffer(*args, mma_type=forward_mma_type, base=backward)
    return MegaMoeBuffers(forward=forward, backward=backward)


# Custom ops take an int key: the buffers are not tensors
_BUFFER_REGISTRY: dict[int, tuple[MegaMoePrecision, MegaMoeBuffers, int]] = {}


def register_mega_moe_buffers(buffers: MegaMoeBuffers, precision: MegaMoePrecision, num_sms: int) -> int:
    key = id(buffers)
    _BUFFER_REGISTRY[key] = (precision, buffers, num_sms)
    return key


def _stage_routing(buffer, num_tokens: int, topk_idx: torch.Tensor, topk_weights: torch.Tensor) -> None:
    buffer.topk_idx[:num_tokens].copy_(topk_idx.view(num_tokens, buffer.num_topk))
    buffer.topk_weights[:num_tokens].copy_(topk_weights.view(num_tokens, buffer.num_topk))


def _stage_inputs(buffer, x: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor) -> None:
    num_tokens = x.shape[0]
    buffer.x[:num_tokens].copy_(x)
    _stage_routing(buffer, num_tokens, topk_idx, topk_weights)


def _grad_buffers(x: torch.Tensor, buffer, dw1_shape, dw2_shape, dw_dtype: torch.dtype):
    num_tokens, hidden = x.shape
    dx = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device=x.device)
    dw1 = torch.empty(dw1_shape, dtype=dw_dtype, device=x.device)
    dw2 = torch.empty(dw2_shape, dtype=dw_dtype, device=x.device)
    dtopk = torch.empty((num_tokens, buffer.num_topk), dtype=torch.float32, device=x.device)
    return dx, dw1, dw2, dtopk


# BF16


@dataclass
class MegaMoeBF16ExpertWeights:
    l1: torch.Tensor
    """Gate/up concatenated ``[E, 2I, H]``."""
    l2: torch.Tensor
    """``[E, H, I]``."""

    @classmethod
    def build(cls, gate_up_proj: torch.Tensor, down_proj: torch.Tensor) -> MegaMoeBF16ExpertWeights:
        return cls(l1=gate_up_proj.to(torch.bfloat16).contiguous(), l2=down_proj.to(torch.bfloat16).contiguous())


def mega_moe_forward_bf16(x, topk_idx, topk_weights, gate_up_proj, down_proj, buffer, activation_clamp):
    import deep_gemm

    weights = MegaMoeBF16ExpertWeights.build(gate_up_proj, down_proj)
    _stage_inputs(buffer, x, topk_idx, topk_weights)
    y = torch.empty_like(x)
    deep_gemm.bf16_mega_moe(y, weights.l1, weights.l2, buffer, activation_clamp=activation_clamp)
    return y


def mega_moe_backward_bf16(dy, x, topk_idx, topk_weights, gate_up_proj, down_proj, buffer, dw_dtype, activation_clamp):
    import deep_gemm

    weights = MegaMoeBF16ExpertWeights.build(gate_up_proj, down_proj)
    _stage_inputs(buffer, x, topk_idx, topk_weights)
    dx, dw1, dw2, dtopk = _grad_buffers(x, buffer, weights.l1.shape, weights.l2.shape, dw_dtype)
    deep_gemm.bf16_mega_moe_backward(
        dx, dw1, dw2, dtopk, dy, weights.l1, weights.l2, buffer, activation_clamp=activation_clamp
    )
    return dx, dw1, dw2, dtopk


# MXFP8 (SM100): E4M3 with 1x32 UE8M0 block scales


@dataclass
class MegaMoeMXFP8ExpertWeights:
    l1: tuple[torch.Tensor, torch.Tensor]
    """Interleaved ``[E, 2I, H]`` scaled along H (forward L1 operand, backward Z recompute)."""
    l2: tuple[torch.Tensor, torch.Tensor] | None
    """``[E, H, I]`` scaled along I (forward L2 operand)."""
    l1_t: tuple[torch.Tensor, torch.Tensor] | None
    """Interleaved ``[E, 2I, H]`` scaled along 2I (backward dX)."""
    l2_t: tuple[torch.Tensor, torch.Tensor] | None
    """``[E, H, I]`` scaled along H (backward dz = dy @ W2)."""

    @classmethod
    def build(
        cls, gate_up_proj: torch.Tensor, down_proj: torch.Tensor, for_backward: bool
    ) -> MegaMoeMXFP8ExpertWeights:
        from deep_gemm.mega import _interleave_weights

        num_experts, two_i, hidden = gate_up_proj.shape
        inter = two_i // 2
        l1_bf16 = _interleave_weights(gate_up_proj.contiguous())
        l1_fp8, l1_sf = mx_cast(l1_bf16.view(num_experts * two_i, hidden), along_rows=False)
        l1 = (
            l1_fp8.view(num_experts, two_i, hidden),
            mx_weight_sf(l1_sf.view(num_experts, two_i, -1), two_i, hidden),
        )
        if not for_backward:
            down = down_proj.contiguous()
            l2_fp8, l2_sf = mx_cast(down.view(num_experts * hidden, inter), along_rows=False)
            l2 = (
                l2_fp8.view(num_experts, hidden, inter),
                mx_weight_sf(l2_sf.view(num_experts, hidden, -1), hidden, inter),
            )
            return cls(l1=l1, l2=l2, l1_t=None, l2_t=None)
        l1_t_fp8, l1_t_sf = cast_weight_along_rows(l1_bf16)
        l1_t = (l1_t_fp8, mx_weight_sf(l1_t_sf, hidden, two_i))
        l2_t_fp8, l2_t_sf = cast_weight_along_rows(down_proj.contiguous())
        l2_t = (l2_t_fp8, mx_weight_sf(l2_t_sf, inter, hidden))
        return cls(l1=l1, l2=None, l1_t=l1_t, l2_t=l2_t)


def mega_moe_forward_mxfp8(x, topk_idx, topk_weights, gate_up_proj, down_proj, buffer, activation_clamp):
    import deep_gemm

    weights = MegaMoeMXFP8ExpertWeights.build(gate_up_proj, down_proj, for_backward=False)
    num_tokens = x.shape[0]
    x_fp8, x_sf = mx_cast(x, along_rows=False)
    buffer.x[:num_tokens].copy_(x_fp8)
    buffer.x_sf[:num_tokens].copy_(pack_ue8m0_k_major(x_sf))
    _stage_routing(buffer, num_tokens, topk_idx, topk_weights)
    y = torch.empty_like(x)
    deep_gemm.mxfp8_mega_moe(y, weights.l1, weights.l2, buffer, activation_clamp=activation_clamp)
    return y


def mega_moe_backward_mxfp8(dy, x, topk_idx, topk_weights, gate_up_proj, down_proj, buffer, dw_dtype, activation_clamp):
    import deep_gemm

    weights = MegaMoeMXFP8ExpertWeights.build(gate_up_proj, down_proj, for_backward=True)
    _stage_routing(buffer, x.shape[0], topk_idx, topk_weights)
    # The kernel writes dW1 in the natural [gate | up] row order, undoing the L1 interleave
    dx, dw1, dw2, dtopk = _grad_buffers(x, buffer, weights.l1[0].shape, weights.l2_t[0].shape, dw_dtype)
    deep_gemm.mxfp8_mega_moe_backward(
        dx, dw1, dw2, dtopk, dy, x, weights.l1, weights.l1_t, weights.l2_t, buffer, activation_clamp=activation_clamp
    )
    return dx, dw1, dw2, dtopk


# FP8 (SM90): E4M3 with 128x128 fp32 block scales for weights, 1x128 for tokens


@dataclass
class MegaMoeFP8ExpertWeights:
    l1: tuple[torch.Tensor, torch.Tensor]
    """``[E, 2I, H]`` natural gate/up rows. Forward: one scale per 64 gate + 64 up rows x 128 columns
    (``[E, I/64, H/128]``); backward: ``[E, 2I/128, H/128]``."""
    l2: tuple[torch.Tensor, torch.Tensor] | None
    """``[E, H, I]`` with ``[E, H/128, I/128]`` scales (forward L2 operand)."""
    l2_t: tuple[torch.Tensor, torch.Tensor] | None
    """W2 transposed ``[E, I, H]`` with ``[E, I/128, H/128]`` scales (backward dh = dy @ W2)."""
    l1_t: tuple[torch.Tensor, torch.Tensor] | None
    """W1 transposed ``[E, H, 2I]`` with ``[E, H/128, 2I/128]`` scales (backward dX)."""

    @classmethod
    def build(cls, gate_up_proj: torch.Tensor, down_proj: torch.Tensor, for_backward: bool) -> MegaMoeFP8ExpertWeights:
        import deep_gemm

        gate_up = gate_up_proj.to(torch.bfloat16).contiguous()
        down = down_proj.to(torch.bfloat16).contiguous()
        if not for_backward:
            return cls(
                l1=deep_gemm.quantize_l1_weight_for_fp8_mega_moe(gate_up),
                l2=deep_gemm.quantize_weight_blockwise_fp8(down),
                l2_t=None,
                l1_t=None,
            )
        return cls(
            l1=deep_gemm.quantize_weight_blockwise_fp8(gate_up),
            l2=None,
            l2_t=deep_gemm.quantize_weight_blockwise_fp8(down, transpose=True),
            l1_t=deep_gemm.quantize_weight_blockwise_fp8(gate_up, transpose=True),
        )


def _quant_rows_1x128(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    t, h = x.shape
    blocks = x.view(t, h // 128, 128)
    scale = blocks.abs().amax(dim=2, keepdim=True).float().clamp(min=1e-12) / 448.0
    q = (blocks.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(t, h)
    return q, scale.view(t, h // 128)


def mega_moe_forward_fp8(x, topk_idx, topk_weights, gate_up_proj, down_proj, buffer, activation_clamp):
    import deep_gemm

    weights = MegaMoeFP8ExpertWeights.build(gate_up_proj, down_proj, for_backward=False)
    num_tokens = x.shape[0]
    x_q, x_sf = _quant_rows_1x128(x)
    buffer.x[:num_tokens].copy_(x_q)
    buffer.x_sf.view(torch.float32)[:num_tokens].copy_(x_sf)
    _stage_routing(buffer, num_tokens, topk_idx, topk_weights)
    y = torch.empty_like(x)
    deep_gemm.fp8_mega_moe(y, weights.l1, weights.l2, buffer, activation_clamp=activation_clamp)
    return y


def mega_moe_backward_fp8(dy, x, topk_idx, topk_weights, gate_up_proj, down_proj, buffer, dw_dtype, activation_clamp):
    import deep_gemm

    weights = MegaMoeFP8ExpertWeights.build(gate_up_proj, down_proj, for_backward=True)
    _stage_inputs(buffer, x, topk_idx, topk_weights)
    l2_t = weights.l2_t[0]
    dx, dw1, dw2, dtopk = _grad_buffers(
        x, buffer, weights.l1[0].shape, (l2_t.shape[0], l2_t.shape[2], l2_t.shape[1]), dw_dtype
    )
    deep_gemm.fp8_mega_moe_backward(
        dx, dw1, dw2, dtopk, dy, *weights.l1, *weights.l2_t, *weights.l1_t, buffer, activation_clamp=activation_clamp
    )
    return dx, dw1, dw2, dtopk


MEGA_MOE_IMPLS: dict[str, tuple[Callable, Callable]] = {
    "bf16": (mega_moe_forward_bf16, mega_moe_backward_bf16),
    "mxfp8": (mega_moe_forward_mxfp8, mega_moe_backward_mxfp8),
    "fp8": (mega_moe_forward_fp8, mega_moe_backward_fp8),
    "fp8_bwd": (mega_moe_forward_bf16, mega_moe_backward_fp8),
}


def _dw_dtype(weight: torch.Tensor) -> torch.dtype:
    return weight.dtype if weight.dtype in (torch.bfloat16, torch.float32) else torch.float32


@torch.library.custom_op("prime_rl::mega_moe_forward", mutates_args=())
def mega_moe_forward_op(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    buffer_key: int,
    activation_clamp: float | None = None,
) -> torch.Tensor:
    precision, buffers, num_sms = _BUFFER_REGISTRY[buffer_key]
    forward, _ = MEGA_MOE_IMPLS[precision]
    with _deep_gemm_num_sms(num_sms):
        y = forward(
            x.to(torch.bfloat16).contiguous(),
            topk_idx,
            topk_weights,
            gate_up_proj,
            down_proj,
            buffers.forward,
            activation_clamp,
        )
    return y.to(x.dtype)


@mega_moe_forward_op.register_fake
def _mega_moe_forward_fake(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    buffer_key: int,
    activation_clamp: float | None = None,
) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("prime_rl::mega_moe_backward", mutates_args=())
def mega_moe_backward_op(
    dy: torch.Tensor,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    buffer_key: int,
    activation_clamp: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    precision, buffers, num_sms = _BUFFER_REGISTRY[buffer_key]
    _, backward = MEGA_MOE_IMPLS[precision]
    with _deep_gemm_num_sms(num_sms):
        dx, dw1, dw2, dtopk = backward(
            dy.to(torch.bfloat16).contiguous(),
            x.to(torch.bfloat16).contiguous(),
            topk_idx,
            topk_weights,
            gate_up_proj,
            down_proj,
            buffers.backward,
            _dw_dtype(gate_up_proj),
            activation_clamp,
        )
    return dx.to(x.dtype), dw1.to(gate_up_proj.dtype), dw2.to(down_proj.dtype), dtopk


@mega_moe_backward_op.register_fake
def _mega_moe_backward_fake(
    dy: torch.Tensor,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    buffer_key: int,
    activation_clamp: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(x),
        torch.empty_like(gate_up_proj),
        torch.empty_like(down_proj),
        torch.empty_like(topk_weights, dtype=torch.float32),
    )


def _mega_moe_setup_context(ctx, inputs, output) -> None:
    x, topk_idx, topk_weights, gate_up_proj, down_proj, buffer_key, activation_clamp = inputs
    ctx.save_for_backward(x, topk_idx, topk_weights, gate_up_proj, down_proj)
    ctx.buffer_key = buffer_key
    ctx.activation_clamp = activation_clamp


def _mega_moe_backward(ctx, grad_y: torch.Tensor):
    x, topk_idx, topk_weights, gate_up_proj, down_proj = ctx.saved_tensors
    dx, dw1, dw2, dtopk = torch.ops.prime_rl.mega_moe_backward(
        grad_y, x, topk_idx, topk_weights, gate_up_proj, down_proj, ctx.buffer_key, ctx.activation_clamp
    )
    return dx, None, dtopk, dw1, dw2, None, None


mega_moe_forward_op.register_autograd(_mega_moe_backward, setup_context=_mega_moe_setup_context)


def _activation_clamp(activation: type[Activation] | Activation) -> float | None:
    if activation is Silu or isinstance(activation, Silu):
        return None
    if isinstance(activation, ClampedSilu):
        return float(activation.limit)
    raise ValueError("Mega MoE requires a SwiGLU (`silu` or DeepSeek V4 `ClampedSilu`) expert activation.")


class MegaMoEExpertCompute:
    """One instance serves every MoE layer: the symmetric buffers are sized once from the shared
    expert shapes and routing, and ``validate`` checks each layer against them. Its kernels run on
    ``num_sms`` SMs (see ``reserve_sms_for_comm``)."""

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden: int,
        intermediate_hidden: int,
        group: ProcessGroup,
        max_tokens_per_rank: int,
        num_sms: int,
        precision: MegaMoePrecision = "bf16",
    ) -> None:
        if not mega_moe_available(precision):
            kernels = " and ".join(f"`{name}`" for name in _REQUIRED_KERNELS[precision])
            raise RuntimeError(
                f"Mega MoE `{precision}` requires an SM{_MIN_CAPABILITY[precision][0]}0+ GPU and the "
                "prime-mega-moe `deep_gemm` wheel from Prime Intellect installed over the public one; "
                f"this build does not expose {kernels}."
            )
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.max_tokens_per_rank = max_tokens_per_rank
        buffers = build_mega_moe_buffers(
            group, num_experts, max_tokens_per_rank, top_k, hidden, intermediate_hidden, precision
        )
        # Registered once here: a registry write inside the compiled, checkpointed block is a side effect dynamo rejects.
        self.buffer_key = register_mega_moe_buffers(buffers, precision, num_sms)

    def validate(self, experts: "GroupedExperts") -> None:
        if experts.gate_proj is None and experts.gate_up_proj is None:
            raise ValueError("Mega MoE requires gated experts (SwiGLU gate+up), got non-gated experts.")
        if any(bias is not None for bias in (experts.gate_proj_bias, experts.up_proj_bias, experts.down_proj_bias)):
            raise ValueError("Mega MoE does not support expert biases.")
        _activation_clamp(experts.activation)
        shape = (experts.num_experts, experts.down_proj.shape[1], experts.hidden_dim)
        expected = (self.num_experts, self.hidden, self.intermediate_hidden)
        if shape != expected:
            raise ValueError(
                "Mega MoE's buffers are sized for (num_experts, hidden, intermediate_hidden)="
                f"{expected}; got experts with {shape}."
            )

    def destroy(self) -> None:
        _, buffers, _ = _BUFFER_REGISTRY.pop(self.buffer_key)
        if buffers.forward is not buffers.backward:
            buffers.forward.destroy()
        buffers.backward.destroy()

    def validate_top_k(self, top_k: int) -> None:
        if top_k != self.top_k:
            raise ValueError(f"Mega MoE's buffers are sized for top_k={self.top_k}; got a router with top_k={top_k}.")

    def __call__(
        self,
        experts: "GroupedExperts",
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> torch.Tensor:
        # The trainer configs reject a max_tokens_per_rank below the micro batch, so every rank fits;
        # this only catches direct use with a smaller buffer
        num_tokens = x.shape[0]
        if num_tokens > self.max_tokens_per_rank:
            raise RuntimeError(
                f"Mega MoE buffer is sized for {self.max_tokens_per_rank} tokens/rank, got {num_tokens}. "
                "Raise `model.moe.dispatch.max_tokens_per_rank`."
            )
        if experts.gate_up_proj is not None:
            gate_up_proj = to_local(experts.gate_up_proj)
        else:
            gate_up_proj = torch.cat([to_local(experts.gate_proj), to_local(experts.up_proj)], dim=1)
        return torch.ops.prime_rl.mega_moe_forward(
            x,
            selected_experts_indices.to(torch.int64),
            top_scores.to(torch.float32),
            gate_up_proj,
            to_local(experts.down_proj),
            self.buffer_key,
            _activation_clamp(experts.activation),
        )
