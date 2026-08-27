# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor

from prime_rl.configs.trainer import EPCommBackend
from prime_rl.trainer.distributed.expert_parallel import expert_parallel
from prime_rl.trainer.models.layers.activations import Activation, ActivationDispatch, ActivationType
from prime_rl.trainer.models.layers.mlp import ExpertType, FeedForward

ScoreFuncType = Literal["softmax", "sigmoid", "topk_softmax"]


@dataclass
class MoEArgs:
    num_experts: int = 8
    num_shared_experts: int = 1

    # experts
    expert_type: ExpertType = "gated"
    activation: ActivationType = "silu"

    # router
    score_func: ScoreFuncType = "sigmoid"
    route_norm: bool = False
    route_scale: float = 1.0
    score_before_experts: bool = True

    # token-choice
    top_k: int = 1
    load_balance_coeff: float | None = 1e-3
    fp8: bool = False  # use FP8 grouped GEMM via DeepGEMM (requires SM90)

    def __post_init__(self) -> None:
        ActivationDispatch[self.activation]


def _broadcast_expert_bias(
    bias: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    target_rows: int,
) -> torch.Tensor:
    bias = torch.repeat_interleave(bias, num_tokens_per_expert.to(torch.int64), dim=0)
    if bias.shape[0] < target_rows:
        bias = F.pad(bias, (0, 0, 0, target_rows - bias.shape[0]))
    return bias


@expert_parallel
def _run_grouped_experts(
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    *,
    gate_proj: torch.Tensor | None,
    grouped_mm_fn: Callable[..., torch.Tensor],
    activation: type[Activation],
    gate_proj_bias: torch.Tensor | None,
    up_proj_bias: torch.Tensor | None,
    down_proj_bias: torch.Tensor | None,
) -> torch.Tensor:
    return _run_grouped_experts_impl(
        up_proj,
        down_proj,
        x,
        num_tokens_per_expert,
        gate_proj=gate_proj,
        grouped_mm_fn=grouped_mm_fn,
        activation=activation,
        gate_proj_bias=gate_proj_bias,
        up_proj_bias=up_proj_bias,
        down_proj_bias=down_proj_bias,
    )


def _run_grouped_experts_impl(
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    *,
    gate_proj: torch.Tensor | None,
    grouped_mm_fn: Callable[..., torch.Tensor],
    activation: type[Activation],
    gate_proj_bias: torch.Tensor | None,
    up_proj_bias: torch.Tensor | None,
    down_proj_bias: torch.Tensor | None,
) -> torch.Tensor:
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    assert x.dim() == 2

    x_bf16 = x.bfloat16()
    up = grouped_mm_fn(x_bf16, up_proj.bfloat16(), offs=offsets)
    if up_proj_bias is not None:
        up = up + _broadcast_expert_bias(up_proj_bias, num_tokens_per_expert, up.shape[0]).bfloat16()

    gate = None
    if gate_proj is not None:
        gate = grouped_mm_fn(x_bf16, gate_proj.bfloat16(), offs=offsets)
        if gate_proj_bias is not None:
            gate = gate + _broadcast_expert_bias(gate_proj_bias, num_tokens_per_expert, gate.shape[0]).bfloat16()

    hidden = activation.apply(gate, up)
    out = grouped_mm_fn(hidden, down_proj.bfloat16(), offs=offsets)

    if down_proj_bias is not None:
        out = out + _broadcast_expert_bias(down_proj_bias, num_tokens_per_expert, out.shape[0]).bfloat16()
    return out.type_as(x)


def _run_experts_grouped_mm_impl(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    fp8: bool = False,
) -> torch.Tensor:
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    # grouped mm between a 2D tensor and a 3D tensor
    assert x.dim() == 2

    if fp8:
        from prime_rl.trainer.models.layers.fp8_grouped_gemm import grouped_fp8_gemm

        h = F.silu(grouped_fp8_gemm(x.bfloat16(), w1.bfloat16().transpose(-2, -1), offsets))
        h = h * grouped_fp8_gemm(x.bfloat16(), w3.bfloat16().transpose(-2, -1), offsets)
        out = grouped_fp8_gemm(h, w2.bfloat16().transpose(-2, -1), offsets).type_as(x)
    else:
        h = F.silu(torch._grouped_mm(x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets))
        h = h * torch._grouped_mm(x.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets)
        out = torch._grouped_mm(h, w2.bfloat16().transpose(-2, -1), offs=offsets).type_as(x)

    return out


_fused_moe = None
_GROUPED_MM_ALIGN_M = 16  # torch._grouped_mm, used by the hand-written backward, needs 16-aligned group extents


def _load_fused_moe_kernel() -> ModuleType:
    # Load prime_kernels.flash_moe if supported on the current running machine.
    global _fused_moe
    if _fused_moe is None:
        if importlib.util.find_spec("prime_kernels") is None:
            raise ModuleNotFoundError(
                "The Prime-Flash-MoE kernel lives in the prime-kernels wheel, which is not available. Install it with `uv sync --extra kernels`."
            )
        import prime_kernels

        _fused_moe = prime_kernels.load("flash_moe")
    return _fused_moe


def _run_experts_fused_kernel(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    top_scores: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    kernel = _load_fused_moe_kernel()  # Try to load the kernel from prime-kernels
    gate_up_weight = torch.cat((w1.to(torch.bfloat16), w3.to(torch.bfloat16)), dim=1)
    sorted_token_ids, expert_ids, num_tokens_post_padded = (
        kernel.moe_align(  # Invoke align kernel first to prepare ids and other inputs
            selected_experts_indices.to(torch.int32).contiguous(), num_experts, kernel.BLOCK_M
        )
    )
    out = torch.empty_like(x, dtype=torch.bfloat16)  # Empty, as split=True zeros out the result anyway
    kernel.fused_moe_bf16(  # Invoke kernel
        x.to(torch.bfloat16),
        gate_up_weight,
        w2.to(torch.bfloat16).contiguous(),
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_scores.to(torch.float32),
        out,
        selected_experts_indices.shape[1],
        block_m=kernel.BLOCK_M,
        split=True,
    )
    return out.type_as(x)


def _quantize_mxfp8(t: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    from torchao.prototype.moe_training.tensor import TrainingWeightWrapperBaseTensor, unwrap_weight
    from torchao.prototype.mx_formats import ScaleCalculationMode
    from torchao.prototype.mx_formats.mx_tensor import to_mx

    if isinstance(t, TrainingWeightWrapperBaseTensor):
        t = unwrap_weight(t)
    scales, data = to_mx(t.to(torch.bfloat16), torch.float8_e4m3fn, block_size, scaling_mode=ScaleCalculationMode.RCEIL)
    return data, scales


def _run_experts_fused_mxfp8_kernel(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    top_scores: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    kernel = _load_fused_moe_kernel()  # Try to load the kernel from prime-kernels
    block_size = kernel.MXFP8_SCALE_BLOCK
    gate_up_weight, gate_up_scales = _quantize_mxfp8(torch.cat((w1, w3), dim=1), block_size)
    down_weight, down_scales = _quantize_mxfp8(w2, block_size)
    x_data, x_scales = _quantize_mxfp8(x, block_size)  # Activation scales stay row major, only the weights are packed
    sorted_token_ids, expert_ids, num_tokens_post_padded = (
        kernel.moe_align(  # Invoke align kernel first to prepare ids and other inputs
            selected_experts_indices.to(torch.int32).contiguous(), num_experts, kernel.BLOCK_M
        )
    )
    out = torch.empty_like(x, dtype=torch.bfloat16)  # Empty, as split=True zeros out the result anyway
    kernel.fused_moe_mxfp8(  # Invoke kernel
        x_data,
        x_scales,
        gate_up_weight,
        kernel.pack_scales_blocked(gate_up_scales),
        down_weight,
        kernel.pack_scales_blocked(down_scales),
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_scores.to(torch.float32),
        out,
        selected_experts_indices.shape[1],
        block_m=kernel.BLOCK_M,
        split=True,
    )
    return out.type_as(x)


def _aligned_group_layout(
    counts: torch.Tensor, num_routed: int, num_experts: int, align_m: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    aligned_counts = torch.clamp((counts + align_m - 1) // align_m, min=1) * align_m
    shift = (torch.cumsum(aligned_counts, 0) - aligned_counts) - (torch.cumsum(counts, 0) - counts)
    dst = torch.arange(num_routed, device=counts.device) + shift.repeat_interleave(counts)
    buf_rows = -(-(num_routed + align_m * num_experts) // align_m) * align_m
    aligned_counts[-1] += buf_rows - aligned_counts.sum()
    return aligned_counts, dst, buf_rows


def _run_experts_fused_reference(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    top_scores: torch.Tensor,
    num_experts: int,
    align_m: int | None = None,
) -> torch.Tensor:
    """Differentiable equivalent of the fused kernel - this implements the backward pass as the fused kernel itself is forward only."""
    top_k = selected_experts_indices.shape[1]
    flat_experts = selected_experts_indices.reshape(-1)
    num_tokens_per_expert = torch.histc(flat_experts.float(), bins=num_experts, min=0, max=num_experts)
    order = torch.argsort(flat_experts, stable=True)
    top_scores_sorted = top_scores.reshape(-1)[order]
    token_idx = order // top_k
    if align_m is None:
        routed_output = _run_experts_grouped_mm_impl(w1, w2, w3, x[token_idx], num_tokens_per_expert)
    else:
        counts = num_tokens_per_expert.to(torch.int64)
        aligned_counts, dst, buf_rows = _aligned_group_layout(counts, order.numel(), num_experts, align_m)
        x_pad = x.new_zeros(buf_rows, x.shape[1])
        x_pad[dst] = x[token_idx]
        routed_output = _run_experts_grouped_mm_impl(w1, w2, w3, x_pad, aligned_counts)[dst]
    routed_output = (routed_output.float() * top_scores_sorted.reshape(-1, 1)).to(x.dtype)
    return torch.zeros_like(x).index_add(0, token_idx, routed_output)


def _run_experts_fused_backward_bf16(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    top_scores: torch.Tensor,
    num_experts: int,
    needs_input_grad: tuple[bool, ...],
) -> tuple[torch.Tensor | None, ...]:
    """Closed-form gradients for the bf16 fused MoE forward."""
    needs_x, needs_w1, needs_w2, needs_w3, needs_scores = (needs_input_grad[p] for p in (0, 1, 2, 3, 5))
    top_k = selected_experts_indices.shape[1]
    flat_experts = selected_experts_indices.reshape(-1)
    num_tokens_per_expert = torch.histc(flat_experts.float(), bins=num_experts, min=0, max=num_experts)
    order = torch.argsort(flat_experts, stable=True)
    token_idx = order // top_k
    counts = num_tokens_per_expert.to(torch.int64)
    aligned_counts, dst, buf_rows = _aligned_group_layout(counts, order.numel(), num_experts, _GROUPED_MM_ALIGN_M)
    offs = torch.cumsum(aligned_counts, dim=0, dtype=torch.int32)

    w1b, w2b, w3b = w1.bfloat16(), w2.bfloat16(), w3.bfloat16()
    x_sorted = x.new_zeros(buf_rows, x.shape[1], dtype=torch.bfloat16)
    x_sorted[dst] = x[token_idx].bfloat16()
    g_sorted = grad_out.new_zeros(buf_rows, grad_out.shape[1], dtype=torch.bfloat16)
    g_sorted[dst] = grad_out[token_idx].bfloat16()
    scores_sorted = top_scores.new_zeros(buf_rows, 1)
    scores_sorted[dst] = top_scores.reshape(-1)[order].reshape(-1, 1)

    a = torch._grouped_mm(x_sorted, w1b.transpose(-2, -1), offs=offs)
    b = torch._grouped_mm(x_sorted, w3b.transpose(-2, -1), offs=offs)
    silu_a = F.silu(a)
    h = silu_a * b

    g2 = torch._grouped_mm(g_sorted, w2b, offs=offs)
    grad_scores = None
    if needs_scores:
        grad_s = (h.float() * g2.float()).sum(dim=-1)[dst]
        grad_scores = (
            torch.zeros_like(top_scores.reshape(-1))
            .index_copy(0, order, grad_s.to(top_scores.dtype))
            .reshape_as(top_scores)
        )
    grad_h = (g2.float() * scores_sorted).to(torch.bfloat16)
    grad_a = torch.ops.aten.silu_backward(grad_h * b, a)
    grad_b = grad_h * silu_a

    grad_x = None
    if needs_x:
        grad_rows = torch._grouped_mm(grad_a, w1b, offs=offs) + torch._grouped_mm(grad_b, w3b, offs=offs)
        grad_x = torch.zeros_like(x).index_add(0, token_idx, grad_rows[dst].type_as(x))
    grad_w1 = torch._grouped_mm(grad_a.t(), x_sorted, offs=offs).type_as(w1) if needs_w1 else None
    grad_w3 = torch._grouped_mm(grad_b.t(), x_sorted, offs=offs).type_as(w3) if needs_w3 else None
    grad_w2 = None
    if needs_w2:
        grad_y = (g_sorted.float() * scores_sorted).to(torch.bfloat16)
        grad_w2 = torch._grouped_mm(grad_y.t(), h, offs=offs).type_as(w2)
    return grad_x, grad_w1, grad_w2, grad_w3, grad_scores


class _FusedMoE(torch.autograd.Function):
    """Forward runs the fused MoE kernel and as the fused kernel is forward only, backward is done by hand."""

    @staticmethod
    def forward(ctx, x, w1, w2, w3, selected_experts_indices, top_scores, num_experts, mxfp8):
        ctx.save_for_backward(x, w1, w2, w3, selected_experts_indices, top_scores)
        ctx.num_experts = num_experts
        ctx.mxfp8 = mxfp8
        run_kernel = _run_experts_fused_mxfp8_kernel if mxfp8 else _run_experts_fused_kernel
        return run_kernel(x, w1, w2, w3, selected_experts_indices, top_scores, num_experts)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        x, w1, w2, w3, selected_experts_indices, top_scores = ctx.saved_tensors
        if not ctx.mxfp8:
            grad_x, grad_w1, grad_w2, grad_w3, grad_scores = _run_experts_fused_backward_bf16(
                grad_out, x, w1, w2, w3, selected_experts_indices, top_scores, ctx.num_experts, ctx.needs_input_grad
            )
            return grad_x, grad_w1, grad_w2, grad_w3, None, grad_scores, None, None
        grad_poses = (0, 1, 2, 3, 5)
        with torch.enable_grad():
            leaves = [
                t.detach().requires_grad_(ctx.needs_input_grad[p])
                for t, p in zip((x, w1, w2, w3, top_scores), grad_poses)
            ]
            out = _run_experts_fused_reference(
                leaves[0],
                leaves[1],
                leaves[2],
                leaves[3],
                selected_experts_indices,
                leaves[4],
                ctx.num_experts,
                align_m=_load_fused_moe_kernel().MXFP8_SCALE_BLOCK,
            )
        wanted = [leaf for leaf in leaves if leaf.requires_grad]
        computed = iter(torch.autograd.grad(out, wanted, grad_out) if wanted else ())
        grads = {p: (next(computed) if leaf.requires_grad else None) for leaf, p in zip(leaves, grad_poses)}
        return grads[0], grads[1], grads[2], grads[3], None, grads[5], None, None


class GroupedExperts(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        *,
        expert_type: ExpertType = "gated",
        activation: ActivationType = "silu",
        bias: bool = False,
        grouped_mm_fn: Callable[..., torch.Tensor] = torch._grouped_mm,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.gate_proj = nn.Parameter(torch.empty(num_experts, hidden_dim, dim)) if expert_type == "gated" else None
        self.up_proj = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.down_proj = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.gate_proj_bias = (
            nn.Parameter(torch.empty(num_experts, hidden_dim)) if bias and self.gate_proj is not None else None
        )
        self.up_proj_bias = nn.Parameter(torch.empty(num_experts, hidden_dim)) if bias else None
        self.down_proj_bias = nn.Parameter(torch.empty(num_experts, dim)) if bias else None

        self.grouped_mm = grouped_mm_fn
        self.activation = ActivationDispatch[activation]
        self.ep_comm_backend: EPCommBackend = "torch"

    def set_ep_comm_backend(self, backend: EPCommBackend) -> None:
        self.ep_comm_backend = backend

    def _forward_deepep(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        gate_proj = self.gate_proj.to_local().transpose(-2, -1) if self.gate_proj is not None else None
        return _run_grouped_experts_impl(
            self.up_proj.to_local().transpose(-2, -1),
            self.down_proj.to_local().transpose(-2, -1),
            x,
            num_tokens_per_expert,
            gate_proj=gate_proj,
            grouped_mm_fn=self.grouped_mm,
            activation=self.activation,
            gate_proj_bias=self.gate_proj_bias.to_local() if self.gate_proj_bias is not None else None,
            up_proj_bias=self.up_proj_bias.to_local() if self.up_proj_bias is not None else None,
            down_proj_bias=self.down_proj_bias.to_local() if self.down_proj_bias is not None else None,
        )

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        if self.ep_comm_backend == "deepep":
            return self._forward_deepep(x, num_tokens_per_expert)

        gate_proj = self.gate_proj.transpose(-2, -1) if self.gate_proj is not None else None
        return _run_grouped_experts(
            self.up_proj.transpose(-2, -1),
            self.down_proj.transpose(-2, -1),
            x,
            num_tokens_per_expert,
            gate_proj=gate_proj,
            grouped_mm_fn=self.grouped_mm,
            activation=self.activation,
            gate_proj_bias=self.gate_proj_bias,
            up_proj_bias=self.up_proj_bias,
            down_proj_bias=self.down_proj_bias,
        )

    def init_weights(self, init_std: float):
        first_projection = self.gate_proj if self.gate_proj is not None else self.up_proj
        nn.init.trunc_normal_(first_projection, mean=0.0, std=0.02)
        remaining = (self.up_proj, self.down_proj) if self.gate_proj is not None else (self.down_proj,)
        for weight in remaining:
            nn.init.trunc_normal_(weight, mean=0.0, std=init_std)
        for bias in (self.gate_proj_bias, self.up_proj_bias, self.down_proj_bias):
            if bias is not None:
                nn.init.zeros_(bias)


def _selected_probability_mass_sum(
    scores: torch.Tensor, top_scores: torch.Tensor, score_func: Literal["softmax", "sigmoid"]
) -> torch.Tensor:
    with torch.no_grad():
        if score_func == "softmax":
            return top_scores.sum()
        selected_prob_mass = top_scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
        return selected_prob_mass.sum(dim=-1).sum()


class TokenChoiceTopKRouter(nn.Module):
    """Route each token to its top-k experts.

    Args:
        dim (int): Dimension of input tokens.
        num_experts (int): Number of experts in each moe layer.
        top_k (int): Number of experts each token will be routed to in token-choice routing.
        score_func (Literal["softmax", "sigmoid", "topk_softmax"]): Score transform. ``topk_softmax``
            selects experts from the logits and normalizes only the selected logits.
        route_norm (bool): Whether to normalize the routing scores when using sigmoid.
        route_scale (float): Scaling factor applied to the routing scores.
        gate_bias (bool): Whether the gate has a trainable logit bias.
        selection_bias (bool): Whether to keep a persistent selection-only bias. The bias affects
            expert selection but not routing weights.
        topk_sorted (bool): Whether selected experts are returned in descending score order.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        score_func: Literal["softmax", "sigmoid", "topk_softmax"],
        route_norm: bool,
        route_scale: float,
        *,
        gate_bias: bool = False,
        selection_bias: bool = False,
        topk_sorted: bool = True,
    ):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=gate_bias)
        self.register_buffer("selection_bias", torch.zeros(num_experts) if selection_bias else None)
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale
        self.topk_sorted = topk_sorted
        self.force_balanced = False
        # Set via model.moe_router_dtype='float32': the gate weight is kept in fp32
        # (exempt from FSDP bf16 casting) and the gate GEMM runs in fp32.
        self.fp32_gate = False

    def forward(
        self, x: torch.Tensor, expert_bias: torch.Tensor | None = None, routed_experts: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs*slen, dim)``.
            expert_bias (torch.Tensor | None, optional): Optional bias tensor for experts with shape ``(num_experts,)``.
                Used for load balancing. Defaults to None.
            routed_experts (torch.Tensor | None, optional): Optional tensor with shape ``(bs * slen, top_k)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - top_scores (torch.Tensor):
                    Routing scores for selected experts with shape ``(bs*slen, top_k)``.
                - selected_experts_indices (torch.Tensor):
                    Expert indices selected for each token with shape ``(bs*slen, top_k)``.
                - num_tokens_per_expert (torch.Tensor):
                    Number of tokens assigned to each expert with shape ``(num_experts,)``.
                - routing_confidence_sum (torch.Tensor):
                    Sum over tokens of the selected-expert probability mass before route normalization/scaling.
        """
        # scores shape (bs*slen, num_experts)
        assert routed_experts is None or routed_experts.shape[-1] == self.top_k, (
            f"routed_experts shape: {routed_experts.shape}, top_k: {self.top_k}"
        )
        if self.fp32_gate:
            gate_bias = self.gate.bias.float() if self.gate.bias is not None else None
            logits = F.linear(x.float(), self.gate.weight.float(), gate_bias)
        else:
            logits = self.gate(x)

        # By default, sigmoid or softmax is performed in float32 to avoid loss explosion
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(logits.float())
        elif self.score_func == "softmax":
            scores = F.softmax(logits.float(), dim=1)
        elif self.score_func == "topk_softmax":
            scores = logits
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        # top scores shape (bs*slen, top_k)
        # NOTE: selection biases are only used for routing. The gating value
        #       top_scores is still derived from the original scores/logits.

        if routed_experts is not None:
            top_scores = scores.gather(dim=1, index=routed_experts)
            selected_experts_indices = routed_experts
        elif self.force_balanced:
            num_tokens = scores.shape[0]
            arange = torch.arange(num_tokens * self.top_k, device=scores.device)
            selected_experts_indices = (arange % self.num_experts).view(num_tokens, self.top_k)
            top_scores = scores.gather(dim=1, index=selected_experts_indices)
        else:
            selection_scores = scores
            if self.selection_bias is not None:
                selection_scores = selection_scores + self.selection_bias
            if expert_bias is not None:
                selection_scores = selection_scores + expert_bias
            _, selected_experts_indices = torch.topk(
                selection_scores,
                k=self.top_k,
                dim=1,
                sorted=self.topk_sorted,
            )
            top_scores = scores.gather(dim=1, index=selected_experts_indices)

        if self.score_func == "topk_softmax":
            top_scores = F.softmax(top_scores, dim=-1, dtype=top_scores.dtype)
            with torch.no_grad():
                routing_confidence_sum = top_scores.sum()
        else:
            routing_confidence_sum = _selected_probability_mass_sum(scores, top_scores, self.score_func)

        if self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator
        top_scores = top_scores * self.route_scale

        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.reshape(-1).float(),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        ).to(torch.int64)

        return top_scores, selected_experts_indices, num_tokens_per_expert, routing_confidence_sum

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=init_std)
        if self.gate.bias is not None:
            nn.init.zeros_(self.gate.bias)


# NOTE: the reason we make this a stateless module is to support
#       expert_tensor_parallel_degree=1 with consistent TP/EP APIs.
class TokenReorderer(nn.Module):
    """
    This module reorders token indices to match the order of experts, enabling
    efficient parallel processing of tokens by experts.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of experts each token will be routed to.
    """

    def __init__(self, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(
        self,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reorders token indices to match the order of experts for MoE routing.

        Args:
            top_scores (torch.Tensor): Routing scores for selected experts,
                shape (batch_size*seq_len, top_k)
            selected_experts_indices (torch.Tensor): Expert indices selected for each token,
                shape (batch_size*seq_len, top_k)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - top_scores_experts_sorted: Scores reordered to match expert ordering
                - token_indices_experts_sorted: Token indices reordered to match expert ordering
                - num_tokens_per_expert: Number of tokens assigned to each expert
        """
        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        selected_experts_indices = selected_experts_indices.reshape(-1)
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.float(),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        ).to(torch.int64)

        # Reorder the token indices to match the order of the experts
        # token_indices_experts_sorted shape (bs*slen*top_k,)
        token_indices_experts_sorted = torch.argsort(selected_experts_indices, stable=True)

        top_scores_experts_sorted = top_scores.view(-1)[token_indices_experts_sorted]
        token_indices_experts_sorted = token_indices_experts_sorted // self.top_k

        return (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
        )


class MoE(nn.Module):
    """Token-choice MoE runtime composed from a router, grouped experts, and optional projections."""

    @classmethod
    def from_args(
        cls,
        args: MoEArgs,
        dim: int,
        hidden_dim: int,
        *,
        shared_expert_hidden_dim: int | None = None,
        gated_shared_expert: bool = False,
    ) -> "MoE":
        grouped_mm_fn = torch._grouped_mm
        if args.fp8:
            from prime_rl.trainer.models.layers.fp8_grouped_gemm import grouped_fp8_gemm

            grouped_mm_fn = grouped_fp8_gemm

        experts = GroupedExperts(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=args.num_experts,
            expert_type=args.expert_type,
            activation=args.activation,
            grouped_mm_fn=grouped_mm_fn,
        )
        router = TokenChoiceTopKRouter(
            dim=dim,
            num_experts=args.num_experts,
            top_k=args.top_k,
            score_func=args.score_func,
            route_norm=args.route_norm,
            route_scale=args.route_scale,
        )
        if args.num_shared_experts > 0:
            shared_hidden_dim = (
                hidden_dim * args.num_shared_experts if shared_expert_hidden_dim is None else shared_expert_hidden_dim
            )
            shared_expert = FeedForward(
                dim=dim,
                hidden_dim=shared_hidden_dim,
                expert_type=args.expert_type,
                activation=args.activation,
            )
            shared_expert_gate = nn.Linear(dim, 1, bias=False) if gated_shared_expert else None
        else:
            if shared_expert_hidden_dim is not None or gated_shared_expert:
                raise ValueError("Shared-expert options require num_shared_experts > 0")
            shared_expert = None
            shared_expert_gate = None

        return cls(
            router=router,
            experts=experts,
            shared_expert=shared_expert,
            shared_expert_gate=shared_expert_gate,
            score_before_experts=args.score_before_experts,
            load_balance_coeff=args.load_balance_coeff,
        )

    def __init__(
        self,
        *,
        router: TokenChoiceTopKRouter,
        experts: GroupedExperts,
        shared_expert: FeedForward | None,
        score_before_experts: bool,
        load_balance_coeff: float | None,
        shared_expert_gate: nn.Linear | None = None,
    ) -> None:
        super().__init__()
        if shared_expert_gate is not None and shared_expert is None:
            raise ValueError("shared_expert_gate requires shared_expert")
        self.router = router
        self.experts = experts
        self.shared_expert = shared_expert
        self.shared_expert_gate = shared_expert_gate
        self.score_before_experts = score_before_experts

        self.ep_comm_backend: EPCommBackend = "torch"
        self.experts.set_ep_comm_backend(self.ep_comm_backend)
        self.reorderer = TokenReorderer(num_experts=experts.num_experts, top_k=router.top_k)
        self.deepep_token_chunk_size: int | None = None
        # Set by model.moe_fused_kernel=true: which fused kernel the routed experts run through.
        # None keeps the ordinary reorderer + grouped-mm path.
        self.fused_kernel: Literal["bf16", "mxfp8"] | None = None

        # define fields for auxiliary-loss-free load balancing (https://arxiv.org/abs/2408.15664)
        # NOTE: tokens_per_expert is accumulated in the model forward pass.
        #       expert_bias is updated outside the model in an optimizer step pre hook
        #       to work with gradient accumulation.
        self.load_balance_coeff = load_balance_coeff
        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff > 0.0
            self.register_buffer(
                "expert_bias",
                torch.zeros(experts.num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias = None
        # tokens_per_expert will be used to track expert usage and to update the expert bias for load balancing
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(experts.num_experts, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer("routing_confidence_sum", torch.tensor(0.0, dtype=torch.float32), persistent=False)

    def set_ep_comm_backend(self, backend: EPCommBackend) -> None:
        self.ep_comm_backend = backend
        self.experts.set_ep_comm_backend(backend)

    def set_deepep_token_chunk_size(self, chunk_size: int | None) -> None:
        self.deepep_token_chunk_size = chunk_size

    def _run_local_routed_experts(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        return self.experts(x, num_tokens_per_expert)

    def _prepare_expert_input(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _prepare_expert_output(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _run_shared_expert(self, x: torch.Tensor) -> torch.Tensor | None:
        if self.shared_expert is None:
            return None
        output = self.shared_expert(x)
        if self.shared_expert_gate is not None:
            output = torch.sigmoid(self.shared_expert_gate(x)) * output
        return output

    def _run_routed_experts(
        self,
        x: torch.Tensor,
        token_indices_experts_sorted: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        top_scores_experts_sorted: torch.Tensor,
    ) -> torch.Tensor:
        dim = x.shape[-1]
        routed_indices = token_indices_experts_sorted.reshape(-1, 1).expand(-1, dim)
        routed_input = torch.gather(x, dim=0, index=routed_indices)

        routed_input = self._prepare_expert_input(routed_input)

        if self.score_before_experts:
            routed_input = (routed_input.to(torch.float32) * top_scores_experts_sorted.reshape(-1, 1)).to(x.dtype)

        routed_output = self.experts(routed_input, num_tokens_per_expert)

        if not self.score_before_experts:
            routed_output = (routed_output.to(torch.float32) * top_scores_experts_sorted.reshape(-1, 1)).to(x.dtype)

        routed_output = self._prepare_expert_output(routed_output)

        return routed_output

    def _run_fused_routed_experts(
        self,
        x: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        top_scores: torch.Tensor,
    ) -> torch.Tensor:
        to_local = lambda tensor: tensor.to_local() if isinstance(tensor, DTensor) else tensor
        return _FusedMoE.apply(
            x,
            to_local(self.experts.w1),
            to_local(self.experts.w2),
            to_local(self.experts.w3),
            selected_experts_indices,
            top_scores,
            self.experts.num_experts,
            self.fused_kernel == "mxfp8",
        )

    def _run_deepep_routed_experts(
        self,
        x: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        top_scores: torch.Tensor,
    ) -> torch.Tensor:
        from prime_rl.trainer.distributed.deepep import (
            combine_tokens,
            dispatch_tokens_async,
            finalize_dispatch_tokens,
            sync_combine,
        )
        from prime_rl.trainer.distributed.expert_parallel import get_ep_group

        if x.shape[0] == 0:
            shared_output = self._run_shared_expert(x)
            return x.new_zeros(x.shape) if shared_output is None else shared_output

        group = get_ep_group(self.experts)
        routed_input = self._prepare_expert_input(x)
        chunk_size = min(self.deepep_token_chunk_size or routed_input.shape[0], routed_input.shape[0])

        def dispatch_chunk(start: int, end: int):
            return dispatch_tokens_async(
                routed_input[start:end],
                selected_experts_indices[start:end],
                top_scores[start:end],
                num_experts=self.experts.num_experts,
                group=group,
                score_before_experts=self.score_before_experts,
            )

        def run_pending_chunk(pending_state):
            hidden_states, num_tokens_per_expert, dispatch_state = finalize_dispatch_tokens(pending_state)
            routed_output = self._run_local_routed_experts(hidden_states, num_tokens_per_expert)
            # Keep combine outside the checkpointed routed-expert region so
            # selective AC only recomputes local expert matmuls.
            return combine_tokens(routed_output, dispatch_state)

        pending_state = dispatch_chunk(0, chunk_size)
        routed_outputs: list[torch.Tensor] = []

        for chunk_start in range(chunk_size, routed_input.shape[0], chunk_size):
            chunk_end = min(chunk_start + chunk_size, routed_input.shape[0])
            next_pending_state = dispatch_chunk(chunk_start, chunk_end)
            routed_outputs.append(run_pending_chunk(pending_state))
            pending_state = next_pending_state

        routed_outputs.append(run_pending_chunk(pending_state))

        shared_output = self._run_shared_expert(x)
        sync_combine()
        routed_output = routed_outputs[0] if len(routed_outputs) == 1 else torch.cat(routed_outputs, dim=0)
        routed_output = self._prepare_expert_output(routed_output)
        return routed_output if shared_output is None else shared_output + routed_output

    def forward(
        self,
        x: torch.Tensor,
        routed_experts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs, slen, dim)``.
            routed_experts (torch.Tensor | None, optional): Optional tensor with shape ``(bs, slen, top_k)``.

        Returns:
            out (torch.Tensor): Output tensor with shape ``(bs, slen, dim)``.
        """
        bs, slen, dim = x.shape
        x = x.view(-1, dim)

        if routed_experts is not None:
            _, _, top_k = routed_experts.shape
            routed_experts = routed_experts.reshape(
                -1, top_k
            )  # we have to reshape here because the original is non-contiguous

        # top_scores and selected_experts_indices shape (bs*slen*top_k,)
        # num_tokens_per_expert shape (num_experts,)
        (
            top_scores,
            selected_experts_indices,
            num_tokens_per_expert,
            routing_confidence_sum,
        ) = self.router(x, self.expert_bias, routed_experts=routed_experts)

        # tokens_per_expert will be used to update the expert bias for load balancing.
        # and also to count the expert usage
        # Full block checkpointing can double count tokens_per_expert because it reruns the router
        # in backward. The selective MoE path avoids that by checkpointing only the
        # routed expert compute below.
        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)
            self.routing_confidence_sum.add_(routing_confidence_sum)

        if self.ep_comm_backend == "deepep":
            routed_output = self._run_deepep_routed_experts(x, selected_experts_indices, top_scores)
            return routed_output.reshape(bs, slen, dim)

        if self.fused_kernel:
            routed_output = self._run_fused_routed_experts(x, selected_experts_indices, top_scores)
            if self.shared_expert is not None:
                routed_output = routed_output + self.shared_expert(x)
            return routed_output.reshape(bs, slen, dim)

        # top_scores and token_indices_experts_sorted shape (bs*slen*top_k,)
        # num_tokens_per_expert shape (num_experts,)
        # NOTE: the reason we need to compute num_tokens_per_expert again is:
        #       1st computation in router is to update self.tokens_per_expert
        #       which would be the same across all TP ranks.
        #       2nd computation in reorderer is for the actual routing and experts computation
        #       which would be sharded over TP ranks if expert_tensor_parallel_degree==1.
        #       If tensor_paralllel_degree == expert_tensor_parallel_degree, they agree.
        (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
        ) = self.reorderer(top_scores, selected_experts_indices)

        routed_output = self._run_routed_experts(
            x,
            token_indices_experts_sorted,
            num_tokens_per_expert,
            top_scores_experts_sorted,
        )
        out = self._run_shared_expert(x)
        if out is None:
            out = torch.zeros_like(x)

        routed_indices = token_indices_experts_sorted.reshape(-1, 1).expand(-1, dim)
        out = out.scatter_add(dim=0, index=routed_indices, src=routed_output)
        out = out.reshape(bs, slen, dim)
        return out

    def init_weights(
        self,
        init_std: float,
        buffer_device: torch.device,
    ):
        self.experts.init_weights(init_std)
        self.router.init_weights(init_std)
        if self.shared_expert is not None:
            self.shared_expert.init_weights(init_std)
        if self.shared_expert_gate is not None:
            nn.init.trunc_normal_(self.shared_expert_gate.weight, mean=0.0, std=init_std)

        with torch.device(buffer_device):
            self.tokens_per_expert = torch.zeros(self.experts.num_experts, dtype=torch.float32)
            self.routing_confidence_sum = torch.tensor(0.0, dtype=torch.float32)
            if self.load_balance_coeff is not None:
                self.expert_bias = torch.zeros(self.experts.num_experts, dtype=torch.float32)
