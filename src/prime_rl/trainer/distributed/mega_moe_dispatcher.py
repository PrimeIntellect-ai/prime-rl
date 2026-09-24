from __future__ import annotations

import torch
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DTensor

from prime_rl.trainer.distributed.token_dispatcher import ExpertFunction, TokenDispatcher
from prime_rl.trainer.models.layers.activations import Silu


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
    raise ValueError("Mega MoE dispatch requires a SwiGLU (`silu` or DeepSeek V4 clamped) expert activation.")


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
        from prime_rl.trainer.models.layers.mega_moe import (
            MegaMoeExpertWeights,
            prepare_mega_moe_weights,
            register_mega_moe_buffer,
        )

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
        from prime_rl.trainer.models.layers.mega_moe import MegaMoeExpertWeights, mega_moe_backward

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


class MegaMoeTokenDispatcher(TokenDispatcher):
    """Fused Mega MoE dispatch + expert compute + combine, forward and backward."""

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden: int,
        intermediate_hidden: int,
        group: ProcessGroup,
        max_tokens_per_rank: int,
        num_reserved_sms: int = 16,
    ) -> None:
        from prime_rl.trainer.models.layers.mega_moe import (
            build_mega_moe_buffer,
            check_mega_moe_dims,
            mega_moe_available,
            reserve_sms_for_comm,
        )

        if not mega_moe_available():
            raise RuntimeError(
                "Mega MoE dispatch requires DeepGEMM's Mega MoE kernels (SM100+/Blackwell and a "
                "deep_gemm build with `bf16_mega_moe` and `bf16_mega_moe_backward`)."
            )
        check_mega_moe_dims(hidden, intermediate_hidden)
        reserve_sms_for_comm(num_reserved_sms)

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.group = group
        self.max_tokens_per_rank = max_tokens_per_rank
        self.buffer = build_mega_moe_buffer(group, num_experts, max_tokens_per_rank, top_k, hidden, intermediate_hidden)

    def run(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        experts: ExpertFunction,
        *,
        score_before_experts: bool,
    ) -> torch.Tensor:
        if score_before_experts:
            raise ValueError(
                "Mega MoE dispatch requires score_before_experts=False (it applies router weights "
                "at combine time, after both expert GEMMs)."
            )
        activation_clamp = _activation_clamp(experts.activation)
        if any(bias is not None for bias in (experts.gate_proj_bias, experts.up_proj_bias, experts.down_proj_bias)):
            raise ValueError("Mega MoE dispatch does not support expert biases.")
        num_tokens = x.shape[0]
        if num_tokens > self.max_tokens_per_rank:
            raise RuntimeError(
                f"Mega MoE buffer is sized for {self.max_tokens_per_rank} tokens/rank, got {num_tokens}. "
                "Raise `model.moe.dispatch.max_tokens_per_rank`."
            )

        if experts.gate_up_proj is None and experts.gate_proj is None:
            raise ValueError("Mega MoE dispatch requires gated experts (SwiGLU gate+up), got non-gated experts.")

        def fused(module, x: torch.Tensor) -> torch.Tensor:
            # Runs inside `experts.forward`, i.e. inside FSDP's pre/post-forward hooks: the weights are
            interleaved = False
            if module.gate_up_proj is not None:
                gate_up_proj = _to_local(module.gate_up_proj)
                interleaved = getattr(module, "mega_moe_interleaved", False)
            else:
                gate_up_proj = torch.cat([_to_local(module.gate_proj), _to_local(module.up_proj)], dim=1)
            return _MegaMoeRoutedExperts.apply(
                x,
                top_scores,
                selected_experts_indices,
                gate_up_proj,
                _to_local(module.down_proj),
                self.buffer,
                interleaved,
                activation_clamp,
            )

        return experts(x, None, fused=fused)

    def synchronize(self) -> None:
        return None
