from __future__ import annotations

import torch
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DTensor

from prime_rl.trainer.distributed.token_dispatcher import ExpertFunction, TokenDispatcher
from prime_rl.trainer.models.layers.activations import Silu


def _to_local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


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
    ) -> torch.Tensor:
        from prime_rl.trainer.models.layers.mega_moe import mega_moe_forward, prepare_mega_moe_weights

        x_bf16 = x.to(torch.bfloat16).contiguous()
        topk_idx = selected_experts_indices.to(torch.int64)
        topk_weights = top_scores.to(torch.float32)
        weights = prepare_mega_moe_weights(gate_up_proj, down_proj)
        y = mega_moe_forward(x_bf16, topk_idx, topk_weights, weights, buffer)
        ctx.save_for_backward(x_bf16, topk_idx, topk_weights, gate_up_proj, down_proj)
        ctx.buffer = buffer
        ctx.x_dtype, ctx.scores_dtype, ctx.scores_shape = x.dtype, top_scores.dtype, top_scores.shape
        return y.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        from prime_rl.trainer.models.layers.mega_moe import mega_moe_backward, prepare_mega_moe_weights

        x_bf16, topk_idx, topk_weights, gate_up_proj, down_proj = ctx.saved_tensors
        weights = prepare_mega_moe_weights(gate_up_proj, down_proj)
        dx, dl1, dl2, dtopk = mega_moe_backward(
            grad_y.to(torch.bfloat16).contiguous(), x_bf16, topk_idx, topk_weights, weights, ctx.buffer
        )
        return (
            dx.to(ctx.x_dtype),
            dtopk.reshape(ctx.scores_shape).to(ctx.scores_dtype),
            None,
            dl1.to(gate_up_proj.dtype),
            dl2.to(down_proj.dtype),
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
    ) -> None:
        from prime_rl.trainer.models.layers.mega_moe import (
            build_mega_moe_buffer,
            check_mega_moe_dims,
            mega_moe_available,
        )

        if not mega_moe_available():
            raise RuntimeError(
                "Mega MoE dispatch requires DeepGEMM's Mega MoE kernels (SM100+/Blackwell and a "
                "deep_gemm build with `bf16_mega_moe` and `bf16_mega_moe_backward`)."
            )
        check_mega_moe_dims(hidden, intermediate_hidden)

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
        if experts.activation is not Silu:
            raise ValueError("Mega MoE dispatch requires the `silu` (SwiGLU) expert activation.")
        if any(bias is not None for bias in (experts.gate_proj_bias, experts.up_proj_bias, experts.down_proj_bias)):
            raise ValueError("Mega MoE dispatch does not support expert biases.")
        num_tokens = x.shape[0]
        if num_tokens > self.max_tokens_per_rank:
            raise RuntimeError(
                f"Mega MoE buffer is sized for {self.max_tokens_per_rank} tokens/rank, got {num_tokens}. "
                "Raise `model.moe.dispatch.max_tokens_per_rank`."
            )

        if experts.gate_up_proj is not None:
            gate_up_proj = _to_local(experts.gate_up_proj)
        elif experts.gate_proj is not None:
            gate_up_proj = torch.cat([_to_local(experts.gate_proj), _to_local(experts.up_proj)], dim=1)
        else:
            raise ValueError("Mega MoE dispatch requires gated experts (SwiGLU gate+up), got non-gated experts.")
        return _MegaMoeRoutedExperts.apply(
            x, top_scores, selected_experts_indices, gate_up_proj, _to_local(experts.down_proj), self.buffer
        )

    def synchronize(self) -> None:
        return None
