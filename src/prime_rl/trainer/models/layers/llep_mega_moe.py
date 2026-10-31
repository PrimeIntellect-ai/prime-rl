"""Least-loaded expert parallelism (LLEP, https://arxiv.org/abs/2601.17111) on the Mega MoE kernels.

When a rank's experts receive more routes than the others', part of those routes is computed by less loaded ranks
on copies of the experts' weights; every token keeps its expert choice. The plan, the weight copies and the expert
compute come from prime-mega-moe (`deep_gemm.llep`); this module wraps them in custom ops: planning is its own op, so
a compiled backward under activation checkpointing recomputes only the plan.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.distributed import ProcessGroup

from prime_rl.configs.trainer import LLEPConfig
from prime_rl.trainer.models.layers.expert_compute import to_local
from prime_rl.trainer.models.layers.mega_moe import _activation_clamp, check_mega_moe_dims, mega_moe_available

if TYPE_CHECKING:
    from deep_gemm.llep import LLEPMegaMoE

    from prime_rl.trainer.models.layers.moe import GroupedExperts

_STATES: dict[int, "LLEPMegaMoE"] = {}


@torch.library.custom_op("prime_rl::llep_plan", mutates_args=())
def llep_plan_op(topk_idx: torch.Tensor, state_key: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The virtual expert of each route, the foreign expert table and a (host) id of the plan."""
    return _STATES[state_key].plan(topk_idx)


@llep_plan_op.register_fake
def _llep_plan_fake(topk_idx, state_key):
    state = _STATES[state_key]
    return (
        torch.empty_like(topk_idx, dtype=torch.int64),
        topk_idx.new_empty((state.group.size(), state.options.max_foreign), dtype=torch.int32),
        torch.empty(1, dtype=torch.int64),
    )


@torch.library.custom_op("prime_rl::llep_mega_moe_forward", mutates_args=())
def llep_mega_moe_forward_op(
    x: torch.Tensor,
    vids: torch.Tensor,
    foreign: torch.Tensor,
    plan_id: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    state_key: int,
    activation_clamp: float | None = None,
) -> torch.Tensor:
    return _STATES[state_key].forward(
        x, vids, foreign, plan_id, topk_weights, gate_up_proj, down_proj, activation_clamp
    )


@llep_mega_moe_forward_op.register_fake
def _llep_mega_moe_forward_fake(
    x, vids, foreign, plan_id, topk_weights, gate_up_proj, down_proj, state_key, activation_clamp=None
):
    return torch.empty_like(x)


@torch.library.custom_op("prime_rl::llep_mega_moe_backward", mutates_args=())
def llep_mega_moe_backward_op(
    dy: torch.Tensor,
    x: torch.Tensor,
    vids: torch.Tensor,
    foreign: torch.Tensor,
    plan_id: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    state_key: int,
    activation_clamp: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _STATES[state_key].backward(
        dy, x, vids, foreign, plan_id, topk_weights, gate_up_proj, down_proj, activation_clamp
    )


@llep_mega_moe_backward_op.register_fake
def _llep_mega_moe_backward_fake(
    dy, x, vids, foreign, plan_id, topk_weights, gate_up_proj, down_proj, state_key, activation_clamp=None
):
    return (
        torch.empty_like(x),
        torch.empty_like(topk_weights, dtype=torch.float32),
        torch.empty_like(gate_up_proj),
        torch.empty_like(down_proj),
    )


def _setup_context(ctx, inputs, output) -> None:
    x, vids, foreign, plan_id, topk_weights, gate_up_proj, down_proj, state_key, activation_clamp = inputs
    ctx.save_for_backward(x, vids, foreign, plan_id, topk_weights, gate_up_proj, down_proj)
    ctx.state_key = state_key
    ctx.activation_clamp = activation_clamp


def _backward(ctx, grad_y):
    x, vids, foreign, plan_id, topk_weights, gate_up_proj, down_proj = ctx.saved_tensors
    dx, dtopk, dw1, dw2 = torch.ops.prime_rl.llep_mega_moe_backward(
        grad_y, x, vids, foreign, plan_id, topk_weights, gate_up_proj, down_proj, ctx.state_key, ctx.activation_clamp
    )
    return dx, None, None, None, dtopk.to(topk_weights.dtype), dw1, dw2, None, None


llep_mega_moe_forward_op.register_autograd(_backward, setup_context=_setup_context)


class LLEPMegaMoEExpertCompute:
    """Fused-dispatch expert compute (pair it with `FusedTokenDispatcher`) running LLEP on the Mega MoE kernels."""

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden: int,
        intermediate_hidden: int,
        group: ProcessGroup,
        max_tokens_per_rank: int,
        num_sms: int,
        options: LLEPConfig,
    ) -> None:
        if not mega_moe_available("bf16") or torch.cuda.get_device_capability()[0] != 9:
            raise RuntimeError(
                "LLEP Mega MoE needs an SM90 GPU and the prime-mega-moe `deep_gemm` build with LLEP support."
            )
        try:
            from deep_gemm.llep import LLEPMegaMoE, LLEPOptions
        except ImportError as error:
            raise RuntimeError(
                "LLEP needs a prime-mega-moe `deep_gemm` build with LLEP support (`deep_gemm.llep`)."
            ) from error

        check_mega_moe_dims(hidden, intermediate_hidden)
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.max_tokens_per_rank = max_tokens_per_rank
        state = LLEPMegaMoE(
            group,
            num_experts,
            top_k,
            hidden,
            intermediate_hidden,
            max_tokens_per_rank,
            num_sms,
            LLEPOptions(**options.model_dump()),
        )
        self.state_key = id(state)
        _STATES[self.state_key] = state

    def validate(self, experts: "GroupedExperts") -> None:
        if experts.gate_proj is None and experts.gate_up_proj is None:
            raise ValueError("LLEP Mega MoE requires gated experts (SwiGLU gate+up).")
        if any(bias is not None for bias in (experts.gate_proj_bias, experts.up_proj_bias, experts.down_proj_bias)):
            raise ValueError("LLEP Mega MoE does not support expert biases.")
        _activation_clamp(experts.activation)
        shape = (experts.num_experts, experts.down_proj.shape[1], experts.hidden_dim)
        expected = (self.num_experts, self.hidden, self.intermediate_hidden)
        if shape != expected:
            raise ValueError(f"LLEP Mega MoE is sized for (num_experts, hidden, intermediate)={expected}; got {shape}.")

    def validate_top_k(self, top_k: int) -> None:
        if top_k != self.top_k:
            raise ValueError(
                f"LLEP Mega MoE's buffers are sized for top_k={self.top_k}; got a router with top_k={top_k}."
            )

    def destroy(self) -> None:
        _STATES.pop(self.state_key).destroy()

    def __call__(
        self,
        experts: "GroupedExperts",
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> torch.Tensor:
        if x.shape[0] > self.max_tokens_per_rank:
            raise RuntimeError(f"LLEP Mega MoE buffer holds {self.max_tokens_per_rank} tokens/rank, got {x.shape[0]}.")
        if experts.gate_up_proj is not None:
            gate_up_proj = to_local(experts.gate_up_proj)
        else:
            gate_up_proj = torch.cat([to_local(experts.gate_proj), to_local(experts.up_proj)], dim=1)
        vids, foreign, plan_id = torch.ops.prime_rl.llep_plan(selected_experts_indices.to(torch.int64), self.state_key)
        return torch.ops.prime_rl.llep_mega_moe_forward(
            x,
            vids,
            foreign,
            plan_id,
            top_scores.to(torch.float32),
            gate_up_proj,
            to_local(experts.down_proj),
            self.state_key,
            _activation_clamp(experts.activation),
        )
