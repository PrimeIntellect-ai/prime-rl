# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor

from prime_rl.trainer.distributed.token_dispatcher import LocalTokenDispatcher, TokenDispatcher
from prime_rl.trainer.models.layers.activations import ActivationDispatch, ActivationType
from prime_rl.trainer.models.layers.grouped_gemm import BF16GroupedGemm, GroupedGemm
from prime_rl.trainer.models.layers.mlp import ExpertType, FeedForward

ScoreFuncType = Literal["softmax", "sigmoid", "topk_softmax"]


@torch.library.custom_op(
    "prime_rl::record_moe_routing_statistics",
    mutates_args=("tokens_per_expert", "routing_confidence_sum"),
)
def record_moe_routing_statistics(
    tokens_per_expert: torch.Tensor,
    routing_confidence_sum: torch.Tensor,
    token_counts: torch.Tensor,
    confidence: torch.Tensor,
) -> None:
    tokens_per_expert.add_(token_counts)
    routing_confidence_sum.add_(confidence)


@record_moe_routing_statistics.register_fake
def _record_moe_routing_statistics_fake(
    tokens_per_expert: torch.Tensor,
    routing_confidence_sum: torch.Tensor,
    token_counts: torch.Tensor,
    confidence: torch.Tensor,
) -> None:
    return None


@dataclass
class MoEArgs:
    num_experts: int = 8

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

    def __post_init__(self) -> None:
        ActivationDispatch[self.activation]


def broadcast_expert_bias(
    bias: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    target_rows: int,
) -> torch.Tensor:
    repeats = num_tokens_per_expert.to(torch.int64)
    padding_rows = repeats.new_tensor(target_rows) - repeats.sum()
    return torch.repeat_interleave(
        torch.cat((bias, bias.new_zeros((1, bias.shape[1])))),
        torch.cat((repeats, padding_rows.unsqueeze(0))),
        dim=0,
        output_size=target_rows,
    )


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
        grouped_gemm: GroupedGemm | None = None,
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

        self.grouped_gemm = grouped_gemm or BF16GroupedGemm()
        self.activation = ActivationDispatch[activation]

    @property
    def token_group_alignment(self) -> int:
        return self.grouped_gemm.token_group_alignment

    def set_grouped_gemm(self, grouped_gemm: GroupedGemm) -> None:
        self.grouped_gemm = grouped_gemm

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        assert x.dim() == 2

        def to_local(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.to_local() if isinstance(tensor, DTensor) else tensor

        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        x_bf16 = x.bfloat16()

        up_proj = to_local(self.up_proj).transpose(-2, -1)
        up = self.grouped_gemm(x_bf16, up_proj.bfloat16(), offs=offsets)
        if self.up_proj_bias is not None:
            up_proj_bias = to_local(self.up_proj_bias)
            up = up + broadcast_expert_bias(up_proj_bias, num_tokens_per_expert, up.shape[0]).bfloat16()

        gate = None
        if self.gate_proj is not None:
            gate_proj = to_local(self.gate_proj).transpose(-2, -1)
            gate = self.grouped_gemm(x_bf16, gate_proj.bfloat16(), offs=offsets)
            if self.gate_proj_bias is not None:
                gate_proj_bias = to_local(self.gate_proj_bias)
                gate = gate + broadcast_expert_bias(gate_proj_bias, num_tokens_per_expert, gate.shape[0]).bfloat16()

        hidden = self.activation.apply(gate, up)
        down_proj = to_local(self.down_proj).transpose(-2, -1)
        output = self.grouped_gemm(hidden, down_proj.bfloat16(), offs=offsets)
        if self.down_proj_bias is not None:
            down_proj_bias = to_local(self.down_proj_bias)
            output = output + broadcast_expert_bias(down_proj_bias, num_tokens_per_expert, output.shape[0]).bfloat16()
        return output.type_as(x)

    def init_weights(self, init_std: float):
        first_projection = self.gate_proj if self.gate_proj is not None else self.up_proj
        nn.init.trunc_normal_(first_projection, mean=0.0, std=0.02)
        remaining = (self.up_proj, self.down_proj) if self.gate_proj is not None else (self.down_proj,)
        for weight in remaining:
            nn.init.trunc_normal_(weight, mean=0.0, std=init_std)
        for bias in (self.gate_proj_bias, self.up_proj_bias, self.down_proj_bias):
            if bias is not None:
                nn.init.zeros_(bias)


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
        self.register_buffer(
            "selection_bias",
            torch.zeros(num_experts, dtype=torch.float32) if selection_bias else None,
        )
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
        self,
        x: torch.Tensor,
        routed_experts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs*slen, dim)``.
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
            if self.score_func in ("softmax", "topk_softmax"):
                routing_confidence_sum = top_scores.sum()
            else:
                selected_probability_mass = top_scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
                routing_confidence_sum = selected_probability_mass.sum()

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


class MoE(nn.Module):
    """Token-choice MoE runtime composed from a router, grouped experts, and optional projections."""

    @classmethod
    def from_args(
        cls,
        args: MoEArgs,
        dim: int,
        hidden_dim: int,
        *,
        shared_expert: FeedForward | None,
    ) -> "MoE":
        experts = GroupedExperts(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=args.num_experts,
            expert_type=args.expert_type,
            activation=args.activation,
        )
        router = TokenChoiceTopKRouter(
            dim=dim,
            num_experts=args.num_experts,
            top_k=args.top_k,
            score_func=args.score_func,
            route_norm=args.route_norm,
            route_scale=args.route_scale,
            selection_bias=args.load_balance_coeff is not None,
        )
        return cls(
            router=router,
            experts=experts,
            shared_expert=shared_expert,
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
    ) -> None:
        super().__init__()
        self.router = router
        self.experts = experts
        self.shared_expert = shared_expert
        self.score_before_experts = score_before_experts

        self.token_dispatcher: TokenDispatcher = LocalTokenDispatcher(
            num_experts=experts.num_experts,
            top_k=router.top_k,
            token_group_alignment=experts.token_group_alignment,
        )
        # define fields for auxiliary-loss-free load balancing (https://arxiv.org/abs/2408.15664)
        # NOTE: tokens_per_expert is accumulated in the model forward pass.
        #       router.selection_bias is updated outside the model in an optimizer step pre hook
        #       to work with gradient accumulation.
        self.load_balance_coeff = load_balance_coeff
        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff > 0.0
        # tokens_per_expert tracks expert usage for selection-bias updates and metrics.
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(experts.num_experts, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer("routing_confidence_sum", torch.tensor(0.0, dtype=torch.float32), persistent=False)

    def set_token_dispatcher(self, token_dispatcher: TokenDispatcher) -> None:
        self.token_dispatcher = token_dispatcher

    def prepare_expert_input(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def prepare_expert_output(self, x: torch.Tensor) -> torch.Tensor:
        return x

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
        ) = self.router(x, routed_experts=routed_experts)

        # Accumulate expert usage for selection-bias updates and metrics.
        with torch.no_grad():
            record_moe_routing_statistics(
                self.tokens_per_expert,
                self.routing_confidence_sum,
                num_tokens_per_expert,
                routing_confidence_sum,
            )

        routed_output = self.token_dispatcher.run(
            self.prepare_expert_input(x),
            top_scores,
            selected_experts_indices,
            self.experts,
            score_before_experts=self.score_before_experts,
        )

        shared_output = None
        if self.shared_expert is not None:
            shared_output = self.shared_expert(x)

        self.token_dispatcher.synchronize()

        routed_output = self.prepare_expert_output(routed_output)

        if shared_output is not None:
            routed_output = routed_output + shared_output

        return routed_output.reshape(bs, slen, dim)

    def init_weights(
        self,
        init_std: float,
        buffer_device: torch.device,
    ):
        self.experts.init_weights(init_std)
        self.router.init_weights(init_std)
        if self.shared_expert is not None:
            self.shared_expert.init_weights(init_std)

        with torch.device(buffer_device):
            self.tokens_per_expert = torch.zeros(self.experts.num_experts, dtype=torch.float32)
            self.routing_confidence_sum = torch.tensor(0.0, dtype=torch.float32)
            if self.router.selection_bias is not None:
                self.router.selection_bias = torch.zeros(self.experts.num_experts, dtype=torch.float32)
