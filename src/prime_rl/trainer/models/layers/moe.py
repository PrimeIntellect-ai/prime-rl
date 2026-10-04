# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from prime_rl.trainer.distributed.token_dispatcher import LocalTokenDispatcher, TokenDispatcher
from prime_rl.trainer.models.fusions import fuse_gate_up_projections
from prime_rl.trainer.models.layers.activations import ActivationDispatch, ActivationType
from prime_rl.trainer.models.layers.expert_compute import BF16ExpertCompute, ExpertCompute
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


@torch.library.custom_op("prime_rl::record_qb_margins", mutates_args=("hist", "margin_range"))
def record_qb_margins(
    hist: torch.Tensor, margin_range: torch.Tensor, grid: torch.Tensor, margins: torch.Tensor
) -> None:
    """Add per-expert margins ``(tokens, experts)`` to ``hist`` ``(experts, bins)`` binned over ``grid = [lo, hi]``.

    Margins outside the grid land in the edge bins, so the count above any in-grid threshold stays exact.
    ``margin_range`` tracks ``[-min, max]`` of the margins seen, the next step's grid.
    """
    num_experts, num_bins = hist.shape
    bins = ((margins - grid[0]) * (num_bins / (grid[1] - grid[0]))).long().clamp_(0, num_bins - 1)
    bins += torch.arange(num_experts, device=bins.device) * num_bins
    hist.view(-1).index_add_(0, bins.view(-1), torch.ones(1, dtype=hist.dtype, device=hist.device).expand(bins.numel()))
    margin_range.copy_(torch.maximum(margin_range, torch.stack([-margins.min(), margins.max()])))


@record_qb_margins.register_fake
def _record_qb_margins_fake(
    hist: torch.Tensor, margin_range: torch.Tensor, grid: torch.Tensor, margins: torch.Tensor
) -> None:
    return None


def qb_upper_quantile(hist: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, top_k: int) -> torch.Tensor:
    """Per-expert threshold ``beta`` with ``tokens * top_k / experts`` margins above it (Kimi K3 QB, Eq. 14).

    ``hist`` holds ``(experts, bins)`` margin counts over ``[lo, hi]``; ``beta`` is interpolated linearly
    within the bin where the count from the top crosses the target.
    """
    num_experts, num_bins = hist.shape
    counts = hist.double()
    target = counts[0].sum() * top_k / num_experts
    above = counts.flip(-1).cumsum(-1).flip(-1)
    crossing = ((above >= target).sum(-1, keepdim=True) - 1).clamp(0, num_bins - 1)
    fraction = (above.gather(-1, crossing) - target) / counts.gather(-1, crossing).clamp(min=1)
    return (lo + (hi - lo) / num_bins * (crossing + fraction).squeeze(-1)).float()


@torch.no_grad()
def update_quantile_balancing(model: nn.Module, group: dist.ProcessGroup) -> dict[str, float]:
    """Set each QB router's selection bias from this step's margins, pooled over ``group``.

    Called once per step after the last micro-batch, so the new bias routes the next step only.
    """
    routers = [m.router for m in model.modules() if isinstance(m, MoE) and m.router.qb_hist is not None]
    if not routers:
        return {}
    hist = torch.cat([router.qb_hist.view(-1) for router in routers])
    margin_range = torch.stack([router.qb_margin_range for router in routers])
    dist.all_reduce(hist, group=group)
    dist.all_reduce(margin_range, op=dist.ReduceOp.MAX, group=group)

    for router, layer_hist, layer_range in zip(routers, hist.split([r.qb_hist.numel() for r in routers]), margin_range):
        beta = qb_upper_quantile(layer_hist.view_as(router.qb_hist), *router.qb_grid, router.top_k)
        router.selection_bias.copy_(beta.mean() - beta)
        router.qb_grid.copy_(layer_range * torch.tensor([-1.0, 1.0], device=layer_range.device))
        router.reset_quantile_balancing_stats()

    bias = torch.stack([router.selection_bias for router in routers])
    stats = torch.stack(
        [-margin_range[:, 0].max(), margin_range[:, 1].max(), bias.min(), bias.max(), bias.std(dim=1).mean()]
    ).tolist()
    return dict(zip(["qb/margin_min", "qb/margin_max", "qb/bias_min", "qb/bias_max", "qb/bias_std"], stats))


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


class GroupedExperts(nn.Module):
    supported_fusions = {"gate_up": fuse_gate_up_projections}

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        *,
        expert_type: ExpertType = "gated",
        activation: ActivationType = "silu",
        bias: bool = False,
        compute: ExpertCompute | None = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.gate_proj = nn.Parameter(torch.empty(num_experts, hidden_dim, dim)) if expert_type == "gated" else None
        self.up_proj = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.register_parameter("gate_up_proj", None)
        self.down_proj = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.gate_proj_bias = (
            nn.Parameter(torch.empty(num_experts, hidden_dim)) if bias and self.gate_proj is not None else None
        )
        self.up_proj_bias = nn.Parameter(torch.empty(num_experts, hidden_dim)) if bias else None
        self.down_proj_bias = nn.Parameter(torch.empty(num_experts, dim)) if bias else None

        self.activation = ActivationDispatch[activation]
        if expert_type == "non_gated":
            self.supported_fusions = {}
        self.set_compute(compute or BF16ExpertCompute())

    def set_compute(self, compute: ExpertCompute) -> None:
        compute.validate(self)
        self.compute = compute

    @property
    def token_group_alignment(self) -> int:
        return self.compute.token_group_alignment

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        return self.compute(self, x, num_tokens_per_expert)

    def init_weights(self, init_std: float):
        if self.gate_up_proj is None:
            first_projection = self.gate_proj if self.gate_proj is not None else self.up_proj
            nn.init.trunc_normal_(first_projection, mean=0.0, std=0.02)
            remaining = (self.up_proj, self.down_proj) if self.gate_proj is not None else (self.down_proj,)
        else:
            gate_proj, up_proj = self.gate_up_proj.chunk(2, dim=1)
            nn.init.trunc_normal_(gate_proj, mean=0.0, std=0.02)
            remaining = (up_proj, self.down_proj)
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
        self.register_buffer("qb_hist", None)
        # Set via model.moe_router_dtype='float32': the gate weight is kept in fp32
        # (exempt from FSDP bf16 casting) and the gate GEMM runs in fp32.
        self.fp32_gate = False

    def enable_quantile_balancing(self, num_bins: int) -> None:
        """Route with top-(k+1) and histogram each expert's margin ``score - cutoff`` for the QB bias update."""
        device = self.selection_bias.device
        self.register_buffer(
            "qb_hist", torch.zeros(self.num_experts, num_bins, dtype=torch.int32, device=device), persistent=False
        )
        self.register_buffer("qb_margin_range", torch.zeros(2, device=device), persistent=False)
        self.register_buffer("qb_grid", torch.zeros(2, device=device), persistent=False)

    def reset_quantile_balancing(self) -> None:
        # Margins of [0, 1] scores under a zero bias lie in [-1, 1]; later steps use the previous step's range.
        self.qb_grid.copy_(torch.tensor([-1.0, 1.0]))
        self.reset_quantile_balancing_stats()

    def reset_quantile_balancing_stats(self) -> None:
        self.qb_hist.zero_()
        self.qb_margin_range.fill_(-float("inf"))

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
            if self.qb_hist is not None and torch.is_grad_enabled():
                # The (k+1)-th biased score is the cutoff an expert must beat to enter the token's top-k.
                top_selection, selected_experts_indices = torch.topk(selection_scores, k=self.top_k + 1, dim=1)
                selected_experts_indices = selected_experts_indices[:, :-1].contiguous()
                margins = scores.detach().float() - top_selection[:, -1:].detach()
                record_qb_margins(self.qb_hist, self.qb_margin_range, self.qb_grid, margins)
            else:
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
