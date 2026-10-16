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
from prime_rl.trainer.models.layers.activations import Activation, ActivationDispatch, ActivationType
from prime_rl.trainer.models.layers.expert_compute import BF16ExpertCompute, ExpertCompute
from prime_rl.trainer.models.layers.mlp import ExpertType, FeedForward

ScoreFuncType = Literal["softmax", "sigmoid", "topk_softmax", "sqrtsoftplus"]


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
    num_null_experts: int = 0
    batch_topk_max_k: int = 0

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
        activation: ActivationType | Activation = "silu",
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

        self.activation = ActivationDispatch[activation] if isinstance(activation, str) else activation
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
        score_func (ScoreFuncType): Score transform. ``topk_softmax`` selects experts from the logits
            and normalizes only the selected logits. ``sqrtsoftplus`` is DeepSeek V4's ``sqrt(softplus(.))``.
        route_norm (bool): Whether to normalize the routing scores when using sigmoid.
        route_scale (float): Scaling factor applied to the routing scores.
        gate_bias (bool): Whether the gate has a trainable logit bias.
        selection_bias (bool): Whether to keep a persistent selection-only bias. The bias affects
            expert selection but not routing weights.
        topk_sorted (bool): Whether selected experts are returned in descending score order.
        num_null_experts (int): Zero-output experts next to the routed ones (AdaMoE). They are scored, selected and
            balanced like the others, so a token that selects them uses fewer routed experts.
        batch_topk_max_k (int): Above 0, position-wise batch top-k while ``batch_rows`` is set: at every position,
            the ``batch_rows`` rows (on every rank of ``batch_group``) select their ``top_k * rows`` highest
            (token, expert) selection scores together, each token from its own ``batch_topk_max_k`` best, so
            ``top_k`` is the mean per token. Unselected slots have index -1 and score 0.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        score_func: ScoreFuncType,
        route_norm: bool,
        route_scale: float,
        *,
        gate_bias: bool = False,
        selection_bias: bool = False,
        topk_sorted: bool = True,
        num_null_experts: int = 0,
        batch_topk_max_k: int = 0,
    ):
        super().__init__()
        num_experts += num_null_experts
        self.gate = nn.Linear(dim, num_experts, bias=gate_bias)
        self.register_buffer(
            "selection_bias",
            torch.zeros(num_experts, dtype=torch.float32) if selection_bias else None,
        )
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_null_experts = num_null_experts
        self.batch_topk_max_k = batch_topk_max_k
        # Set by the training loop: rows per forward, and the data-parallel group whose rows route together.
        self.batch_rows: int | None = None
        self.batch_group: dist.ProcessGroup | None = None
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs*slen, dim)``.
            routed_experts (torch.Tensor | None, optional): Optional tensor with shape ``(bs * slen, top_k)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - top_scores (torch.Tensor):
                    Routing scores for selected experts with shape ``(bs*slen, top_k)``.
                - selected_experts_indices (torch.Tensor):
                    Expert indices selected for each token with shape ``(bs*slen, top_k)``.
                - num_tokens_per_expert (torch.Tensor):
                    Number of tokens assigned to each expert with shape ``(num_experts,)``.
                - routing_confidence_sum (torch.Tensor):
                    Sum over tokens of the selected-expert probability mass before route normalization/scaling.
                - scores (torch.Tensor):
                    Scores of all experts with shape ``(bs*slen, num_experts)``.
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
        elif self.score_func == "sqrtsoftplus":
            scores = F.softplus(logits.float()).sqrt()
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
            if self.batch_topk_max_k and self.batch_rows is not None:
                candidate_scores, selected_experts_indices = torch.topk(
                    selection_scores, k=self.batch_topk_max_k, dim=1
                )
                selected = self.batch_select(candidate_scores.detach())
                selected_experts_indices = selected_experts_indices.masked_fill(~selected, -1)
                top_scores = scores.gather(dim=1, index=selected_experts_indices.clamp_min(0)) * selected
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

        return top_scores, selected_experts_indices, num_tokens_per_expert, routing_confidence_sum, scores

    @property
    def variable_k(self) -> bool:
        """Whether tokens can use different numbers of routed experts."""
        return self.num_null_experts > 0 or self.batch_topk_max_k > 0

    @property
    def dispatch_top_k(self) -> int:
        """Routes per dispatched row: variable-k routing dispatches one (token, expert) pair per row."""
        return 1 if self.variable_k else self.top_k

    def batch_select(self, candidate_scores: torch.Tensor) -> torch.Tensor:
        """Which of the ``(rows * slen, batch_topk_max_k)`` candidates win their position's batch top-k. Every rank
        of ``batch_group`` gathers all candidates and computes the same winners."""
        rows, max_k = self.batch_rows, self.batch_topk_max_k
        candidates = candidate_scores.view(rows, -1, max_k)
        rank = 0
        if self.batch_group is not None:
            rank = dist.get_rank(self.batch_group)
            gathered = candidates.new_empty(dist.get_world_size(self.batch_group) * rows, *candidates.shape[1:])
            dist.all_gather_into_tensor(gathered, candidates.contiguous(), group=self.batch_group)
            candidates = gathered
        per_position = candidates.transpose(0, 1).reshape(candidates.shape[1], -1)
        winners = per_position.topk(self.top_k * candidates.shape[0], dim=1, sorted=False).indices
        selected = torch.zeros_like(per_position, dtype=torch.bool).scatter_(1, winners, True)
        selected = selected.view(per_position.shape[0], -1, max_k)[:, rank * rows : (rank + 1) * rows]
        return selected.transpose(0, 1).reshape(-1, max_k)

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=init_std)
        if self.gate.bias is not None:
            nn.init.zeros_(self.gate.bias)


def sequence_balance(scores: torch.Tensor, selected_experts: torch.Tensor, num_rows: int, top_k: int) -> torch.Tensor:
    """DeepSeek-V3's sequence-wise balance loss without its coefficient, per row: ``sum_i f_i P_i`` with
    ``f_i`` the row's share of expert ``i``'s selections times ``num_experts / top_k`` (1 when balanced) and
    ``P_i`` its mean score normalized over all experts. 1 for a balanced row, up to ``num_experts / top_k``.

    Args:
        scores: ``(num_rows * slen, num_experts)`` scores of all experts.
        selected_experts: ``(num_rows * slen, slots)`` selected experts, -1 for an unused slot.
        top_k: Mean selections per token.
    """
    num_experts = scores.shape[1]
    selections = selected_experts.view(num_rows, -1)
    counts = torch.zeros(num_rows, num_experts + 1, device=scores.device).scatter_add_(
        1, selections.remainder(num_experts + 1), torch.ones_like(selections, dtype=torch.float32)
    )[:, :num_experts]
    f = counts * (num_experts / (scores.shape[0] // num_rows * top_k))
    p = (scores / scores.sum(dim=-1, keepdim=True)).view(num_rows, -1, num_experts).mean(dim=1)
    return (f * p).sum(dim=-1)


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
            num_null_experts=args.num_null_experts,
            batch_topk_max_k=args.batch_topk_max_k,
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
            top_k=router.dispatch_top_k,
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
            torch.zeros(router.num_experts, dtype=torch.float32),
            persistent=False,
        )
        # Training tokens per number of routed experts used, for variable-k routing.
        self.register_buffer(
            "routed_k_counts",
            torch.zeros((router.batch_topk_max_k or router.top_k) + 1, dtype=torch.float32)
            if router.variable_k
            else None,
            persistent=False,
        )
        self.register_buffer("routing_confidence_sum", torch.tensor(0.0, dtype=torch.float32), persistent=False)
        # Sequence-wise balance (DeepSeek-V3 eq. 17-20): with balance_rows set by the training loop, which knows
        # how many packed rows a sample holds, every forward with grad computes the per-row balance for its loss.
        self.balance_rows: int | None = None
        self.seq_balance: torch.Tensor | None = None

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
            scores,
        ) = self.router(x, routed_experts=routed_experts)

        if self.balance_rows is not None and torch.is_grad_enabled():
            self.seq_balance = sequence_balance(scores, selected_experts_indices, self.balance_rows, self.router.top_k)

        # Accumulate expert usage for selection-bias updates and metrics.
        with torch.no_grad():
            record_moe_routing_statistics(
                self.tokens_per_expert,
                self.routing_confidence_sum,
                num_tokens_per_expert,
                routing_confidence_sum,
            )

        if self.router.variable_k:
            # Dispatch only the (token, routed expert) pairs: no unselected slots or null experts.
            routed = (selected_experts_indices >= 0) & (selected_experts_indices < self.experts.num_experts)
            if torch.is_grad_enabled():
                with torch.no_grad():
                    self.routed_k_counts += torch.bincount(routed.sum(dim=1), minlength=self.routed_k_counts.numel())
            token_indices, slots = routed.nonzero(as_tuple=True)
            pair_output = self.token_dispatcher.run(
                self.prepare_expert_input(x)[token_indices],
                top_scores[token_indices, slots].unsqueeze(1),
                selected_experts_indices[token_indices, slots].unsqueeze(1),
                self.experts,
                score_before_experts=self.score_before_experts,
            )
            routed_output = (
                torch.zeros(x.shape, dtype=torch.float32, device=x.device)
                .index_add(0, token_indices, pair_output.float())
                .to(x.dtype)
            )
        else:
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
            self.tokens_per_expert = torch.zeros(self.router.num_experts, dtype=torch.float32)
            self.routing_confidence_sum = torch.tensor(0.0, dtype=torch.float32)
            if self.routed_k_counts is not None:
                self.routed_k_counts = torch.zeros_like(self.routed_k_counts, device=buffer_device)
            if self.router.selection_bias is not None:
                self.router.selection_bias = torch.zeros(self.router.num_experts, dtype=torch.float32)
