import torch
from torch import nn

from prime_rl.trainer.models.layers.activations import ActivationType
from prime_rl.trainer.models.layers.mlp import FeedForward
from prime_rl.trainer.models.layers.moe.base import GroupedExperts, MoE, TokenChoiceTopKRouter


class SigmoidOutputGatedFeedForward(FeedForward):
    """Gated feed-forward layer with a sigmoid output gate."""

    def __init__(self, dim: int, hidden_dim: int, activation: ActivationType) -> None:
        super().__init__(
            dim=dim,
            hidden_dim=hidden_dim,
            expert_type="gated",
            activation=activation,
        )
        self.output_gate = nn.Linear(dim, 1, bias=False)

    def forward(self, x: torch.Tensor, routed_experts: torch.Tensor | None = None) -> torch.Tensor:
        return torch.sigmoid(self.output_gate(x)) * super().forward(x, routed_experts)

    def init_weights(self, init_std: float = 0.02) -> None:
        super().init_weights(init_std)
        nn.init.trunc_normal_(self.output_gate.weight, mean=0.0, std=init_std)


class SigmoidOutputGatedMoE(MoE):
    """Top-k MoE with renormalized softmax scores and one sigmoid-gated shared expert."""

    def __init__(
        self,
        *,
        dim: int,
        expert_hidden_dim: int,
        shared_expert_hidden_dim: int,
        num_experts: int,
        top_k: int,
        activation: ActivationType,
        init_std: float,
        load_balance_coeff: float | None = None,
    ) -> None:
        experts = GroupedExperts(
            dim=dim,
            hidden_dim=expert_hidden_dim,
            num_experts=num_experts,
            expert_type="gated",
            activation=activation,
        )
        experts.init_weights(init_std)
        # TODO: Align the router projection dtype with vLLM during end-to-end KL validation.
        router = TokenChoiceTopKRouter(
            dim=dim,
            num_experts=num_experts,
            top_k=top_k,
            score_func="softmax",
            route_norm=True,
            route_scale=1.0,
            selection_bias=load_balance_coeff is not None,
        )
        shared_expert = SigmoidOutputGatedFeedForward(
            dim=dim,
            hidden_dim=shared_expert_hidden_dim,
            activation=activation,
        )
        super().__init__(
            router=router,
            experts=experts,
            shared_expert=shared_expert,
            score_before_experts=False,
            load_balance_coeff=load_balance_coeff,
        )


__all__ = ["SigmoidOutputGatedFeedForward", "SigmoidOutputGatedMoE"]
