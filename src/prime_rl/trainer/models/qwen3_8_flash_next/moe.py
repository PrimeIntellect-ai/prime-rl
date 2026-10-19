from prime_rl.trainer.models.layers.activations import ActivationType
from prime_rl.trainer.models.layers.mlp import SigmoidGatedFeedForward
from prime_rl.trainer.models.layers.moe import GroupedExperts, MoE, TokenChoiceTopKRouter


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
        router = TokenChoiceTopKRouter(
            dim=dim,
            num_experts=num_experts,
            top_k=top_k,
            score_func="softmax",
            route_norm=True,
            route_scale=1.0,
            selection_bias=load_balance_coeff is not None,
        )
        shared_expert = SigmoidGatedFeedForward(
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


__all__ = ["SigmoidOutputGatedMoE"]
