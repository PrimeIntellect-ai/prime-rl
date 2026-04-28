# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from prime_rl.configs.trainer import EPCommBackend, ExpertBackend
from prime_rl.trainer.models.layers.moe.experts import GroupedExperts, NonGatedGroupedExperts
from prime_rl.trainer.models.layers.moe.ffn import BCFeedForward, BCNonGatedFeedForward
from prime_rl.trainer.models.layers.moe.routing import NemotronHRouter, TokenChoiceTopKRouter, TokenReorderer


@dataclass
class MoEArgs:
    num_experts: int = 8
    num_shared_experts: int = 1

    # router
    score_func: Literal["softmax", "sigmoid"] = "sigmoid"
    route_norm: bool = False
    route_scale: float = 1.0
    score_before_experts: bool = True

    # token-choice
    top_k: int = 1
    use_grouped_mm: bool = True  # grouped mm or for-loop for the experts computation
    load_balance_coeff: float | None = 1e-3

    def __post_init__(self):
        assert self.use_grouped_mm, "use_grouped_mm must be True"


class MoE(nn.Module):
    """
    The flow goes as follows:
    1. forward()
    2. if self.ep_comm_backend == "deepep":
        deepep_forward() -> maybe_checkpoint(run_experts_for_deepep()) -> self.experts()
    3. else:
        maybe_checkpoint(run_experts_for_torch_a2a()) -> expert_parallel(self.experts())

    # TODO (matej): unify the two flows into 1 when selective AC refactor lands
    """

    def __init__(self, moe_args: MoEArgs, dim: int, hidden_dim: int):
        super().__init__()

        num_experts = moe_args.num_experts
        self.experts = GroupedExperts(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            use_grouped_mm=moe_args.use_grouped_mm,
        )
        self.router = TokenChoiceTopKRouter(
            dim=dim,
            num_experts=num_experts,
            top_k=moe_args.top_k,
            score_func=moe_args.score_func,
            route_norm=moe_args.route_norm,
            route_scale=moe_args.route_scale,
        )
        self.reorderer = TokenReorderer(num_experts=num_experts, top_k=moe_args.top_k)
        # TODO: Add the s back and use FF when the weights support it
        self.shared_expert = (
            BCFeedForward(dim=dim, hidden_dim=hidden_dim * moe_args.num_shared_experts)
            if moe_args.num_shared_experts > 0
            else None
        )
        self.score_before_experts = moe_args.score_before_experts

        # define fields for auxiliary-loss-free load balancing (https://arxiv.org/abs/2408.15664)
        # NOTE: tokens_per_expert is accumulated in the model forward pass.
        #       expert_bias is updated outside the model in an optimizer step pre hook
        #       to work with gradient accumulation.
        self.load_balance_coeff = moe_args.load_balance_coeff
        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff > 0.0
            self.register_buffer(
                "expert_bias",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias = None
        # tokens_per_expert will be used to track expert usage and to update the expert bias for load balancing
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )

        self.set_ep_comm_backend("torch")
        self.set_expert_backend("grouped_mm")

    def set_ep_comm_backend(self, backend: EPCommBackend) -> None:
        """Recursively set the EP comm backend - the experts will also set the backend and their required forward functions."""
        self.ep_comm_backend = backend
        self.experts.set_ep_comm_backend(backend)

    def set_expert_backend(self, backend: ExpertBackend) -> None:
        """Recursively set the expert backend - the experts will also set the backend and their required forward functions."""
        if backend == "sonic":
            assert not self.score_before_experts, "Sonic backend does not support score_before_experts=True."
        self.expert_backend = backend
        self.experts.set_expert_backend(backend)

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
        ) = self.router(x, self.expert_bias, routed_experts=routed_experts)

        # tokens_per_expert will be used to update the expert bias for load balancing.
        # and also to count the expert usage
        # Full block checkpointing can double count tokens_per_expert because it reruns the router
        # in backward. The selective MoE path avoids that by checkpointing only the
        # routed expert compute below.
        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)

        if self.ep_comm_backend == "deepep":
            routed_output = self.deepep_forward(x, selected_experts_indices, top_scores)
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

        routed_output = self.run_experts_for_torch_a2a(
            x,
            token_indices_experts_sorted,
            num_tokens_per_expert,
            top_scores_experts_sorted,
        )
        if self.shared_expert is not None:
            out = self.shared_expert(x)
        else:
            out = torch.zeros_like(x)

        routed_indices = token_indices_experts_sorted.reshape(-1, 1).expand(-1, dim)
        out = out.scatter_add(dim=0, index=routed_indices, src=routed_output)
        out = out.reshape(bs, slen, dim)
        return out

    def run_experts_for_torch_a2a(
        self,
        x: torch.Tensor,
        token_indices_experts_sorted: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        top_scores_experts_sorted: torch.Tensor,
    ) -> torch.Tensor:
        dim = x.shape[-1]
        routed_indices = token_indices_experts_sorted.reshape(-1, 1).expand(-1, dim)
        routed_input = torch.gather(x, dim=0, index=routed_indices)

        if self.score_before_experts:
            routed_input = (routed_input.to(torch.float32) * top_scores_experts_sorted.reshape(-1, 1)).to(x.dtype)

        routed_output = self.experts(routed_input, num_tokens_per_expert)

        if not self.score_before_experts:
            routed_output = (routed_output.to(torch.float32) * top_scores_experts_sorted.reshape(-1, 1)).to(x.dtype)

        return routed_output

    def deepep_forward(
        self,
        x: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        top_scores: torch.Tensor,
    ) -> torch.Tensor:
        from prime_rl.trainer.distributed.deepep import combine_tokens, dispatch_tokens
        from prime_rl.trainer.distributed.expert_parallel import get_ep_group

        if x.shape[0] == 0:
            shared_output = self.shared_expert(x) if self.shared_expert is not None else None
            return x.new_zeros(x.shape) if shared_output is None else shared_output

        group = get_ep_group(self.experts)
        hidden_states, expert_kwargs, combine_kwargs = dispatch_tokens(
            x,
            selected_experts_indices,
            top_scores,
            num_experts=self.experts.num_experts,
            group=group,
            expert_backend=self.expert_backend,
            score_before_experts=self.score_before_experts,
        )
        routed_output = self.run_experts_for_deepep(hidden_states, **expert_kwargs)
        routed_output = combine_tokens(routed_output, **combine_kwargs)

        if self.shared_expert is not None:
            routed_output = routed_output + self.shared_expert(x)

        return routed_output

    def run_experts_for_deepep(
        self,
        x: torch.Tensor,
        **expert_kwargs,
    ) -> torch.Tensor:
        return self.experts(x, **expert_kwargs)

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
            if self.load_balance_coeff is not None:
                self.expert_bias = torch.zeros(self.experts.num_experts, dtype=torch.float32)


class LatentMoE(nn.Module):
    """NemotronH-style Mixture of Experts with latent projections.

    The input is projected to a latent space before expert computation,
    and the output is projected back. Experts use relu2 activation without gating.

    The flow goes as follows:
    1. forward()
    2. if self.ep_comm_backend == "deepep":
        deepep_forward() -> maybe_checkpoint(run_experts_for_deepep()) -> self.experts()
    3. else:
        maybe_checkpoint(run_experts_for_torch_a2a()) -> expert_parallel(self.experts())

    # TODO (matej): unify the two flows into 1 when selective AC refactor lands
    # TODO (matej): this should inherit from some MoE base class
    """

    def __init__(
        self,
        dim: int,
        latent_dim: int | None,
        moe_intermediate_size: int,
        shared_expert_intermediate_size: int,
        num_experts: int,
        top_k: int,
        n_group: int,
        topk_group: int,
        norm_topk_prob: bool,
        routed_scaling_factor: float,
        use_grouped_mm: bool,
        load_balance_coeff: float | None,
    ):
        super().__init__()
        effective_latent_dim = latent_dim if latent_dim is not None else dim

        self.router = NemotronHRouter(
            dim=dim,
            num_experts=num_experts,
            top_k=top_k,
            n_group=n_group,
            topk_group=topk_group,
            norm_topk_prob=norm_topk_prob,
        )
        self.experts = NonGatedGroupedExperts(
            input_dim=effective_latent_dim,
            intermediate_dim=moe_intermediate_size,
            num_experts=num_experts,
            use_grouped_mm=use_grouped_mm,
        )
        self.reorderer = TokenReorderer(num_experts=num_experts, top_k=top_k)
        self.shared_expert = BCNonGatedFeedForward(dim=dim, hidden_dim=shared_expert_intermediate_size)

        if latent_dim is not None:
            self.fc1_latent_proj = nn.Linear(dim, latent_dim, bias=False)
            self.fc2_latent_proj = nn.Linear(latent_dim, dim, bias=False)
        else:
            self.fc1_latent_proj = nn.Identity()
            self.fc2_latent_proj = nn.Identity()

        self.routed_scaling_factor = routed_scaling_factor
        self.load_balance_coeff = load_balance_coeff
        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff > 0.0
            self.register_buffer(
                "expert_bias",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias = None
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )
        self.set_ep_comm_backend("torch")
        self.set_expert_backend("grouped_mm")

    def set_ep_comm_backend(self, backend: EPCommBackend) -> None:
        self.ep_comm_backend = backend
        self.experts.set_ep_comm_backend(backend)

    def set_expert_backend(self, backend: ExpertBackend) -> None:
        self.expert_backend = backend
        self.experts.set_expert_backend(backend)

    def run_experts_for_deepep(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        return self.experts(x, num_tokens_per_expert)

    def run_experts_for_torch_a2a(
        self,
        x: torch.Tensor,
        token_indices_experts_sorted: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        top_scores_experts_sorted: torch.Tensor,
    ) -> torch.Tensor:
        dim = x.shape[-1]
        token_indices_expanded = token_indices_experts_sorted.reshape(-1, 1).expand(-1, dim)
        routed_input = torch.gather(x, dim=0, index=token_indices_expanded)

        routed_input = self.fc1_latent_proj(routed_input)
        routed_output = self.experts(routed_input, num_tokens_per_expert)

        routed_output = (routed_output.float() * top_scores_experts_sorted.reshape(-1, 1)).to(routed_output.dtype)
        routed_output = routed_output * self.routed_scaling_factor

        routed_output = self.fc2_latent_proj(routed_output)
        return routed_output

    def deepep_forward(
        self,
        x: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        top_scores: torch.Tensor,
    ) -> torch.Tensor:
        from prime_rl.trainer.distributed.deepep import (
            combine_tokens,
            dispatch_tokens,
        )
        from prime_rl.trainer.distributed.expert_parallel import get_ep_group

        if x.shape[0] == 0:
            return self.shared_expert(x)

        group = get_ep_group(self.experts)
        # Project before dispatch so DeepEP communicates the smaller latent activations.
        latent_x = self.fc1_latent_proj(x)
        hidden_states, expert_kwargs, combine_kwargs = dispatch_tokens(
            latent_x,
            selected_experts_indices,
            top_scores,
            num_experts=self.experts.num_experts,
            group=group,
            score_before_experts=False,
        )
        routed_output = self.run_experts_for_deepep(hidden_states, **expert_kwargs)
        routed_output = combine_tokens(routed_output, **combine_kwargs)
        shared_output = self.shared_expert(x)
        routed_output = routed_output * self.routed_scaling_factor
        routed_output = self.fc2_latent_proj(routed_output)
        return shared_output + routed_output

    def forward(self, x: torch.Tensor, routed_experts: torch.Tensor | None = None) -> torch.Tensor:
        bs, slen, dim = x.shape
        x_flat = x.view(-1, dim)

        top_scores, selected_experts_indices, num_tokens_per_expert = self.router(x_flat, self.expert_bias)

        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)

        if self.ep_comm_backend == "deepep":
            routed_output = self.deepep_forward(x_flat, selected_experts_indices, top_scores)
            return routed_output.reshape(bs, slen, dim)

        (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
        ) = self.reorderer(top_scores, selected_experts_indices)

        routed_output = self.run_experts_for_torch_a2a(
            x_flat,
            token_indices_experts_sorted,
            num_tokens_per_expert,
            top_scores_experts_sorted,
        )

        out = self.shared_expert(x_flat)

        token_indices_full = token_indices_experts_sorted.reshape(-1, 1).expand(-1, dim)
        out = out.scatter_add(dim=0, index=token_indices_full, src=routed_output)
        out = out.reshape(bs, slen, dim)
        return out

    def init_weights(self, init_std: float, buffer_device: torch.device):
        self.experts.init_weights(init_std)
        self.router.init_weights(init_std)

        with torch.device(buffer_device):
            self.tokens_per_expert = torch.zeros(self.experts.num_experts, dtype=torch.float32)
            if self.load_balance_coeff is not None:
                self.expert_bias = torch.zeros(self.experts.num_experts, dtype=torch.float32)
