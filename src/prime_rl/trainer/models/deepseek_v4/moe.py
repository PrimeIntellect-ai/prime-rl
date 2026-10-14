"""DeepSeek V4 mixture of experts: router, routed experts and shared expert.

Both MLP layer types live here. `num_hash_layers` picks between standard token-choice
routing and the hash routing of the bootstrap layers, which replaces the learned
selection with a frozen token-id lookup but keeps the learned gating weights.
"""

import torch
from torch import nn

from prime_rl.trainer.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from prime_rl.trainer.models.layers.activations import ClampedSilu
from prime_rl.trainer.models.layers.mlp import FeedForward
from prime_rl.trainer.models.layers.moe import GroupedExperts, MoE, TokenChoiceTopKRouter


class DeepseekV4HashRouter(TokenChoiceTopKRouter):
    """A bootstrap layer's router: selection is a frozen token-id lookup, gating still learned.

    `tid2eid` replaces the top-k of the scores with `tid2eid[token_id]`, read from the checkpoint
    and zeros until one fills it. A frozen selection cannot be steered, so a hash layer has no use
    for the aux-loss-free load-balancing bias, and HF's `DeepseekV4HashRouter` carries no
    `e_score_correction_bias` to load into it either; `selection_bias=False` leaves that buffer
    unbuilt, keeping the state dict aligned with HF's.
    """

    def __init__(self, *, vocab_size: int, **router_kwargs) -> None:
        super().__init__(**router_kwargs, selection_bias=False)
        self.register_buffer("tid2eid", torch.zeros(vocab_size, self.top_k, dtype=torch.long), persistent=True)


class DeepseekV4MoE(MoE):
    """A V4 MoE layer, hash-routed or standard according to `config.num_hash_layers`.

    Subclasses the shared `MoE` so `configure_moe_runtime` / `setup_fsdp` keep recognizing
    it, then hands the three V4-specific pieces to the base constructor. `MoE.forward`'s
    orchestration is unchanged.

    A hash layer routes each token to `tid2eid[token_id]`, the frozen table its
    `DeepseekV4HashRouter` carries, instead of to the top-k of the router's scores. That is
    precisely what the shared router's `routed_experts` bypass does, so only the `forward` below
    differs between the two layer types: it reads the indices out of the table and lets the base
    class weight them with the learned scores as usual.
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        assert config.hidden_act == "silu", (
            f"the experts hardcode SiLU; hidden_act={config.hidden_act!r} is not supported"
        )
        assert not config.mlp_bias, "mlp_bias is not supported"
        is_hash = layer_idx < config.num_hash_layers

        router_kwargs = dict(
            dim=config.hidden_size,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            score_func=config.scoring_func,
            # HF normalizes the top-k scores unconditionally and never reads
            # `config.norm_topk_prob`, so neither do we.
            route_norm=True,
            route_scale=config.routed_scaling_factor,
        )
        router = (
            DeepseekV4HashRouter(**router_kwargs, vocab_size=config.vocab_size)
            if is_hash
            else TokenChoiceTopKRouter(**router_kwargs, selection_bias=True)
        )
        activation = ClampedSilu(config.swiglu_limit)
        experts = GroupedExperts(
            dim=config.hidden_size,
            hidden_dim=config.moe_intermediate_size,
            num_experts=config.n_routed_experts,
            activation=activation,
        )
        # HF sizes its shared expert at `moe_intermediate_size` regardless of
        # `n_shared_experts`, which therefore only decides whether one exists at all.
        shared_expert = (
            FeedForward(dim=config.hidden_size, hidden_dim=config.moe_intermediate_size, activation=activation)
            if config.n_shared_experts > 0
            else None
        )

        super().__init__(
            router=router,
            experts=experts,
            shared_expert=shared_expert,
            # HF scales each expert's output by its routing weight after `down_proj`.
            score_before_experts=False,
            load_balance_coeff=None if is_hash else 1e-3,
        )
        self.layer_idx = layer_idx
        self.is_hash = is_hash

    def init_weights(self, init_std: float, buffer_device: torch.device) -> None:
        super().init_weights(init_std, buffer_device)
        # HF draws both halves of its fused gate_up_proj from std=0.02; the shared init uses init_std for up.
        nn.init.trunc_normal_(self.experts.up_proj, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        routed_experts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs, slen, dim)``.
            input_ids (torch.Tensor | None, optional): Token ids with shape ``(bs, slen)``.
                Required by a hash layer, ignored by a standard one.
            routed_experts (torch.Tensor | None, optional): Optional tensor with shape
                ``(bs, slen, top_k)``. Replayed expert indices take precedence over the table.

        Returns:
            out (torch.Tensor): Output tensor with shape ``(bs, slen, dim)``.
        """
        if self.is_hash and routed_experts is None:
            assert input_ids is not None, f"layer {self.layer_idx} is hash-routed and needs input_ids"
            # `(vocab_size, top_k)` indexed by `(bs, slen)` token ids gives `(bs, slen, top_k)`.
            routed_experts = self.router.tid2eid[input_ids]
        return super().forward(x, routed_experts=routed_experts)
