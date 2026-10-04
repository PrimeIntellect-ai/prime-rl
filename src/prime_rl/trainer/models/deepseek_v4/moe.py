"""DeepSeek V4 mixture of experts: router, routed experts and shared expert.

Both MLP layer types live here. `num_hash_layers` picks between standard token-choice
routing and the hash routing of the bootstrap layers, which replaces the learned
selection with a frozen token-id lookup but keeps the learned gating weights.
"""

import torch
import torch.nn.functional as F
from torch import nn

from prime_rl.trainer.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from prime_rl.trainer.models.layers.mlp import FeedForward
from prime_rl.trainer.models.layers.moe import GroupedExperts, MoE, ScoreFuncType, TokenChoiceTopKRouter


class ClampedSwiglu:
    """SwiGLU with both branches clamped, as in HF's `DeepseekV4Experts`.

    Structurally an `Activation`, but with `limit` as instance state: V4 reads its clamp
    off `config.swiglu_limit`, while the shared `ActivationDispatch` holds stateless
    classes keyed by name and so has nowhere to put a per-model number.
    """

    def __init__(self, limit: float) -> None:
        self.limit = limit

    def apply(self, gate: torch.Tensor | None, up: torch.Tensor) -> torch.Tensor:
        assert gate is not None, "V4's routed experts are gated"
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return F.silu(gate) * up


class DeepseekV4HashRouter(TokenChoiceTopKRouter):
    """A bootstrap layer's router: selection is a frozen token-id lookup, gating still learned.

    `tid2eid` replaces the top-k of the scores with `tid2eid[token_id]`, read from the checkpoint
    and zeros until one fills it. A frozen selection cannot be steered, so a hash layer has no use
    for the aux-loss-free load-balancing bias, and HF's `DeepseekV4HashRouter` carries no
    `e_score_correction_bias` to load into it either; `selection_bias=False` leaves that buffer
    unbuilt, keeping the state dict aligned with HF's.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        score_func: ScoreFuncType,
        route_norm: bool,
        route_scale: float,
        vocab_size: int,
    ) -> None:
        super().__init__(
            dim=dim,
            num_experts=num_experts,
            top_k=top_k,
            score_func=score_func,
            route_norm=route_norm,
            route_scale=route_scale,
            selection_bias=False,
        )
        self.register_buffer("tid2eid", torch.zeros(vocab_size, top_k, dtype=torch.long), persistent=True)


class DeepseekV4Experts(GroupedExperts):
    """Routed experts with V4's clamped SwiGLU.

    The shared `GroupedExperts` already stores the stacked `gate_proj`/`up_proj`/`down_proj`
    that the on-disk per-expert `w1`/`w2`/`w3` convert into, and its `forward` reaches the
    activation through `self.activation`, so only that attribute and the initialization
    spread differ from the base class.
    """

    def __init__(self, dim: int, hidden_dim: int, num_experts: int, swiglu_limit: float) -> None:
        super().__init__(dim, hidden_dim, num_experts, expert_type="gated")
        self.activation = ClampedSwiglu(swiglu_limit)

    def init_weights(self, init_std: float) -> None:
        # Both halves of HF's fused gate_up_proj are drawn from the same std=0.02
        # distribution, so gate and up match that here despite being separate tensors.
        # The base class scales up_proj by init_std instead.
        nn.init.trunc_normal_(self.gate_proj, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.up_proj, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.down_proj, mean=0.0, std=init_std)


class DeepseekV4MLP(FeedForward):
    """Dense SwiGLU MLP with V4's clamp, used as the MoE layer's shared expert.

    The shared `FeedForward` already names its projections the way HF's `LlamaMLP` (which
    HF's `DeepseekV4MLP` subclasses) does, and initializes them the same way, so only the
    activation changes: gate and up are clamped before the SwiGLU. Unlike the routed
    experts they are separate tensors here, so there is nothing to chunk.
    """

    def __init__(self, config: DeepseekV4Config):
        # `FeedForward` builds its projections without a bias whatever `bias` says.
        assert not config.mlp_bias, "mlp_bias is not supported by the shared `FeedForward`"
        super().__init__(
            dim=config.hidden_size,
            hidden_dim=config.moe_intermediate_size,
            activation=config.hidden_act,
        )
        self.limit = config.swiglu_limit

    def forward(self, x: torch.Tensor, routed_experts: torch.Tensor | None = None) -> torch.Tensor:
        gate = self.gate_proj(x).clamp(max=self.limit)
        up = self.up_proj(x).clamp(min=-self.limit, max=self.limit)
        return self.down_proj(F.silu(gate) * up)


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
            f"the routed experts hardcode SiLU; hidden_act={config.hidden_act!r} is not supported"
        )
        if config.scoring_func != "sqrtsoftplus":
            raise ValueError(f"V4 routes with sqrt(softplus(.)); scoring_func={config.scoring_func!r} is not supported")

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
        experts = DeepseekV4Experts(
            dim=config.hidden_size,
            hidden_dim=config.moe_intermediate_size,
            num_experts=config.n_routed_experts,
            swiglu_limit=config.swiglu_limit,
        )
        # HF sizes its shared expert at `moe_intermediate_size` regardless of
        # `n_shared_experts`, which therefore only decides whether one exists at all.
        shared_expert = DeepseekV4MLP(config) if config.n_shared_experts > 0 else None

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
