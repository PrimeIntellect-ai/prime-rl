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
from prime_rl.trainer.models.layers.moe import GroupedExperts, MoE, TokenChoiceTopKRouter


class _ClampedSwiglu:
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


class DeepseekV4Router(TokenChoiceTopKRouter):
    """Token-choice router scored with `sqrt(softplus(.))`.

    `TokenChoiceTopKRouter.forward` picks its scoring function from an inline
    `if/elif/else: raise` chain with no hook to extend, and `"sqrtsoftplus"` is outside
    the `ScoreFuncType` it accepts, so the whole method is restated below. Only the
    scoring branch is new: the `routed_experts` bypass, the selection bias, the
    normalization, the scaling and the per-expert token count are the base class's,
    unchanged.
    """

    def forward(
        self, x: torch.Tensor, routed_experts: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert routed_experts is None or routed_experts.shape[-1] == self.top_k, (
            f"routed_experts shape: {routed_experts.shape}, top_k: {self.top_k}"
        )
        if self.fp32_gate:
            gate_bias = self.gate.bias.float() if self.gate.bias is not None else None
            logits = F.linear(x.float(), self.gate.weight.float(), gate_bias)
        else:
            logits = self.gate(x)

        # Scoring runs in float32 to avoid loss explosion, as in the base class. HF scores
        # in the input dtype instead, so bf16 activations drift from HF by ~1e-3 here.
        if self.score_func == "sqrtsoftplus":
            scores = F.softplus(logits.float()).sqrt()
        elif self.score_func == "sigmoid":
            scores = torch.sigmoid(logits.float())
        elif self.score_func == "softmax":
            scores = F.softmax(logits.float(), dim=1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        # NOTE: the selection bias only steers selection. The gating value top_scores is
        #       still derived from the original scores.
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
            _, selected_experts_indices = torch.topk(selection_scores, k=self.top_k, dim=1, sorted=self.topk_sorted)
            top_scores = scores.gather(dim=1, index=selected_experts_indices)

        with torch.no_grad():
            if self.score_func == "softmax":
                routing_confidence_sum = top_scores.sum()
            else:
                selected_probability_mass = top_scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
                routing_confidence_sum = selected_probability_mass.sum()

        if self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator
        top_scores = top_scores * self.route_scale

        num_tokens_per_expert = torch.histc(
            selected_experts_indices.reshape(-1).float(),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        ).to(torch.int64)

        return top_scores, selected_experts_indices, num_tokens_per_expert, routing_confidence_sum


class DeepseekV4Experts(GroupedExperts):
    """Routed experts with V4's clamped SwiGLU.

    The shared `GroupedExperts` already stores the stacked `gate_proj`/`up_proj`/`down_proj`
    that the on-disk per-expert `w1`/`w2`/`w3` convert into, and its `forward` reaches the
    activation through `self.activation`, so only that attribute and the initialization
    spread differ from the base class.
    """

    def __init__(self, dim: int, hidden_dim: int, num_experts: int, swiglu_limit: float) -> None:
        super().__init__(dim, hidden_dim, num_experts, expert_type="gated")
        self.activation = _ClampedSwiglu(swiglu_limit)

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

    A hash layer routes each token to `tid2eid[token_id]`, a frozen table read from the
    checkpoint, instead of to the top-k of the router's scores. That is precisely what the
    shared router's `routed_experts` bypass does, so the router class is the same for both
    layer types and only the `forward` below differs: it looks the indices up and lets the
    base class weight them with the learned scores as usual.
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        assert config.hidden_act == "silu", (
            f"the routed experts hardcode SiLU; hidden_act={config.hidden_act!r} is not supported"
        )

        is_hash = layer_idx < config.num_hash_layers

        router = DeepseekV4Router(
            dim=config.hidden_size,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            score_func=config.scoring_func,
            # HF normalizes the top-k scores unconditionally and never reads
            # `config.norm_topk_prob`, so neither do we.
            route_norm=True,
            route_scale=config.routed_scaling_factor,
            # A frozen selection cannot be steered, so a hash layer has no use for the
            # aux-loss-free load-balancing bias, and HF's `DeepseekV4HashRouter` carries no
            # `e_score_correction_bias` to load into it either. `False` leaves the
            # `selection_bias` buffer unbuilt, keeping the state dict aligned with HF's.
            selection_bias=not is_hash,
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

        if is_hash:
            # HF hangs the table off the router (`gate.tid2eid`); prime-rl's router is
            # shared by both layer types, so it lives here instead, exactly like the
            # `selection_bias` it stands in for. Zeros until a checkpoint fills it.
            self.register_buffer(
                "tid2eid",
                torch.zeros(config.vocab_size, config.num_experts_per_tok, dtype=torch.long),
                persistent=True,
            )

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
            bs, slen, _ = x.shape
            routed_experts = self.tid2eid[input_ids.reshape(-1)].view(bs, slen, -1)
        return super().forward(x, routed_experts=routed_experts)
