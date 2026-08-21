"""DeepSeek V4 mixture of experts: router, routed experts and shared expert.

Only standard token-choice routing lives here. The hash-routed bootstrap layers
(`mlp_layer_types == "hash_moe"`) are a separate step; see TODO.md.
"""

from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torchtitan.distributed.expert_parallel import expert_parallel

from prime_rl.configs.trainer import EPCommBackend
from prime_rl.trainer.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from prime_rl.trainer.models.layers.mlp import MLP, MLPConfig
from prime_rl.trainer.models.layers.moe import (
    MoE,
    MoEArgs,
    TokenChoiceTopKRouter,
    _selected_probability_mass_sum,
)


class DeepseekV4Router(TokenChoiceTopKRouter):
    """Token-choice router scored with `sqrt(softplus(.))`.

    `TokenChoiceTopKRouter.forward` picks its scoring function from an inline
    `if/elif/else: raise` chain with no hook to extend, so the whole method is restated
    below. Only the scoring branch is new: the `routed_experts` bypass, the
    load-balancing `expert_bias`, the normalization, the scaling and the per-expert token
    count are the base class's, unchanged.
    """

    def forward(
        self, x: torch.Tensor, expert_bias: torch.Tensor | None = None, routed_experts: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert routed_experts is None or routed_experts.shape[-1] == self.top_k, (
            f"routed_experts shape: {routed_experts.shape}, top_k: {self.top_k}"
        )
        scores = self.gate(x.to(torch.float32)) if self.fp32_gate else self.gate(x)

        # Scoring runs in float32 to avoid loss explosion, as in the base class. HF scores
        # in the input dtype instead, so bf16 activations drift from HF by ~1e-3 here.
        if self.score_func == "sqrtsoftplus":
            scores = F.softplus(scores.to(torch.float32)).sqrt()
        elif self.score_func == "sigmoid":
            scores = torch.sigmoid(scores.to(torch.float32))
        elif self.score_func == "softmax":
            scores = F.softmax(scores.to(torch.float32), dim=1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        # NOTE: The expert_bias is only used for routing. The gating value
        #       top_scores is still derived from the original scores.
        if routed_experts is not None:
            top_scores = scores.gather(dim=1, index=routed_experts)
            selected_experts_indices = routed_experts
        elif self.force_balanced:
            num_tokens = scores.shape[0]
            arange = torch.arange(num_tokens * self.top_k, device=scores.device)
            selected_experts_indices = (arange % self.num_experts).view(num_tokens, self.top_k)
            top_scores = scores.gather(dim=1, index=selected_experts_indices)
        elif expert_bias is not None:
            _, selected_experts_indices = torch.topk(scores + expert_bias, k=self.top_k, dim=1)
            top_scores = scores.gather(dim=1, index=selected_experts_indices)
        else:
            top_scores, selected_experts_indices = torch.topk(scores, k=self.top_k, dim=1)

        routing_confidence_sum = _selected_probability_mass_sum(scores, top_scores, self.score_func)

        if self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator
        top_scores = top_scores * self.route_scale

        num_tokens_per_expert = torch.histc(
            selected_experts_indices.reshape(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        return top_scores, selected_experts_indices, num_tokens_per_expert, routing_confidence_sum


def _apply_swiglu_clamp(gate_up: torch.Tensor, limit: float) -> torch.Tensor:
    """SwiGLU over a fused gate/up pair, both clamped as in HF's `DeepseekV4Experts`."""
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


def _run_deepseek_v4_experts_for_loop_impl(
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    _unused: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    limit: float,
) -> torch.Tensor:
    # NOTE: this would incur a synchronization between device and host
    num_tokens_per_expert = num_tokens_per_expert.tolist()

    # side-effect code due to the usage of generate_permute_indices
    num_padding = x.shape[0] - sum(num_tokens_per_expert)

    x = torch.split(
        x[: sum(num_tokens_per_expert)],
        split_size_or_sections=num_tokens_per_expert,
        dim=0,
    )
    out_experts_splits = []
    for expert_idx, x_expert in enumerate(x):
        h = _apply_swiglu_clamp(torch.matmul(x_expert, gate_up_proj[expert_idx].transpose(-2, -1)), limit)
        h = torch.matmul(h, down_proj[expert_idx].transpose(-2, -1))
        out_experts_splits.append(h)
    out = torch.cat(out_experts_splits, dim=0)

    # side-effect code due to the usage of generate_permute_indices
    return torch.vstack((out, out.new_zeros((num_padding, out.shape[-1]))))


def _run_deepseek_v4_experts_grouped_mm_impl(
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    _unused: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    limit: float,
) -> torch.Tensor:
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    assert x.dim() == 2

    gate_up = torch._grouped_mm(x.bfloat16(), gate_up_proj.bfloat16().transpose(-2, -1), offs=offsets)
    h = _apply_swiglu_clamp(gate_up, limit)
    return torch._grouped_mm(h, down_proj.bfloat16().transpose(-2, -1), offs=offsets).type_as(x)


class DeepseekV4Experts(nn.Module):
    """Routed experts holding HF's fused `gate_up_proj` / `down_proj` weights.

    HF's `DeepseekV4Experts.forward` takes unsorted per-token expert indices and builds a
    one-hot mask itself. prime-rl's `MoE` hands its experts tokens already sorted into
    contiguous per-expert blocks, so the signature here is `(x, num_tokens_per_expert)`
    like every other prime-rl expert module. Parameter names and shapes are HF's, so
    checkpoint expert weights need no conversion.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        swiglu_limit: float,
        use_grouped_mm: bool,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * hidden_dim, dim))
        self.down_proj = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.use_grouped_mm = use_grouped_mm
        self.ep_comm_backend: EPCommBackend = "torch"
        # `expert_parallel` pins the wrapped signature to (w1, w2, w3, x, counts), so the
        # clamp limit is bound here instead of being passed per call, and `down_proj`
        # stands in for the unused third weight slot.
        self._for_loop_impl = partial(_run_deepseek_v4_experts_for_loop_impl, limit=swiglu_limit)
        self._grouped_mm_impl = partial(_run_deepseek_v4_experts_grouped_mm_impl, limit=swiglu_limit)
        self._run_for_loop = expert_parallel(self._for_loop_impl)
        self._run_grouped_mm = expert_parallel(self._grouped_mm_impl)

    def set_ep_comm_backend(self, backend: EPCommBackend) -> None:
        self.ep_comm_backend = backend

    def _forward_deepep(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        gate_up_proj = self.gate_up_proj.to_local()
        down_proj = self.down_proj.to_local()
        impl = self._grouped_mm_impl if self.use_grouped_mm else self._for_loop_impl
        return impl(gate_up_proj, down_proj, down_proj, x, num_tokens_per_expert)

    def forward(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        if self.ep_comm_backend == "deepep":
            return self._forward_deepep(x, num_tokens_per_expert)

        run = self._run_grouped_mm if self.use_grouped_mm else self._run_for_loop
        return run(self.gate_up_proj, self.down_proj, self.down_proj, x, num_tokens_per_expert)

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.gate_up_proj, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.down_proj, mean=0.0, std=init_std)


class DeepseekV4MLP(MLP):
    """Dense SwiGLU MLP with V4's clamp, used as the MoE layer's shared expert.

    The shared `MLP` already names its projections the way HF's `LlamaMLP` (which HF's
    `DeepseekV4MLP` subclasses) does, so only the activation changes: gate and up are
    clamped before the SwiGLU. Unlike the routed experts they are separate tensors here,
    so there is nothing to chunk.
    """

    def __init__(self, config: DeepseekV4Config):
        # `MLP` builds its projections without a bias whatever `MLPConfig.bias` says.
        assert not config.mlp_bias, "mlp_bias is not supported by the shared `MLP`"
        super().__init__(
            MLPConfig(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                gate_act=config.hidden_act,
                bias=config.mlp_bias,
            )
        )
        self.limit = config.swiglu_limit

    def forward(self, x: torch.Tensor, routed_experts: torch.Tensor | None = None) -> torch.Tensor:
        gate = self.gate_proj(x).clamp(max=self.limit)
        up = self.up_proj(x).clamp(min=-self.limit, max=self.limit)
        return self.down_proj(self.gate_act_fn(gate) * up)

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.gate_proj.weight, mean=0.0, std=0.02)
        for linear in (self.up_proj, self.down_proj):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class DeepseekV4MoE(MoE):
    """Standard (non-hash) V4 MoE layer.

    Subclasses the shared `MoE` so `apply_ep` / `setup_fsdp` keep recognizing it, then
    swaps in the three V4-specific pieces. `MoE.forward`'s orchestration is unchanged.
    """

    def __init__(self, config: DeepseekV4Config):
        assert config.hidden_act == "silu", (
            f"the routed experts hardcode SiLU; hidden_act={config.hidden_act!r} is not supported"
        )
        assert not getattr(config, "fp8", False), "FP8 training is not supported for DeepSeek V4"

        moe_args = MoEArgs(
            num_experts=config.n_routed_experts,
            num_shared_experts=config.n_shared_experts,
            score_func=config.scoring_func,
            # HF normalizes the top-k scores unconditionally and never reads
            # `config.norm_topk_prob`, so neither do we.
            route_norm=True,
            route_scale=config.routed_scaling_factor,
            # HF scales each expert's output by its routing weight after `down_proj`.
            score_before_experts=False,
            top_k=config.num_experts_per_tok,
            use_grouped_mm=config.use_grouped_mm,
            load_balance_coeff=1e-3,
        )
        super().__init__(moe_args, dim=config.hidden_size, hidden_dim=config.moe_intermediate_size)

        self.router = DeepseekV4Router(
            dim=config.hidden_size,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            score_func=config.scoring_func,
            route_norm=moe_args.route_norm,
            route_scale=moe_args.route_scale,
        )
        self.experts = DeepseekV4Experts(
            dim=config.hidden_size,
            hidden_dim=config.moe_intermediate_size,
            num_experts=config.n_routed_experts,
            swiglu_limit=config.swiglu_limit,
            use_grouped_mm=config.use_grouped_mm,
        )
        self.experts.set_ep_comm_backend(self.ep_comm_backend)
        # HF sizes its shared expert at `moe_intermediate_size` regardless of
        # `n_shared_experts`, which therefore only decides whether one exists at all.
        self.shared_expert = DeepseekV4MLP(config) if config.n_shared_experts > 0 else None
