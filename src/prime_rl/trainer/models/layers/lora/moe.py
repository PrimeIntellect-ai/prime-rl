import math

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from prime_rl.trainer.models.layers.activations import ClampedSwiglu
from prime_rl.trainer.models.layers.expert_compute import broadcast_expert_bias
from prime_rl.trainer.models.layers.lora.base import LoRAModule
from prime_rl.trainer.models.layers.moe import GroupedExperts


def _run_lora_grouped_mm(
    x: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Apply LoRA via grouped matrix multiplication.

    Args:
        x: Input tensor [total_tokens, in_features]
        lora_A: Low-rank A matrices [num_experts, rank, in_features]
        lora_B: Low-rank B matrices [num_experts, out_features, rank]
        offsets: Cumulative token counts per expert [num_experts]

    Returns:
        LoRA output [total_tokens, out_features]
    """
    _a_out = torch._grouped_mm(x.bfloat16(), lora_A.bfloat16().transpose(-2, -1), offs=offsets)
    lora_out = torch._grouped_mm(_a_out, lora_B.bfloat16().transpose(-2, -1), offs=offsets)
    return lora_out


def _permute_for_ep(
    base_layer: GroupedExperts, x: torch.Tensor, num_tokens_per_expert: torch.Tensor, experts_per_ep_rank: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Standard EP needs token permutation into local expert order; DeepEP tokens are already dispatched.

    Returns the (possibly permuted) input, its per-expert token counts and the permutation indices (None if unpermuted).
    """
    if getattr(base_layer, "ep_comm_backend", "torch") == "deepep":
        return x, num_tokens_per_expert, None

    from torchtitan.experiments.kernels.moe.indices import generate_permute_indices

    from prime_rl.trainer.distributed.expert_parallel import TOKEN_GROUP_ALIGN_SIZE_M

    num_ep_ranks = num_tokens_per_expert.shape[0] // experts_per_ep_rank
    with torch.no_grad():
        permuted_indices, num_tokens_per_expert, _ = generate_permute_indices(
            num_tokens_per_expert,
            experts_per_ep_rank,
            num_ep_ranks,
            x.shape[0] + experts_per_ep_rank * TOKEN_GROUP_ALIGN_SIZE_M,
            TOKEN_GROUP_ALIGN_SIZE_M,
        )
    x = torch.vstack((x, x.new_zeros((x.shape[-1]))))
    return x[permuted_indices, :], num_tokens_per_expert, permuted_indices


def _unpermute_for_ep(out: torch.Tensor, permuted_indices: torch.Tensor | None, num_tokens: int) -> torch.Tensor:
    """Undo ``_permute_for_ep``, restoring the dispatched token order of ``num_tokens`` tokens."""
    if permuted_indices is None:
        return out
    out_unpermuted = out.new_zeros((num_tokens + 1, out.shape[-1]))
    out_unpermuted[permuted_indices, :] = out
    return out_unpermuted[:-1]


class LoRAGroupedExperts(LoRAModule):
    """
    GroupedExperts + LoRA with grouped GEMM.
    Adapts every expert projection (gate_proj for gated experts, up_proj and down_proj).
    Compatible with vLLM's 2D per-expert MoE LoRA format when broadcasting weights.
    """

    def __init__(self, base_layer: GroupedExperts, rank: int, alpha: float = 32.0, dropout: float = 0.0):
        super().__init__(base_layer, rank, alpha, dropout)
        self.num_experts = base_layer.num_experts
        # up_proj shape: [num_experts, hidden_dim, dim]
        self.hidden_dim = base_layer.up_proj.shape[1]
        self.dim = base_layer.up_proj.shape[2]

        if rank % 8 != 0 or self.dim % 8 != 0 or self.hidden_dim % 8 != 0:
            raise ValueError("grouped_mm requires rank and expert dimensions divisible by 8")

        self.gated = base_layer.gate_proj is not None
        # Order sets parameter init RNG order and adapter state dict order
        self.projections = ("gate_proj", "down_proj", "up_proj") if self.gated else ("up_proj", "down_proj")
        for proj in self.projections:
            in_dim, out_dim = (self.hidden_dim, self.dim) if proj == "down_proj" else (self.dim, self.hidden_dim)
            base_weight = getattr(base_layer, proj)
            factory_kwargs = {"device": base_weight.device, "dtype": base_weight.dtype}
            self.register_parameter(
                f"{proj}_lora_A", nn.Parameter(torch.empty(self.num_experts, rank, in_dim, **factory_kwargs))
            )
            self.register_parameter(
                f"{proj}_lora_B", nn.Parameter(torch.empty(self.num_experts, out_dim, rank, **factory_kwargs))
            )

        self.reset_parameters()

    def _lora_weights(self, proj: str) -> tuple[nn.Parameter, nn.Parameter]:
        return getattr(self, f"{proj}_lora_A"), getattr(self, f"{proj}_lora_B")

    def reset_parameters(self) -> None:
        for proj in self.projections:
            lora_A, lora_B = self._lora_weights(proj)
            nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
            nn.init.zeros_(lora_B)

    def get_lora_param_counts(self) -> tuple[int, int]:
        adapter_params = sum(a.numel() + b.numel() for a, b in map(self._lora_weights, self.projections))
        adapted_params = sum(getattr(self.base_layer, proj).numel() for proj in self.projections)
        return adapter_params, adapted_params

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        """Per-expert slices, e.g. ``{expert_id}.gate_proj.lora_A.weight``."""
        weights = {}
        for proj in self.projections:
            for name, param in zip(("lora_A", "lora_B"), self._lora_weights(proj)):
                weight = param.detach()
                # With EP, LoRA weights are DTensors sharded across expert-parallel ranks.
                # Gather them before per-expert indexing.
                if isinstance(weight, DTensor):
                    weight = weight.full_tensor()
                weights[proj, name] = weight

        # The clone is necessary to avoid views that cause giant memory spikes
        return {
            f"{expert_id}.{proj}.{name}.weight": weight[expert_id].clone()
            for expert_id in range(self.num_experts)
            for (proj, name), weight in weights.items()
        }

    def forward(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        base_weights = {proj: getattr(self.base_layer, proj) for proj in self.projections}
        lora_weights = {proj: self._lora_weights(proj) for proj in self.projections}

        num_tokens = x.shape[0]
        permuted_indices = None
        if isinstance(base_weights["up_proj"], DTensor):
            base_weights = {proj: w.to_local() for proj, w in base_weights.items()}
            lora_weights = {proj: (a.to_local(), b.to_local()) for proj, (a, b) in lora_weights.items()}
            x, num_tokens_per_expert, permuted_indices = _permute_for_ep(
                self.base_layer, x, num_tokens_per_expert, base_weights["up_proj"].shape[0]
            )

        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

        def project(proj: str, h: torch.Tensor, lora_h: torch.Tensor) -> torch.Tensor:
            base_out = torch._grouped_mm(h, base_weights[proj].bfloat16().transpose(-2, -1), offs=offsets)
            lora_out = _run_lora_grouped_mm(lora_h, *lora_weights[proj], offsets)
            return base_out + self.scaling * lora_out.bfloat16()

        lora_x = self.lora_dropout(x)
        gate = project("gate_proj", x.bfloat16(), lora_x) if self.gated else None
        h = self.base_layer.activation.apply(gate, project("up_proj", x.bfloat16(), lora_x))
        out = project("down_proj", h, self.lora_dropout(h)).type_as(x)

        return _unpermute_for_ep(out, permuted_indices, num_tokens)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(base={self.base_layer}, rank={self.rank}, "
            f"num_experts={self.num_experts}, alpha={self.alpha}, dropout={self.lora_dropout})"
        )


class LoRAGptOssGroupedExperts(LoRAModule):
    """
    GPT-OSS GroupedExperts + LoRA.

    Preserves GPT-OSS's combined gate/up adapter format while applying it to the
    canonical split gate_proj/up_proj runtime weights.
    """

    def __init__(self, base_layer: GroupedExperts, rank: int, alpha: float = 32.0, dropout: float = 0.0):
        super().__init__(base_layer, rank, alpha, dropout)
        if base_layer.gate_proj is None or base_layer.activation is not ClampedSwiglu:
            raise ValueError("LoRAGptOssGroupedExperts requires gated GPT-OSS experts")
        if any(
            bias is None for bias in (base_layer.gate_proj_bias, base_layer.up_proj_bias, base_layer.down_proj_bias)
        ):
            raise ValueError("GPT-OSS experts require projection biases")

        self.num_experts = base_layer.num_experts
        self.hidden_size = base_layer.up_proj.shape[2]
        self.intermediate_size = base_layer.up_proj.shape[1]
        self.gate_up_out = 2 * self.intermediate_size

        if rank % 8 != 0 or self.hidden_size % 8 != 0 or self.intermediate_size % 8 != 0:
            raise ValueError("grouped_mm requires rank and expert dimensions divisible by 8")

        up_kwargs = {"device": base_layer.up_proj.device, "dtype": base_layer.up_proj.dtype}
        down_kwargs = {"device": base_layer.down_proj.device, "dtype": base_layer.down_proj.dtype}
        self.gate_up_lora_A = nn.Parameter(torch.empty(self.num_experts, rank, self.hidden_size, **up_kwargs))
        self.gate_up_lora_B = nn.Parameter(torch.empty(self.num_experts, self.gate_up_out, rank, **up_kwargs))
        self.down_lora_A = nn.Parameter(torch.empty(self.num_experts, rank, self.intermediate_size, **down_kwargs))
        self.down_lora_B = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, rank, **down_kwargs))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.gate_up_lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.gate_up_lora_B)
        nn.init.kaiming_uniform_(self.down_lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.down_lora_B)

    def get_lora_param_counts(self) -> tuple[int, int]:
        adapter_params = (
            self.gate_up_lora_A.numel()
            + self.gate_up_lora_B.numel()
            + self.down_lora_A.numel()
            + self.down_lora_B.numel()
        )
        adapted_params = (
            self.base_layer.gate_proj.numel() + self.base_layer.up_proj.numel() + self.base_layer.down_proj.numel()
        )
        return adapter_params, adapted_params

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        """vLLM-compatible 3D MoE adapter format.

        For 3D MoE models (gpt-oss in vLLM has `is_3d_moe_weight = True`), vLLM expects:
        - `experts.base_layer.lora_{A,B}.weight` for the gate_up projection
        - `experts.lora_{A,B}.weight` for the down projection
        with experts stacked into the rank dim. See
        vllm/lora/model_manager.py::_stack_moe_lora_weights, which reshapes
            lora_A: (num_experts*rank, in)  -> (num_experts, rank, in)
            lora_B: (out, rank*num_experts) -> (out, rank, num_experts) -> (num_experts, out, rank)
        """
        detached_gu_a = self.gate_up_lora_A.detach()
        detached_gu_b = self.gate_up_lora_B.detach()
        detached_d_a = self.down_lora_A.detach()
        detached_d_b = self.down_lora_B.detach()

        if isinstance(detached_gu_a, DTensor):
            detached_gu_a = detached_gu_a.full_tensor()
            detached_gu_b = detached_gu_b.full_tensor()
            detached_d_a = detached_d_a.full_tensor()
            detached_d_b = detached_d_b.full_tensor()

        # lora_A: (num_experts, rank, in) -> (num_experts*rank, in)
        gu_a_flat = detached_gu_a.reshape(self.num_experts * self.rank, self.hidden_size).clone()
        d_a_flat = detached_d_a.reshape(self.num_experts * self.rank, self.intermediate_size).clone()
        # lora_B: (num_experts, out, rank) -> (out, rank, num_experts) -> (out, rank*num_experts)
        # vLLM's reshape treats the last dim of lora_B as (rank, num_experts) with experts fast-varying.
        gu_b_flat = detached_gu_b.permute(1, 2, 0).contiguous().reshape(self.gate_up_out, self.rank * self.num_experts)
        d_b_flat = detached_d_b.permute(1, 2, 0).contiguous().reshape(self.hidden_size, self.rank * self.num_experts)

        return {
            "base_layer.lora_A.weight": gu_a_flat,
            "base_layer.lora_B.weight": gu_b_flat,
            "lora_A.weight": d_a_flat,
            "lora_B.weight": d_b_flat,
        }

    def forward(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        gu_a, gu_b = self.gate_up_lora_A, self.gate_up_lora_B
        d_a, d_b = self.down_lora_A, self.down_lora_B

        gate_proj = self.base_layer.gate_proj
        up_proj = self.base_layer.up_proj
        down_proj = self.base_layer.down_proj
        gate_proj_bias = self.base_layer.gate_proj_bias
        up_proj_bias = self.base_layer.up_proj_bias
        down_proj_bias = self.base_layer.down_proj_bias

        num_tokens = x.shape[0]
        permuted_indices = None
        if isinstance(up_proj, DTensor):
            gate_proj = gate_proj.to_local()
            up_proj = up_proj.to_local()
            down_proj = down_proj.to_local()
            gate_proj_bias = gate_proj_bias.to_local()
            up_proj_bias = up_proj_bias.to_local()
            down_proj_bias = down_proj_bias.to_local()
            gu_a = gu_a.to_local()
            gu_b = gu_b.to_local()
            d_a = d_a.to_local()
            d_b = d_b.to_local()
            x, num_tokens_per_expert, permuted_indices = _permute_for_ep(
                self.base_layer, x, num_tokens_per_expert, up_proj.shape[0]
            )

        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        lora_x = self.lora_dropout(x)

        gate = torch._grouped_mm(x.bfloat16(), gate_proj.bfloat16().transpose(-2, -1), offs=offsets)
        gate = gate + broadcast_expert_bias(gate_proj_bias, num_tokens_per_expert, gate.shape[0]).bfloat16()
        up = torch._grouped_mm(x.bfloat16(), up_proj.bfloat16().transpose(-2, -1), offs=offsets)
        up = up + broadcast_expert_bias(up_proj_bias, num_tokens_per_expert, up.shape[0]).bfloat16()

        gate_up_lora = _run_lora_grouped_mm(lora_x, gu_a, gu_b, offsets)
        gate = gate + self.scaling * gate_up_lora[..., ::2].bfloat16()
        up = up + self.scaling * gate_up_lora[..., 1::2].bfloat16()

        h = self.base_layer.activation.apply(gate, up)
        lora_h = self.lora_dropout(h)

        out_base = torch._grouped_mm(h, down_proj.bfloat16().transpose(-2, -1), offs=offsets)
        out_base = out_base + broadcast_expert_bias(down_proj_bias, num_tokens_per_expert, out_base.shape[0]).bfloat16()
        out_lora = _run_lora_grouped_mm(lora_h, d_a, d_b, offsets)
        out = (out_base + self.scaling * out_lora.bfloat16()).type_as(x)

        return _unpermute_for_ep(out, permuted_indices, num_tokens)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(base={self.base_layer}, rank={self.rank}, "
            f"num_experts={self.num_experts}, alpha={self.alpha}, dropout={self.lora_dropout})"
        )
