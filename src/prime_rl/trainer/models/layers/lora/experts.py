import math

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from prime_rl.trainer.models.layers.activations import ClampedSwiglu
from prime_rl.trainer.models.layers.expert_compute import broadcast_expert_bias
from prime_rl.trainer.models.layers.lora.base import LoRAModule, lora_parameter
from prime_rl.trainer.models.layers.moe import GroupedExperts


def _to_local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _full(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach()
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def _run_lora_grouped_mm(
    x: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Per-expert low-rank product via grouped matrix multiplication.

    Args:
        x: Input tensor [total_tokens, in_features]
        lora_A: Low-rank A matrices [num_experts, rank, in_features]
        lora_B: Low-rank B matrices [num_experts, out_features, rank]
        offsets: Cumulative token counts per expert [num_experts]

    Returns:
        LoRA output [total_tokens, out_features]
    """
    _a_out = torch._grouped_mm(x.bfloat16(), _to_local(lora_A).bfloat16().transpose(-2, -1), offs=offsets)
    return torch._grouped_mm(_a_out, _to_local(lora_B).bfloat16().transpose(-2, -1), offs=offsets)


def _check_grouped_mm_dims(*dims: int) -> None:
    if any(dim % 8 != 0 for dim in dims):
        raise ValueError("grouped_mm requires rank and expert dimensions divisible by 8")


class LoRAGroupedExperts(LoRAModule):
    """
    Gated GroupedExperts + LoRA on gate_proj, up_proj and down_proj.
    Exports the vLLM per-expert MoE LoRA format.
    """

    def __init__(self, base_layer: GroupedExperts, rank: int, alpha: float = 32.0, dropout: float = 0.0):
        super().__init__(base_layer)
        if base_layer.gate_proj is None:
            raise ValueError("LoRAGroupedExperts requires gated experts")

        self.num_experts = base_layer.num_experts
        self.dim = base_layer.gate_proj.shape[2]
        self.hidden_dim = base_layer.gate_proj.shape[1]
        _check_grouped_mm_dims(rank, self.dim, self.hidden_dim)

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        n, like = self.num_experts, base_layer.gate_proj
        self.gate_proj_lora_A = lora_parameter(n, rank, self.dim, like=like)
        self.gate_proj_lora_B = lora_parameter(n, self.hidden_dim, rank, like=like)
        self.down_proj_lora_A = lora_parameter(n, rank, self.hidden_dim, like=base_layer.down_proj)
        self.down_proj_lora_B = lora_parameter(n, self.dim, rank, like=base_layer.down_proj)
        self.up_proj_lora_A = lora_parameter(n, rank, self.dim, like=base_layer.up_proj)
        self.up_proj_lora_B = lora_parameter(n, self.hidden_dim, rank, like=base_layer.up_proj)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Kaiming uniform for A, zeros for B."""
        nn.init.kaiming_uniform_(self.gate_proj_lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.gate_proj_lora_B)
        nn.init.kaiming_uniform_(self.down_proj_lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.down_proj_lora_B)
        nn.init.kaiming_uniform_(self.up_proj_lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj_lora_B)

    def get_lora_param_counts(self) -> tuple[int, int]:
        adapter_params = (
            self.gate_proj_lora_A.numel()
            + self.gate_proj_lora_B.numel()
            + self.down_proj_lora_A.numel()
            + self.down_proj_lora_B.numel()
            + self.up_proj_lora_A.numel()
            + self.up_proj_lora_B.numel()
        )
        adapted_params = (
            self.base_layer.gate_proj.numel() + self.base_layer.down_proj.numel() + self.base_layer.up_proj.numel()
        )
        return adapter_params, adapted_params

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        """Per-expert slices, e.g. ``{expert_id}.gate_proj.lora_A.weight`` (vLLM 2D MoE LoRA format)."""
        weights = {
            "gate_proj.lora_A": _full(self.gate_proj_lora_A),
            "gate_proj.lora_B": _full(self.gate_proj_lora_B),
            "down_proj.lora_A": _full(self.down_proj_lora_A),
            "down_proj.lora_B": _full(self.down_proj_lora_B),
            "up_proj.lora_A": _full(self.up_proj_lora_A),
            "up_proj.lora_B": _full(self.up_proj_lora_B),
        }
        # Clone so each tensor owns its storage instead of viewing the full stacked weight.
        return {
            f"{expert_id}.{name}.weight": weight[expert_id].clone()
            for expert_id in range(self.num_experts)
            for name, weight in weights.items()
        }

    def forward(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        gate_proj = _to_local(self.base_layer.gate_proj)
        up_proj = _to_local(self.base_layer.up_proj)
        down_proj = _to_local(self.base_layer.down_proj)

        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        lora_x = self.lora_dropout(x)

        h1_base = torch._grouped_mm(x.bfloat16(), gate_proj.bfloat16().transpose(-2, -1), offs=offsets)
        gate_proj_lora_out = _run_lora_grouped_mm(lora_x, self.gate_proj_lora_A, self.gate_proj_lora_B, offsets)
        h1 = h1_base + self.scaling * gate_proj_lora_out.bfloat16()

        h3_base = torch._grouped_mm(x.bfloat16(), up_proj.bfloat16().transpose(-2, -1), offs=offsets)
        up_proj_lora_out = _run_lora_grouped_mm(lora_x, self.up_proj_lora_A, self.up_proj_lora_B, offsets)
        h3 = h3_base + self.scaling * up_proj_lora_out.bfloat16()

        h = self.base_layer.activation.apply(h1, h3)

        lora_h = self.lora_dropout(h)
        h2_base = torch._grouped_mm(h, down_proj.bfloat16().transpose(-2, -1), offs=offsets)
        down_proj_lora_out = _run_lora_grouped_mm(lora_h, self.down_proj_lora_A, self.down_proj_lora_B, offsets)
        return (h2_base + self.scaling * down_proj_lora_out.bfloat16()).type_as(x)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(base={self.base_layer}, rank={self.rank}, "
            f"num_experts={self.num_experts}, alpha={self.alpha}, dropout={self.lora_dropout})"
        )


class LoRANonGatedGroupedExperts(LoRAModule):
    """
    Non-gated GroupedExperts + LoRA on up_proj and down_proj.
    """

    def __init__(self, base_layer: GroupedExperts, rank: int, alpha: float = 32.0, dropout: float = 0.0):
        super().__init__(base_layer)
        if base_layer.gate_proj is not None:
            raise ValueError("LoRANonGatedGroupedExperts requires non-gated experts")

        self.num_experts = base_layer.num_experts
        # up_proj shape: [num_experts, intermediate_dim, input_dim]
        self.hidden_dim = base_layer.up_proj.shape[1]
        self.dim = base_layer.up_proj.shape[2]
        _check_grouped_mm_dims(rank, self.dim, self.hidden_dim)

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        n = self.num_experts
        self.up_proj_lora_A = lora_parameter(n, rank, self.dim, like=base_layer.up_proj)
        self.up_proj_lora_B = lora_parameter(n, self.hidden_dim, rank, like=base_layer.up_proj)
        self.down_proj_lora_A = lora_parameter(n, rank, self.hidden_dim, like=base_layer.down_proj)
        self.down_proj_lora_B = lora_parameter(n, self.dim, rank, like=base_layer.down_proj)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.up_proj_lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj_lora_B)
        nn.init.kaiming_uniform_(self.down_proj_lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.down_proj_lora_B)

    def get_lora_param_counts(self) -> tuple[int, int]:
        adapter_params = (
            self.up_proj_lora_A.numel()
            + self.up_proj_lora_B.numel()
            + self.down_proj_lora_A.numel()
            + self.down_proj_lora_B.numel()
        )
        adapted_params = self.base_layer.up_proj.numel() + self.base_layer.down_proj.numel()
        return adapter_params, adapted_params

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        weights = {
            "up_proj.lora_A": _full(self.up_proj_lora_A),
            "up_proj.lora_B": _full(self.up_proj_lora_B),
            "down_proj.lora_A": _full(self.down_proj_lora_A),
            "down_proj.lora_B": _full(self.down_proj_lora_B),
        }
        return {
            f"{expert_id}.{name}.weight": weight[expert_id].clone()
            for expert_id in range(self.num_experts)
            for name, weight in weights.items()
        }

    def forward(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        up_proj = _to_local(self.base_layer.up_proj)
        down_proj = _to_local(self.base_layer.down_proj)

        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        lora_x = self.lora_dropout(x)

        h_base = torch._grouped_mm(x.bfloat16(), up_proj.bfloat16().transpose(-2, -1), offs=offsets)
        up_proj_lora_out = _run_lora_grouped_mm(lora_x, self.up_proj_lora_A, self.up_proj_lora_B, offsets)
        h = self.base_layer.activation.apply(None, h_base + self.scaling * up_proj_lora_out.bfloat16())

        lora_h = self.lora_dropout(h)
        out_base = torch._grouped_mm(h, down_proj.bfloat16().transpose(-2, -1), offs=offsets)
        down_proj_lora_out = _run_lora_grouped_mm(lora_h, self.down_proj_lora_A, self.down_proj_lora_B, offsets)
        return (out_base + self.scaling * down_proj_lora_out.bfloat16()).type_as(x)

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
        super().__init__(base_layer)
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
        _check_grouped_mm_dims(rank, self.hidden_size, self.intermediate_size)

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        n = self.num_experts
        self.gate_up_lora_A = lora_parameter(n, rank, self.hidden_size, like=base_layer.up_proj)
        self.gate_up_lora_B = lora_parameter(n, self.gate_up_out, rank, like=base_layer.up_proj)
        self.down_lora_A = lora_parameter(n, rank, self.intermediate_size, like=base_layer.down_proj)
        self.down_lora_B = lora_parameter(n, self.hidden_size, rank, like=base_layer.down_proj)

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
        gu_a = _full(self.gate_up_lora_A)
        gu_b = _full(self.gate_up_lora_B)
        d_a = _full(self.down_lora_A)
        d_b = _full(self.down_lora_B)

        # lora_A: (num_experts, rank, in) -> (num_experts*rank, in)
        gu_a_flat = gu_a.reshape(self.num_experts * self.rank, self.hidden_size).clone()
        d_a_flat = d_a.reshape(self.num_experts * self.rank, self.intermediate_size).clone()
        # lora_B: (num_experts, out, rank) -> (out, rank, num_experts) -> (out, rank*num_experts)
        # vLLM's reshape treats the last dim of lora_B as (rank, num_experts) with experts fast-varying.
        gu_b_flat = gu_b.permute(1, 2, 0).contiguous().reshape(self.gate_up_out, self.rank * self.num_experts)
        d_b_flat = d_b.permute(1, 2, 0).contiguous().reshape(self.hidden_size, self.rank * self.num_experts)

        return {
            "base_layer.lora_A.weight": gu_a_flat,
            "base_layer.lora_B.weight": gu_b_flat,
            "lora_A.weight": d_a_flat,
            "lora_B.weight": d_b_flat,
        }

    def forward(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        gate_proj = _to_local(self.base_layer.gate_proj)
        up_proj = _to_local(self.base_layer.up_proj)
        down_proj = _to_local(self.base_layer.down_proj)
        gate_proj_bias = _to_local(self.base_layer.gate_proj_bias)
        up_proj_bias = _to_local(self.base_layer.up_proj_bias)
        down_proj_bias = _to_local(self.base_layer.down_proj_bias)

        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        lora_x = self.lora_dropout(x)

        gate = torch._grouped_mm(x.bfloat16(), gate_proj.bfloat16().transpose(-2, -1), offs=offsets)
        gate = gate + broadcast_expert_bias(gate_proj_bias, num_tokens_per_expert, gate.shape[0]).bfloat16()
        up = torch._grouped_mm(x.bfloat16(), up_proj.bfloat16().transpose(-2, -1), offs=offsets)
        up = up + broadcast_expert_bias(up_proj_bias, num_tokens_per_expert, up.shape[0]).bfloat16()

        gate_up_lora = _run_lora_grouped_mm(lora_x, self.gate_up_lora_A, self.gate_up_lora_B, offsets)
        gate = gate + self.scaling * gate_up_lora[..., ::2].bfloat16()
        up = up + self.scaling * gate_up_lora[..., 1::2].bfloat16()

        h = self.base_layer.activation.apply(gate, up)
        lora_h = self.lora_dropout(h)

        out_base = torch._grouped_mm(h, down_proj.bfloat16().transpose(-2, -1), offs=offsets)
        out_base = out_base + broadcast_expert_bias(down_proj_bias, num_tokens_per_expert, out_base.shape[0]).bfloat16()
        out_lora = _run_lora_grouped_mm(lora_h, self.down_lora_A, self.down_lora_B, offsets)
        return (out_base + self.scaling * out_lora.bfloat16()).type_as(x)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(base={self.base_layer}, rank={self.rank}, "
            f"num_experts={self.num_experts}, alpha={self.alpha}, dropout={self.lora_dropout})"
        )
