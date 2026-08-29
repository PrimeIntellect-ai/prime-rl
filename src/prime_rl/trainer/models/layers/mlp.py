from typing import Literal

import torch
from torch import nn

from prime_rl.trainer.models.layers.activations import ActivationDispatch, ActivationType

ExpertType = Literal["gated", "non_gated"]


class FeedForward(nn.Module):
    """Dense feed-forward layer using the canonical projection names."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        *,
        expert_type: ExpertType = "gated",
        activation: ActivationType = "silu",
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias) if expert_type == "gated" else None
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)
        self.activation = ActivationDispatch[activation]

    def forward(self, x: torch.Tensor, routed_experts: torch.Tensor | None = None) -> torch.Tensor:
        gate = self.gate_proj(x) if self.gate_proj is not None else None
        up = self.up_proj(x)
        return self.down_proj(self.activation.apply(gate, up))

    def init_weights(self, init_std: float = 0.02) -> None:
        first_projection = self.gate_proj if self.gate_proj is not None else self.up_proj
        nn.init.trunc_normal_(first_projection.weight, mean=0.0, std=0.02)
        remaining = (self.up_proj, self.down_proj) if self.gate_proj is not None else (self.down_proj,)
        for linear in remaining:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
        for linear in (self.gate_proj, self.up_proj, self.down_proj):
            if linear is not None and linear.bias is not None:
                nn.init.zeros_(linear.bias)
