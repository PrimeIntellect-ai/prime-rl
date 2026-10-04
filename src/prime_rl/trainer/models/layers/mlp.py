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
