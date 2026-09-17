import torch
import torch.nn.functional as F
from torch import nn


class Qwen3_5RMSNorm(nn.Module):
    """Qwen's zero-centered RMSNorm, whose effective scale is ``1 + weight``."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_size))
        self.variance_epsilon = eps
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.square().mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return ((1.0 + self.weight.float()) * hidden_states).to(input_dtype)


class Qwen3_5RMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float, activation: str) -> None:
        super().__init__()
        if activation not in ("silu", "swish", "sigmoid"):
            raise ValueError(f"Unsupported activation: {activation}")
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.activation = activation

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.square().mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states * self.weight.float()

        gate = gate.float()
        gate = torch.sigmoid(gate) if self.activation == "sigmoid" else F.silu(gate)
        return (hidden_states * gate).to(input_dtype)


__all__ = ["Qwen3_5RMSNorm", "Qwen3_5RMSNormGated"]
