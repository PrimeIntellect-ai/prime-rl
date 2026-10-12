import torch
from torch import nn


class RMSNorm(nn.Module):
    """Zero-centered RMSNorm whose effective scale is ``1 + weight``."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_size))
        self.eps = eps
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        variance = normalized.square().mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.eps)
        return ((1.0 + self.weight.float()) * normalized).to(input_dtype)


__all__ = ["RMSNorm"]
