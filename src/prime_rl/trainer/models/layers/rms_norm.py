from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class RMSNormConfig:
    hidden_size: int
    eps: float = 1e-6


class RMSNorm(nn.Module):
    def __init__(self, config: RMSNormConfig) -> None:
        """
        Glm4MoeRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.variance_epsilon = config.eps

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        if residual is not None:
            hidden_states = hidden_states + residual
            residual = residual.to(input_dtype)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        if residual is None:
            return hidden_states
        else:
            return hidden_states, residual

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
