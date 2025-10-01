from dataclasses import dataclass

import torch
from torch import nn
from transformers.integrations import use_kernel_forward_from_hub


@dataclass
class RMSNormConfig:
    hidden_size: int
    eps: float = 1e-6


@use_kernel_forward_from_hub("RMSNorm")
class RMSNorm(nn.Module):
    def __init__(self, config: RMSNormConfig) -> None:
        """
        Glm4MoeRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.variance_epsilon = config.eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
