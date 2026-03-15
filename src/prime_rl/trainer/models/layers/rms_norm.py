from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from transformers.integrations import use_kernel_forward_from_hub


RMSNormImplementation = Literal["torch", "quack"]


@dataclass
class RMSNormConfig:
    hidden_size: int
    eps: float = 1e-6
    impl: RMSNormImplementation = "torch"


@use_kernel_forward_from_hub("RMSNorm")
class RMSNorm(nn.Module):
    def __init__(self, config: RMSNormConfig) -> None:
        """
        Glm4MoeRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.variance_epsilon = config.eps
        self.impl = config.impl
        self._quack_rmsnorm = None
        if self.impl == "quack":
            try:
                from quack import rmsnorm as quack_rmsnorm
            except ImportError as exc:
                raise RuntimeError("'model.rms_norm_impl=\"quack\"' requires quack-kernels to be installed") from exc
            self._quack_rmsnorm = quack_rmsnorm

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.impl == "quack":
            if hidden_states.device.type != "cuda":
                raise RuntimeError("'model.rms_norm_impl=\"quack\"' requires CUDA tensors at runtime")
            assert self._quack_rmsnorm is not None
            return self._quack_rmsnorm(hidden_states, self.weight, eps=self.variance_epsilon)

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
