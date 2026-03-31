from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn
from transformers.integrations import use_kernel_forward_from_hub


@lru_cache(maxsize=1)
def _get_quack_rmsnorm():
    """Lazy-load quack rmsnorm. Returns None if unavailable or GPU is pre-Hopper."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        return None
    try:
        from quack import rmsnorm

        return rmsnorm
    except ImportError:
        return None


class _ContiguousGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad.contiguous()


def _contiguous_grad(x: torch.Tensor) -> torch.Tensor:
    """Identity in forward, makes gradient contiguous in backward.

    Quack's RMSNorm backward kernel requires contiguous gradients (stride[-1]==1)
    but upstream ops like attention permute can produce non-contiguous ones.
    """
    return _ContiguousGrad.apply(x) if x.requires_grad else x


@dataclass
class RMSNormConfig:
    hidden_size: int
    eps: float = 1e-6


@use_kernel_forward_from_hub("RMSNorm")
class RMSNorm(nn.Module):
    def __init__(self, config: RMSNormConfig) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.variance_epsilon = config.eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        quack_fn = _get_quack_rmsnorm() if hidden_states.is_cuda else None
        if quack_fn is not None:
            out = quack_fn(hidden_states, self.weight, eps=self.variance_epsilon)
            return _contiguous_grad(out)
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        return F.layer_norm(x.float(), (self.dim,), self.weight.float(), self.bias.float(), self.eps).type_as(x)
