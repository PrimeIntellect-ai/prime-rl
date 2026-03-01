from dataclasses import dataclass

import torch
from torch import nn
from transformers.activations import ACT2FN

from prime_rl.trainer.models.layers.quack_backend import (
    quack_gated_linear_func,
    quack_kernels_enabled,
    quack_linear_gated_func,
)

_GATE_ACT2QUACK_GATED_ACT = {
    "silu": "swiglu",
    "gelu": "geglu",
    "gelu_pytorch_tanh": "geglu",
    "relu": "reglu",
    "sigmoid": "glu",
}


@dataclass
class MLPConfig:
    hidden_size: int
    intermediate_size: int
    gate_act: str
    bias: bool


class MLP(nn.Module):
    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.gate_act_fn = ACT2FN[config.gate_act]
        self._quack_activation = _GATE_ACT2QUACK_GATED_ACT.get(config.gate_act)

    def forward(self, x):
        if self._quack_activation is not None and quack_kernels_enabled():
            gate_up_weight = torch.stack((self.gate_proj.weight, self.up_proj.weight), dim=1).reshape(
                2 * self.intermediate_size, self.hidden_size
            )
            preact, postact = quack_linear_gated_func(
                x,
                gate_up_weight,
                activation=self._quack_activation,
                store_preact=torch.is_grad_enabled(),
            )
            return quack_gated_linear_func(
                preact,
                self.down_proj.weight,
                postact,
                activation=self._quack_activation,
            )

        down_proj = self.down_proj(self.gate_act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
