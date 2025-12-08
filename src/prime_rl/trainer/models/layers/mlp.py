from dataclasses import dataclass

from torch import nn
from transformers.activations import ACT2FN


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

    def forward(self, x):
        down_proj = self.down_proj(self.gate_act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
