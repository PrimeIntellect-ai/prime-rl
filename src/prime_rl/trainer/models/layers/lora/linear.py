import torch
from torch import nn

from prime_rl.trainer.models.layers.lora.base import LoRAModule, lora_parameter


class LoRALinear(LoRAModule):
    """Linear layer with a low-rank adapter: base(x) + alpha / rank * B(A(dropout(x)))."""

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float = 32.0, dropout: float = 0.0):
        super().__init__(base_layer, rank, alpha, dropout)
        self.lora_A = lora_parameter(rank, base_layer.in_features, like=base_layer.weight)
        self.lora_B = lora_parameter(base_layer.out_features, rank, like=base_layer.weight)
        self.reset_parameters()

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        return {"lora_A.weight": self.lora_A.detach(), "lora_B.weight": self.lora_B.detach()}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2d = x.view(-1, x.shape[-1])
        base_out = self.base_layer(x2d)
        lora_out = self.lora_dropout(x2d) @ self.lora_A.T @ self.lora_B.T
        return (base_out + self.scaling * lora_out).view(*x.shape[:-1], -1)
