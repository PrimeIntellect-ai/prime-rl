import math

import torch
from torch import nn

from prime_rl.trainer.models.layers.lora.base import LoRAModule, lora_parameter


class LoRALinear(LoRAModule):
    """Linear layer with a low-rank adapter: base(x) + alpha / rank * B(A(dropout(x)))."""

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float = 32.0, dropout: float = 0.0):
        super().__init__(base_layer)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.lora_A = lora_parameter(rank, base_layer.in_features, like=base_layer.weight)
        self.lora_B = lora_parameter(base_layer.out_features, rank, like=base_layer.weight)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        return {"lora_A.weight": self.lora_A.detach(), "lora_B.weight": self.lora_B.detach()}

    def get_lora_param_counts(self) -> tuple[int, int]:
        return self.lora_A.numel() + self.lora_B.numel(), self.base_layer.weight.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return self.base_layer(x) + self.scaling * lora_out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(base={self.base_layer}, rank={self.rank}, "
            f"alpha={self.alpha}, dropout={self.lora_dropout})"
        )
