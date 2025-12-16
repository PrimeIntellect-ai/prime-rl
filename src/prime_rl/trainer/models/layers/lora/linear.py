import math

import torch
from torch import nn

from .base import LoRAModule


class LoRALinear(LoRAModule):
    """
    LoRA (Low-Rank Adaptation) linear layer.
    Implements the low-rank decomposition: ΔW = B @ A
    where A ∈ R^(rank x in_features), B ∈ R^(out_features x rank)
    Forward pass: y = x @ (W + ΔW).T = x @ W.T + x @ A.T @ B.T * (alpha / rank)
    """

    def __init__(
        self,
        base_layer: nn.Module,
        rank: int,
        in_features: int | None = None,
        out_features: int | None = None,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__(base_layer)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        if in_features is None:
            in_features = base_layer.in_features
        if out_features is None:
            out_features = base_layer.out_features

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self._init_lora_parameters()

    def _init_lora_parameters(self, generator: torch.Generator | None = None) -> None:
        """Initialize LoRA parameters following standard LoRA initialization."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5), generator=generator)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: base_output + lora_output"""
        base_output = self.base_layer(x)
        lora_x = self.lora_dropout(x)
        lora_output = (lora_x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base_output + lora_output
