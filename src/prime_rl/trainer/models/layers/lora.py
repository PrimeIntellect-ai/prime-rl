import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) linear layer.

    Implements the low-rank decomposition: ΔW = B @ A
    where A ∈ R^(rank x in_features), B ∈ R^(out_features x rank)

    Forward pass: y = x @ (W + ΔW).T = x @ W.T + x @ A.T @ B.T * (alpha / rank)
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_linear = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.empty(rank, base_linear.in_features))
        self.lora_B = nn.Parameter(torch.empty(base_linear.out_features, rank))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.reset_parameters()

        for param in self.base_linear.parameters():
            param.requires_grad = False

    def reset_parameters(self):
        """Initialize LoRA parameters following standard LoRA initialization."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: base_output + lora_output"""
        base_output = self.base_linear(x)
        lora_x = self.lora_dropout(x)
        lora_output = (lora_x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base_output + lora_output

    def merge_weights(self) -> nn.Linear:
        """Merge LoRA weights into base layer and return a new linear layer."""
        delta_weight = (self.lora_B @ self.lora_A) * self.scaling
        merged_layer = nn.Linear(
            self.base_linear.in_features,
            self.base_linear.out_features,
            bias=self.base_linear.bias is not None,
            device=self.base_linear.weight.device,
            dtype=self.base_linear.weight.dtype,
        )

        merged_layer.weight.data = self.base_linear.weight.data + delta_weight
        if self.base_linear.bias is not None:
            merged_layer.bias.data = self.base_linear.bias.data.clone()

        return merged_layer

    def __repr__(self):
        return f"{self.__class__.__name__}(base={self.base_linear}, rank={self.rank}, alpha={self.alpha}, dropout={self.lora_dropout})"


def _run_lora_grouped_mm(x: torch.Tensor, lora_A: torch.Tensor, lora_B: torch.Tensor, offsets: torch.LongTensor):
    _a_out = torch._grouped_mm(x, lora_A.transpose(-2, -1), offsets)
    lora_out = torch._grouped_mm(_a_out, lora_B.transpose(-2, -1), offsets)
    return lora_out


def _run_lora_for_loop(x: torch.Tensor, lora_A: torch.Tensor, lora_B: torch.Tensor, offsets: torch.LongTensor):
    lora_out_splits = []
    for i in range(offsets.shape[0]):
        if i == 0:
            _a_out = torch.matmul(x[0 : offsets[i]], lora_A[i].transpose(-2, -1))
            lora_out = torch.matmul(_a_out, lora_B[i].transpose(-2, -1))
        else:
            _a_out = torch.matmul(x[offsets[i - 1] : offsets[i]], lora_A[i].transpose(-2, -1))
            lora_out = torch.matmul(_a_out, lora_B[i].transpose(-2, -1))
        lora_out_splits.append(lora_out)
    return torch.cat(lora_out_splits, dim=0)


class MultiLoRALinear(nn.Module):
    """
    Linear + multi-LoRA with grouped GEMM.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        n_adapters: int,
        alpha: float = 16.0,
        dropout: float = 0.0,
        use_grouped_mm: bool = True,
    ):
        super().__init__()
        if rank <= 0 or n_adapters <= 0:
            raise ValueError("rank and n_adapters must be > 0")

        # Set use_grouped_mm to False if CUDA compute capability < 9.0
        if torch.cuda.is_available():
            cc_major, _ = torch.cuda.get_device_capability()
            if cc_major != 9:
                use_grouped_mm = False
        else:
            use_grouped_mm = False

        self.base_linear = base_linear
        self.rank = rank
        self.n_adapters = n_adapters
        self.alpha = alpha
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.scaling = alpha / rank
        self.use_grouped_mm = use_grouped_mm
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        # LoRA weights: one low-rank pair per adapter
        # [n_adapters, in, r]
        self.lora_A = nn.Parameter(
            torch.empty(
                n_adapters,
                rank,
                self.in_features,
                device=self.base_linear.weight.device,
                dtype=self.base_linear.weight.dtype,
            )
        )
        # [n_adapters, r, out]
        self.lora_B = nn.Parameter(
            torch.empty(
                n_adapters,
                self.out_features,
                rank,
                device=self.base_linear.weight.device,
                dtype=self.base_linear.weight.dtype,
            )
        )

        self.reset_parameters()

        for param in self.base_linear.parameters():
            param.requires_grad = False

    def reset_parameters(self, index: int | None = None):
        if index is None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            nn.init.kaiming_uniform_(self.lora_A[index], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[index])

    def forward(self, x: torch.Tensor, offsets: torch.LongTensor):
        """
        x: [..., in_features]
        offsets: [n_adapters]
        """
        assert x.dim() == 2
        assert offsets[-1] == x.shape[0]

        base_out = self.base_linear(x)
        lora_x = self.lora_dropout(x)
        if self.use_grouped_mm:
            lora_out = _run_lora_grouped_mm(lora_x, self.lora_A, self.lora_B, offsets)
        else:
            lora_out = _run_lora_for_loop(lora_x, self.lora_A, self.lora_B, offsets)
        return base_out + self.scaling * lora_out

    def merge_weights(self, index: int) -> nn.Linear:
        """Merge LoRA weights into base layer and return a new linear layer."""
        delta_weight = (self.lora_B[index] @ self.lora_A[index]) * self.scaling
        merged_layer = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.base_linear.bias is not None,
            device=self.base_linear.weight.device,
            dtype=self.base_linear.weight.dtype,
        )

        merged_layer.weight.data = self.base_linear.weight.data + delta_weight
        if self.base_linear.bias is not None:
            merged_layer.bias.data = self.base_linear.bias.data.clone()

        return merged_layer

    def __repr__(self):
        return f"{self.__class__.__name__}(base={self.base_linear}, rank={self.rank}, n_adapters={self.n_adapters}, alpha={self.alpha}, dropout={self.lora_dropout})"
