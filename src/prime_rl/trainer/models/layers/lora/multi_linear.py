import math

import torch
from torch import nn

from prime_rl.trainer.models.layers.lora.base import MultiLoRAModule, get_lora_num_tokens, get_multilora_scaling


def _run_lora_grouped_mm(
    x: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    offsets: torch.LongTensor,
) -> torch.Tensor:
    _a_out = torch._grouped_mm(x, lora_A.transpose(-2, -1), offsets)
    lora_out = torch._grouped_mm(_a_out, lora_B.transpose(-2, -1), offsets)
    return lora_out


def _run_lora_for_loop(
    x: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    offsets: torch.LongTensor,
) -> torch.Tensor:
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


class MultiLoRALinear(MultiLoRAModule):
    """
    Linear + multi-LoRA with grouped GEMM.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        n_adapters: int,
        alpha: float = 16.0,
        dropout: float = 0.0,
        use_grouped_mm: bool = True,
    ):
        super().__init__(base_layer)
        if rank <= 0 or n_adapters <= 0:
            raise ValueError("rank and n_adapters must be > 0")

        # Set use_grouped_mm to False if CUDA compute capability < 9.0
        if torch.cuda.is_available():
            cc_major, _ = torch.cuda.get_device_capability()
            if cc_major != 9:
                use_grouped_mm = False
        else:
            use_grouped_mm = False
        if rank % 8 != 0 or base_layer.in_features % 8 != 0 or base_layer.out_features % 8 != 0:
            use_grouped_mm = False

        self.rank = rank
        self.n_adapters = n_adapters
        self.alpha = alpha
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.use_grouped_mm = use_grouped_mm
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        self._lora_num_tokens = get_lora_num_tokens()
        self._scaling_factors = get_multilora_scaling()

        # LoRA weights: one low-rank pair per adapter
        # [n_adapters, in, r]
        self.lora_A = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        rank,
                        self.in_features,
                        device=self.base_layer.weight.device,
                        dtype=self.base_layer.weight.dtype,
                    )
                )
                for _ in range(n_adapters)
            ]
        )
        # [n_adapters, r, out]
        self.lora_B = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        self.out_features,
                        rank,
                        device=self.base_layer.weight.device,
                        dtype=self.base_layer.weight.dtype,
                    )
                )
                for _ in range(n_adapters)
            ]
        )

        self.reset_parameters()

    def reset_parameters(self, index: int | None = None) -> None:
        if index is None:
            for i in range(self.n_adapters):
                self.reset_parameters(i)
        else:
            nn.init.kaiming_uniform_(self.lora_A[index], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[index])

    def named_parameters_for_adapter(self, idx: int) -> list[tuple[str, nn.Parameter]]:
        """Get named parameters for a specific adapter index.

        Args:
            idx: The adapter index to get parameters for

        Returns:
            List of (name, parameter) tuples for the specified adapter
        """
        return [
            ("lora_A", self.lora_A[idx]),
            ("lora_B", self.lora_B[idx]),
        ]

    def get_lora_param_counts(self) -> tuple[int, int]:
        """Get the number of LoRA adapter parameters and adapted base parameters.

        Returns:
            A tuple of (adapter_params, adapted_params) where:
            - adapter_params: Number of parameters in ONE LoRA adapter (lora_A + lora_B)
            - adapted_params: Number of base layer parameters being adapted by LoRA
        """
        adapter_params = self.lora_A[0].numel() + self.lora_B[0].numel()
        adapted_params = self.base_layer.weight.numel()
        return adapter_params, adapted_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., in_features]
        """
        ori_shape = x.shape
        new_shape = ori_shape[:-1] + (self.out_features,)
        x = x.view(-1, x.shape[-1])
        offsets = self._lora_num_tokens.cumsum(dim=0)
        assert offsets[-1] == x.shape[0], f"offsets: {offsets}, x.shape: {x.shape}"

        base_out = self.base_layer(x)
        lora_x = self.lora_dropout(x)

        combined_lora_A = torch.stack([i for i in self.lora_A], dim=0)
        combined_lora_B = torch.stack([i for i in self.lora_B], dim=0)
        if self.use_grouped_mm:
            lora_out = _run_lora_grouped_mm(lora_x, combined_lora_A, combined_lora_B, offsets)
        else:
            lora_out = _run_lora_for_loop(lora_x, combined_lora_A, combined_lora_B, offsets)

        # Apply per-token scaling
        per_token_scaling = torch.repeat_interleave(self._scaling_factors, self._lora_num_tokens).unsqueeze(-1)
        return (base_out + per_token_scaling * lora_out).view(new_shape)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(base={self.base_layer}, rank={self.rank}, "
            f"n_adapters={self.n_adapters}, alpha={self.alpha}, dropout={self.lora_dropout})"
        )
