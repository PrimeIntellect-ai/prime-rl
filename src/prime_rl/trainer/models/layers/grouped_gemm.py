from dataclasses import dataclass
from types import ModuleType
from typing import Protocol

import torch


class GroupedGemm(Protocol):
    token_group_alignment: int

    def __call__(
        self,
        x: torch.Tensor,
        weight_t: torch.Tensor,
        *,
        offs: torch.Tensor,
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class BF16GroupedGemm:
    token_group_alignment: int = 8

    def __call__(
        self,
        x: torch.Tensor,
        weight_t: torch.Tensor,
        *,
        offs: torch.Tensor,
    ) -> torch.Tensor:
        return torch._grouped_mm(x, weight_t, offs=offs)


@dataclass(frozen=True)
class DeepGemmFP8GroupedGemm:
    token_group_alignment: int = 8

    def __call__(
        self,
        x: torch.Tensor,
        weight_t: torch.Tensor,
        *,
        offs: torch.Tensor,
    ) -> torch.Tensor:
        from prime_rl.trainer.models.layers.fp8_grouped_gemm import grouped_fp8_gemm

        return grouped_fp8_gemm(x, weight_t, offs)


@dataclass(frozen=True)
class MXFP8GroupedGemm:
    kernel: ModuleType
    high_precision_wgrad: bool
    token_group_alignment: int

    def __call__(
        self,
        x: torch.Tensor,
        weight_t: torch.Tensor,
        *,
        offs: torch.Tensor,
    ) -> torch.Tensor:
        return self.kernel.grouped_gemm(
            x,
            weight_t,
            offs,
            high_precision_wgrad=self.high_precision_wgrad,
        )
