from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING

import torch

from prime_rl.trainer.models.layers.recipe import GEMMRecipe, RecipeData

if TYPE_CHECKING:
    from prime_rl.trainer.models.layers.moe import GroupedExperts


class GroupedGemmRecipe(GEMMRecipe["GroupedExperts"]):
    token_group_alignment: int

    @abstractmethod
    def __call__(
        self,
        x: torch.Tensor,
        weight_t: torch.Tensor,
        *,
        offs: torch.Tensor,
    ) -> torch.Tensor: ...

    def quantize(self, mod: "GroupedExperts") -> tuple["GroupedExperts", RecipeData | None]:
        mod.set_grouped_gemm(self)
        return mod, None


@dataclass(frozen=True)
class BF16GroupedGemmRecipe(GroupedGemmRecipe):
    name = "bf16"
    impl = "torch"
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
class DeepGemmFP8GroupedGemmRecipe(GroupedGemmRecipe):
    name = "fp8_blockwise"
    impl = "deepgemm"
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
class MXFP8GroupedGemmData(RecipeData):
    recipe: str
    high_precision_wgrad: bool


@dataclass(frozen=True)
class MXFP8GroupedGemmRecipe(GroupedGemmRecipe):
    name = "mxfp8"
    impl = "custom"

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

    def quantize(self, mod: "GroupedExperts") -> tuple["GroupedExperts", RecipeData | None]:
        mod.set_grouped_gemm(self)
        recipe_name = "mxfp8_rceil_wgrad_with_hp" if self.high_precision_wgrad else "mxfp8_rceil"
        return mod, MXFP8GroupedGemmData(recipe=recipe_name, high_precision_wgrad=self.high_precision_wgrad)
