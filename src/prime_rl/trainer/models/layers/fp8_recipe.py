from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from prime_rl.trainer.models.layers.fp8_linear import Float8BlockwiseLinear
from prime_rl.trainer.models.layers.lowprecision import LinearRecipe
from prime_rl.trainer.models.layers.recipe import RecipeData


@dataclass(frozen=True)
class Fp8LinearData(RecipeData):
    block_size: int = 128


class Fp8LinearRecipe(LinearRecipe):
    name = "fp8_blockwise"
    impl = "deepgemm"

    def is_shape_supported(self, in_features: int, out_features: int) -> bool:
        return in_features % 128 == 0 and out_features % 128 == 0

    def quantize(self, mod: nn.Linear) -> tuple[nn.Linear, Fp8LinearData]:
        return Float8BlockwiseLinear.from_linear(mod), Fp8LinearData()
