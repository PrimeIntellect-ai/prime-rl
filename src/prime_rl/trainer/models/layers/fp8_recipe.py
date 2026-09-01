from __future__ import annotations

from torch import nn

from prime_rl.trainer.models.layers.fp8_linear import Float8BlockwiseLinear
from prime_rl.trainer.models.layers.lowprecision import LinearRecipe


class Fp8LinearRecipe(LinearRecipe):
    name = "fp8_blockwise"

    def linear_cls(self) -> type[nn.Linear]:
        return Float8BlockwiseLinear

    def is_linear_shape_supported(self, in_features: int, out_features: int) -> bool:
        return in_features % 128 == 0 and out_features % 128 == 0

    def convert_linear(self, mod: nn.Linear) -> nn.Linear:
        return Float8BlockwiseLinear.from_linear(mod)
