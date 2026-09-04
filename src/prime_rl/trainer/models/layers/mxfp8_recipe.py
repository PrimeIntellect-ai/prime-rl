from __future__ import annotations

from torch import nn
from torchao.prototype.moe_training.config import MXFP8TrainingOpConfig, MXFP8TrainingRecipe

from prime_rl.configs.trainer import MXFP8Recipe
from prime_rl.trainer.models.layers.lowprecision import LinearRecipe
from prime_rl.trainer.models.layers.mxfp8_linear import MXFP8Linear


class Mxfp8LinearRecipe(LinearRecipe):
    """MXFP8 linear layer."""

    name = "mxfp8"

    def __init__(self, recipe: MXFP8Recipe = "mxfp8_rceil") -> None:
        self.recipe = recipe
        op_config = MXFP8TrainingOpConfig.from_recipe(MXFP8TrainingRecipe(recipe))
        self.kernel_preference = op_config.kernel_preference
        self.wgrad_with_hp = op_config.wgrad_with_hp
        self.scale_calculation_mode = op_config.scale_calculation_mode

    def linear_cls(self) -> type[nn.Linear]:
        return MXFP8Linear

    def is_linear_shape_supported(self, in_features: int, out_features: int) -> bool:
        return in_features % 32 == 0 and out_features % 32 == 0

    def convert_linear(self, mod: nn.Linear) -> nn.Linear:
        return MXFP8Linear.from_linear(
            mod,
            kernel_preference=self.kernel_preference,
            wgrad_with_hp=self.wgrad_with_hp,
            scale_calculation_mode=self.scale_calculation_mode,
        )
