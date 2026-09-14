from __future__ import annotations

from dataclasses import dataclass

from torch import nn
from torchao.prototype.moe_training.config import MXFP8TrainingOpConfig, MXFP8TrainingRecipe
from torchao.prototype.mx_formats import ScaleCalculationMode
from torchao.quantization.quantize_.common import KernelPreference

from prime_rl.configs.trainer import MXFP8Recipe
from prime_rl.trainer.models.layers.lowprecision import LinearRecipe
from prime_rl.trainer.models.layers.mxfp8_linear import MXFP8Linear
from prime_rl.trainer.models.layers.recipe import RecipeData


@dataclass(frozen=True)
class Mxfp8LinearData(RecipeData):
    recipe: MXFP8Recipe
    kernel_preference: KernelPreference
    wgrad_with_hp: bool
    scale_calculation_mode: ScaleCalculationMode


class Mxfp8LinearRecipe(LinearRecipe):
    """MXFP8 linear layer."""

    name = "mxfp8"
    impl = "torchao"

    def __init__(self, recipe: MXFP8Recipe = "mxfp8_rceil") -> None:
        op_config = MXFP8TrainingOpConfig.from_recipe(MXFP8TrainingRecipe(recipe))
        self.data = Mxfp8LinearData(
            recipe=recipe,
            kernel_preference=op_config.kernel_preference,
            wgrad_with_hp=op_config.wgrad_with_hp,
            scale_calculation_mode=op_config.scale_calculation_mode,
        )

    def is_shape_supported(self, in_features: int, out_features: int) -> bool:
        return in_features % 32 == 0 and out_features % 32 == 0

    def quantize(self, mod: nn.Linear) -> tuple[nn.Linear, Mxfp8LinearData]:
        return (
            MXFP8Linear.from_linear(
                mod,
                kernel_preference=self.data.kernel_preference,
                wgrad_with_hp=self.data.wgrad_with_hp,
                scale_calculation_mode=self.data.scale_calculation_mode,
            ),
            self.data,
        )
