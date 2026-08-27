from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn
from torchao.prototype.moe_training import mxfp8_grouped_mm as tao_mxfp8_gmm
from torchao.prototype.moe_training.config import MXFP8TrainingOpConfig, MXFP8TrainingRecipe
from torchtitan.distributed.expert_parallel import set_token_group_alignment_size_m

from prime_rl.configs.trainer import MXFP8Recipe
from prime_rl.trainer.models.layers.lowprecision import (
    GroupedGemmRecipe,
    GroupedLayout,
    LinearRecipe,
    QuantizedActivation,
)
from prime_rl.trainer.models.layers.mxfp8_grouped_gemm import (
    _MXFP8_TOKEN_GROUP_ALIGN,
    _chunk_dim0_quantize_for_large_inputs,
    _fallback_to_triton_rearrange_for_wide_moes,
)
from prime_rl.trainer.models.layers.mxfp8_linear import MXFP8Linear
from prime_rl.utils.logger import get_logger


class Mxfp8LinearRecipe(LinearRecipe):
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


@dataclass(frozen=True, eq=False)
class Mxfp8GroupedLayout(GroupedLayout):
    offs: Tensor
    total_m: int


@dataclass(frozen=True, eq=False)
class Mxfp8QuantizedActivation(QuantizedActivation):
    x: Tensor


class Mxfp8GroupedGemmRecipe(GroupedGemmRecipe):
    name = "mxfp8"

    def __init__(self, recipe: MXFP8Recipe = "mxfp8_rceil") -> None:
        self.recipe = recipe
        op_config = MXFP8TrainingOpConfig.from_recipe(MXFP8TrainingRecipe(recipe))
        self.kernel_preference = op_config.kernel_preference
        self.wgrad_with_hp = op_config.wgrad_with_hp
        self.scale_calculation_mode = op_config.scale_calculation_mode
        self.out_dtype = op_config.out_dtype
        _fallback_to_triton_rearrange_for_wide_moes()
        _chunk_dim0_quantize_for_large_inputs()
        set_token_group_alignment_size_m(_MXFP8_TOKEN_GROUP_ALIGN)

    def build_grouped_layout(self, offs: Tensor, total_m: int) -> GroupedLayout:
        return Mxfp8GroupedLayout(offs=offs, total_m=total_m)

    def quantize_grouped_activation(self, x: Tensor, layout: GroupedLayout) -> QuantizedActivation:
        assert isinstance(layout, Mxfp8GroupedLayout)
        return Mxfp8QuantizedActivation(x=x)

    def grouped_gemm(
        self,
        x: Tensor,
        weight: Tensor,
        layout: GroupedLayout,
        x_q: QuantizedActivation | None = None,
    ) -> Tensor:
        assert isinstance(layout, Mxfp8GroupedLayout)
        return tao_mxfp8_gmm._to_mxfp8_then_scaled_grouped_mm(
            x,
            weight,
            layout.offs,
            out_dtype=self.out_dtype,
            kernel_preference=self.kernel_preference,
            wgrad_with_hp=self.wgrad_with_hp,
            scale_calculation_mode=self.scale_calculation_mode,
            pad_token_groups_for_grouped_mm=True,
        )


_ACTIVE_MXFP8_GROUPED_GEMM_RECIPE: Mxfp8GroupedGemmRecipe | None = None


def get_active_mxfp8_grouped_gemm_recipe() -> Mxfp8GroupedGemmRecipe:
    assert _ACTIVE_MXFP8_GROUPED_GEMM_RECIPE is not None, (
        "No active Mxfp8GroupedGemmRecipe — call apply_mxfp8_grouped_gemm_recipe(model, recipe) first."
    )
    return _ACTIVE_MXFP8_GROUPED_GEMM_RECIPE


def apply_mxfp8_grouped_gemm_recipe(model: nn.Module, recipe: Mxfp8GroupedGemmRecipe) -> None:
    global _ACTIVE_MXFP8_GROUPED_GEMM_RECIPE
    _ACTIVE_MXFP8_GROUPED_GEMM_RECIPE = recipe

    from prime_rl.trainer.models.layers.moe import GroupedExperts, NonGatedGroupedExperts

    enabled = 0
    for module in model.modules():
        if isinstance(module, (GroupedExperts, NonGatedGroupedExperts)):
            module.set_mxfp8_grouped_gemm(True)
            enabled += 1
    get_logger().info(f"Enabled MXFP8 grouped GEMM (direct recipe path) on {enabled} modules (recipe={recipe.recipe})")
