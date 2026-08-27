from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchao.prototype.moe_training import mxfp8_grouped_mm as tao_mxfp8_gmm
from torchao.prototype.moe_training.config import MXFP8TrainingOpConfig, MXFP8TrainingRecipe
from torchtitan.distributed.expert_parallel import set_token_group_alignment_size_m

from prime_rl.configs.trainer import MXFP8Recipe
from prime_rl.trainer.models.layers.lowprecision import (
    LinearRecipe,
    MoEExpertKernel,
    PreparedActivations,
    PreparedWeights,
)
from prime_rl.trainer.models.layers.mxfp8_grouped_gemm import (
    _MXFP8_TOKEN_GROUP_ALIGN,
    _chunk_dim0_quantize_for_large_inputs,
    _fallback_to_triton_rearrange_for_wide_moes,
)
from prime_rl.trainer.models.layers.mxfp8_linear import MXFP8Linear
from prime_rl.utils.logger import get_logger


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


@dataclass(frozen=True, eq=False)
class Mxfp8PreparedWeights(PreparedWeights):
    w1: Tensor
    w2: Tensor
    w3: Tensor | None


@dataclass(frozen=True, eq=False)
class Mxfp8PreparedActivations(PreparedActivations):
    x: Tensor
    offs: Tensor
    orig_dtype: torch.dtype


class Mxfp8MoEExpertKernel(MoEExpertKernel):
    """MXFP8 MoE kernel using torchao."""
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

    def preprocess_weights(self, w1: Tensor, w2: Tensor, w3: Tensor | None) -> PreparedWeights:
        return Mxfp8PreparedWeights(
            w1=w1.bfloat16().transpose(-2, -1),
            w2=w2.bfloat16().transpose(-2, -1),
            w3=w3.bfloat16().transpose(-2, -1) if w3 is not None else None,
        )

    def preprocess_activations(self, x: Tensor, num_tokens_per_expert: Tensor) -> PreparedActivations:
        offs = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        return Mxfp8PreparedActivations(x=x.bfloat16(), offs=offs, orig_dtype=x.dtype)

    def _grouped_mm(self, x: Tensor, weight: Tensor, offs: Tensor) -> Tensor:
        return tao_mxfp8_gmm._to_mxfp8_then_scaled_grouped_mm(
            x,
            weight,
            offs,
            out_dtype=self.out_dtype,
            kernel_preference=self.kernel_preference,
            wgrad_with_hp=self.wgrad_with_hp,
            scale_calculation_mode=self.scale_calculation_mode,
            pad_token_groups_for_grouped_mm=True,
        )

    def compute(self, weights: PreparedWeights, activations: PreparedActivations) -> Tensor:
        assert isinstance(weights, Mxfp8PreparedWeights)
        assert isinstance(activations, Mxfp8PreparedActivations)
        x, offs = activations.x, activations.offs
        if weights.w3 is not None:
            h = F.silu(self._grouped_mm(x, weights.w1, offs))
            h = h * self._grouped_mm(x, weights.w3, offs)
        else:
            h = self._grouped_mm(x, weights.w1, offs).relu() ** 2
        return self._grouped_mm(h, weights.w2, offs)

    def postprocess_activations(self, out: Tensor, activations: PreparedActivations) -> Tensor:
        assert isinstance(activations, Mxfp8PreparedActivations)
        return out.to(activations.orig_dtype)


def apply_mxfp8_moe_expert_kernel(model: nn.Module, kernel: Mxfp8MoEExpertKernel) -> None:
    from prime_rl.trainer.models.layers.moe import GroupedExperts, NonGatedGroupedExperts

    enabled = 0
    for module in model.modules():
        if isinstance(module, (GroupedExperts, NonGatedGroupedExperts)):
            module.set_moe_expert_kernel(kernel)
            enabled += 1
    get_logger().info(f"Enabled MXFP8 MoE expert kernel (direct path) on {enabled} modules (recipe={kernel.recipe})")
