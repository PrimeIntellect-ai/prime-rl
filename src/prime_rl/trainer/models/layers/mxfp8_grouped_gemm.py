from __future__ import annotations

from torch import nn

from prime_rl.configs.trainer import MXFP8Recipe
from prime_rl.trainer.models.layers.moe import GroupedExperts, NonGatedGroupedExperts
from prime_rl.utils.logger import get_logger

_MXFP8_TOKEN_GROUP_ALIGN: int = 32


def apply_mxfp8_moe_grouped_gemm(model: nn.Module, recipe: MXFP8Recipe) -> None:
    from torchao.prototype.moe_training.config import MXFP8TrainingOpConfig, MXFP8TrainingRecipe
    from torchao.quantization.quant_api import quantize_
    from torchtitan.distributed.expert_parallel import set_token_group_alignment_size_m

    set_token_group_alignment_size_m(_MXFP8_TOKEN_GROUP_ALIGN)
    op_config = MXFP8TrainingOpConfig.from_recipe(MXFP8TrainingRecipe(recipe))

    def filter_fn(module: nn.Module, fqn: str) -> bool:
        return isinstance(module, (GroupedExperts, NonGatedGroupedExperts))

    quantize_(model, config=op_config, filter_fn=filter_fn)
    get_logger().info(
        f"Wrapped MoE expert weights with MXFP8 grouped GEMM (recipe={recipe}, "
        f"token_group_align={_MXFP8_TOKEN_GROUP_ALIGN})"
    )
