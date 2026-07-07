from __future__ import annotations

from torch import nn

from prime_rl.configs.trainer import MXFP8Recipe
from prime_rl.trainer.models.layers.moe import GroupedExperts, NonGatedGroupedExperts
from prime_rl.utils.logger import get_logger

_MXFP8_TOKEN_GROUP_ALIGN: int = 32


def _relax_torchao_mxfp8_version_gate() -> None:
    """Relax torchao's over-strict torch-nightly assertion for the MXFP8 grouped GEMM.

    torchao's ``_get_tensor_cls_for_config`` refuses the MXFP8 training op unless two
    post-2.11.0 DTensor symbols (``_ops.scaled_mm_single_dim_strategy`` and
    ``_dispatch.is_pinned_handler``) are present, otherwise raising "install the latest
    torch nightly". This is a false gate for us: the grouped GEMM runs full forward and
    backward on our pinned torch (2.11) — current torch already ships the equivalent
    non-single-dim ``scaled_mm_strategy``. Replace the check with a direct return of the
    weight-wrapper class; only the MXFP8 branch is intercepted, everything else defers to
    the original implementation.
    """
    from torchao.prototype.moe_training import conversion_utils as cu
    from torchao.prototype.moe_training.config import MXFP8TrainingOpConfig
    from torchao.prototype.moe_training.tensor import MXFP8TrainingWeightWrapperTensor

    if getattr(cu._get_tensor_cls_for_config, "_prime_rl_relaxed", False):
        return

    original = cu._get_tensor_cls_for_config

    def _get_tensor_cls_for_config(config):
        if isinstance(config, MXFP8TrainingOpConfig):
            return MXFP8TrainingWeightWrapperTensor
        return original(config)

    _get_tensor_cls_for_config._prime_rl_relaxed = True
    cu._get_tensor_cls_for_config = _get_tensor_cls_for_config


def apply_mxfp8_moe_grouped_gemm(model: nn.Module, recipe: MXFP8Recipe) -> None:
    from torchao.prototype.moe_training.config import MXFP8TrainingOpConfig, MXFP8TrainingRecipe
    from torchao.quantization.quant_api import quantize_
    from torchtitan.distributed.expert_parallel import set_token_group_alignment_size_m

    _relax_torchao_mxfp8_version_gate()
    set_token_group_alignment_size_m(_MXFP8_TOKEN_GROUP_ALIGN)
    op_config = MXFP8TrainingOpConfig.from_recipe(MXFP8TrainingRecipe(recipe))

    def filter_fn(module: nn.Module, fqn: str) -> bool:
        return isinstance(module, (GroupedExperts, NonGatedGroupedExperts))

    quantize_(model, config=op_config, filter_fn=filter_fn)
    get_logger().info(
        f"Wrapped MoE expert weights with MXFP8 grouped GEMM (recipe={recipe}, "
        f"token_group_align={_MXFP8_TOKEN_GROUP_ALIGN})"
    )
