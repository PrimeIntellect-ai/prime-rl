from __future__ import annotations

from torch import nn
from torchtitan.distributed.expert_parallel import set_token_group_alignment_size_m

from prime_rl.configs.trainer import NVFP4Backward
from prime_rl.trainer.models.layers.moe import GroupedExperts
from prime_rl.utils.logger import get_logger

_NVFP4_TOKEN_GROUP_ALIGN = 32


def apply_nvfp4_moe_grouped_gemm(
    model: nn.Module,
    backward: NVFP4Backward = "bf16_dequantized",
) -> None:
    """Select the NVFP4 grouped path for supported routed-expert modules."""

    if backward not in ("bf16", "bf16_dequantized"):
        raise ValueError("backward must be 'bf16' or 'bf16_dequantized'")

    from prime_rl_kernels.nvfp4 import prepare_for_compile

    prepare_for_compile()
    set_token_group_alignment_size_m(_NVFP4_TOKEN_GROUP_ALIGN)
    replaced = 0
    for module in model.modules():
        if isinstance(module, GroupedExperts):
            module.nvfp4 = True
            module.nvfp4_backward = backward
            replaced += 1

    if replaced == 0:
        raise ValueError("NVFP4 grouped GEMM was requested, but the model has no supported GroupedExperts modules")
    get_logger().info(
        f"Enabled NVFP4 grouped GEMM for {replaced} routed-expert modules "
        f"(token_group_align={_NVFP4_TOKEN_GROUP_ALIGN}, backward={backward})"
    )
