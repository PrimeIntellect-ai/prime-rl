from __future__ import annotations

import torch
from torch import nn

from prime_rl.configs.trainer import MXFP8Recipe
from prime_rl.trainer.models.layers.moe import GroupedExperts, NonGatedGroupedExperts
from prime_rl.utils.logger import get_logger
from torchao.prototype.moe_training import conversion_utils as cu
from torchao.prototype.moe_training import mxfp8_grouped_mm as tao_mxfp8_gmm
from torchao.prototype.moe_training.config import MXFP8TrainingOpConfig, MXFP8TrainingRecipe
from torchao.prototype.moe_training.kernels.mxfp8 import triton_mx_block_rearrange_2d_M_groups
from torchao.prototype.moe_training.tensor import MXFP8TrainingWeightWrapperTensor
from torchtitan.experiments.kernels.moe import indices as tt_indices
from torchao.quantization.quant_api import quantize_
from torchtitan.distributed.expert_parallel import set_token_group_alignment_size_m

_MXFP8_TOKEN_GROUP_ALIGN: int = 32
# torchao CUDA scale rearrange kernel supports at most 32 token groups,
# the Triton variant has no such cap and produces the same swizzled layout, so fall back to it for wider MoEs
_CUDA_REARRANGE_MAX_GROUPS: int = 32

def _fallback_to_triton_rearrange_for_wide_moes() -> None:
    if getattr(tao_mxfp8_gmm.mx_block_rearrange_2d_M_groups_cuda, "_prime_rl_wide_moe", False):
        return
    original = tao_mxfp8_gmm.mx_block_rearrange_2d_M_groups_cuda

    def mx_block_rearrange_2d_M_groups(
        scales_tensor: torch.Tensor, input_group_end_offsets: torch.Tensor, chunks_per_tb: int = 4
    ) -> torch.Tensor:
        if input_group_end_offsets.shape[0] > _CUDA_REARRANGE_MAX_GROUPS:
            return triton_mx_block_rearrange_2d_M_groups(scales_tensor, input_group_end_offsets)
        return original(scales_tensor, input_group_end_offsets, chunks_per_tb)

    mx_block_rearrange_2d_M_groups._prime_rl_wide_moe = True
    tao_mxfp8_gmm.mx_block_rearrange_2d_M_groups_cuda = mx_block_rearrange_2d_M_groups


def _relax_torchao_mxfp8_version_gate() -> None:
    if getattr(cu._get_tensor_cls_for_config, "_prime_rl_relaxed", False):
        return
    original = cu._get_tensor_cls_for_config

    def _get_tensor_cls_for_config(config):
        if isinstance(config, MXFP8TrainingOpConfig):
            return MXFP8TrainingWeightWrapperTensor
        return original(config)

    _get_tensor_cls_for_config._prime_rl_relaxed = True
    cu._get_tensor_cls_for_config = _get_tensor_cls_for_config


def _align_permute_indices_buffer() -> None:
    """
    Align the permuted index buffer to a multiple of alignment as the kernel requires rows % 32 = 0 and cols % 32 = 0.
    """
    if getattr(tt_indices.generate_permute_indices, "_prime_rl_aligned", False):
        return
    original = tt_indices.generate_permute_indices

    def generate_permute_indices(
        tokens_per_expert_group, experts_per_rank, num_ranks, max_len, alignment, use_cpu=False
    ):
        al_mask = alignment - 1
        assert alignment > 0 and (alignment & al_mask) == 0
        max_len = (max_len + al_mask) & ~al_mask
        return original(tokens_per_expert_group, experts_per_rank, num_ranks, max_len, alignment, use_cpu=use_cpu)

    generate_permute_indices._prime_rl_aligned = True
    tt_indices.generate_permute_indices = generate_permute_indices


def apply_mxfp8_moe_grouped_gemm(model: nn.Module, recipe: MXFP8Recipe) -> None:
    _relax_torchao_mxfp8_version_gate()
    _align_permute_indices_buffer()
    _fallback_to_triton_rearrange_for_wide_moes()
    set_token_group_alignment_size_m(_MXFP8_TOKEN_GROUP_ALIGN)
    op_config = MXFP8TrainingOpConfig.from_recipe(MXFP8TrainingRecipe(recipe))

    def filter_fn(module: nn.Module, fqn: str) -> bool:
        return isinstance(module, (GroupedExperts, NonGatedGroupedExperts))

    quantize_(model, config=op_config, filter_fn=filter_fn)
    get_logger().info(
        f"Wrapped MoE expert weights with MXFP8 grouped GEMM (recipe={recipe}, "
        f"token_group_align={_MXFP8_TOKEN_GROUP_ALIGN})"
    )
