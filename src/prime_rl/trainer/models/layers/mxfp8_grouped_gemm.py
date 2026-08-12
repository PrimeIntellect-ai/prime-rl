from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import nn
from torchao.prototype.moe_training import conversion_utils as cu
from torchao.prototype.moe_training import mxfp8_grouped_mm as tao_mxfp8_gmm
from torchao.prototype.moe_training.config import MXFP8TrainingOpConfig, MXFP8TrainingRecipe
from torchao.prototype.moe_training.kernels.mxfp8 import quant as tao_mxfp8_quant
from torchao.prototype.moe_training.kernels.mxfp8 import triton_mx_block_rearrange_2d_M_groups
from torchao.prototype.moe_training.tensor import MXFP8TrainingWeightWrapperTensor
from torchao.quantization.quant_api import quantize_
from torchtitan.distributed.expert_parallel import set_token_group_alignment_size_m
from torchtitan.experiments.kernels.moe import indices as tt_indices

from prime_rl.configs.trainer import MXFP8Recipe
from prime_rl.trainer.models.layers.moe import GroupedExperts, NonGatedGroupedExperts
from prime_rl.utils.logger import get_logger

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


@triton.jit
def _start_index_after_padding_pow2(
    group_pid,
    orig_offsets,
    num_groups: tl.constexpr,
    padding_size: tl.constexpr,
):
    """Prefix sum to compute the start index of a given group. Unlike the torchao original,
    this rounds the arange up to a power of 2 and masks, so non-power-of-2 group counts work."""
    NUM_GROUPS_POW2: tl.constexpr = triton.next_power_of_2(num_groups)
    idx = tl.arange(0, NUM_GROUPS_POW2)
    valid = idx < num_groups
    offsets = tl.load(orig_offsets + idx, mask=valid, other=0)
    prev_offsets = tl.load(orig_offsets + idx - 1, mask=valid & (idx > 0), other=0)
    group_sizes = tl.where(idx > 0, offsets - prev_offsets, offsets)
    padded_sizes = tl.cdiv(group_sizes, padding_size) * padding_size
    prefix_mask = valid & (idx < group_pid)
    group_start_idx = tl.sum(tl.where(prefix_mask, padded_sizes, 0))
    return group_start_idx


def _support_non_pow2_expert_groups() -> None:
    """torchao's scale-swizzle kernels compute per-group prefix sums with
    ``tl.arange(0, num_groups)``, which triton rejects when the local expert count is not a
    power of 2 (e.g. GLM-4.5: 160 experts / EP=8 = 20 per rank). The swizzle kernels resolve
    the helper through their module globals at first launch, so replacing it is enough."""
    if getattr(tao_mxfp8_quant._start_index_after_padding, "_prime_rl_non_pow2", False):
        return
    _start_index_after_padding_pow2._prime_rl_non_pow2 = True
    tao_mxfp8_quant._start_index_after_padding = _start_index_after_padding_pow2


def apply_mxfp8_moe_grouped_gemm(model: nn.Module, recipe: MXFP8Recipe) -> None:
    _relax_torchao_mxfp8_version_gate()
    _align_permute_indices_buffer()
    _fallback_to_triton_rearrange_for_wide_moes()
    _support_non_pow2_expert_groups()
    set_token_group_alignment_size_m(_MXFP8_TOKEN_GROUP_ALIGN)
    op_config = MXFP8TrainingOpConfig.from_recipe(MXFP8TrainingRecipe(recipe))

    def filter_fn(module: nn.Module, fqn: str) -> bool:
        return isinstance(module, (GroupedExperts, NonGatedGroupedExperts))

    quantize_(model, config=op_config, filter_fn=filter_fn)
    get_logger().info(
        f"Wrapped MoE expert weights with MXFP8 grouped GEMM (recipe={recipe}, "
        f"token_group_align={_MXFP8_TOKEN_GROUP_ALIGN})"
    )
