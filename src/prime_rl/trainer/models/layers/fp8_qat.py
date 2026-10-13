"""FP8 forward matmuls with a BF16 straight-through backward surrogate."""

import re

import torch
from torch import nn

from prime_rl.trainer.models.kernels.fp8_qat import cast_qat_weight
from prime_rl.trainer.models.kernels.fp8_utils import (
    GROUP_ALIGNMENT,
    build_grouped_layout,
    grouped_per_token_cast_to_fp8_triton,
    per_token_cast_to_fp8_triton,
    unpack_rows_triton,
)
from prime_rl.trainer.models.layers.expert_compute import DeepGemmFP8ExpertCompute
from prime_rl.trainer.models.layers.fp8_linear import Float8BlockwiseLinear
from prime_rl.trainer.models.layers.grouped_linear import GroupedLinear
from prime_rl.utils.logger import get_logger


@torch.library.custom_op("prime_rl::fp8_qat_linear", mutates_args=())
def fp8_qat_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    groups: int,
    rounding_seed: torch.Tensor | None = None,
    rounding_salt: torch.Tensor | None = None,
    cos_sin_cache: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    import deep_gemm

    # Pinned vLLM defaults to DeepGEMM UE8M0 on both Hopper and Blackwell.
    if position_ids is not None:
        from prime_rl.trainer.models.kernels.deepseek_v4.dsv4_rope import inverse_rope_cast_to_fp8

        x_fp8, x_scale = inverse_rope_cast_to_fp8(x, cos_sin_cache, position_ids, groups)
        output_prefix = (*x.shape[:-2], groups)
    else:
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        x_fp8, x_scale = per_token_cast_to_fp8_triton(x_2d, use_ue8m0=True, scale_floor=1e-10 if groups == 1 else 0.0)
        output_prefix = x.shape[:-1]
    w_fp8, w_scale = cast_qat_weight(weight, True, rounding_seed=rounding_seed, rounding_salt=rounding_salt)
    out_features = weight.shape[0] // groups
    out = x.new_empty((*output_prefix, out_features))
    if groups == 1:
        deep_gemm.fp8_gemm_nt(
            (x_fp8.view(-1, weight.shape[-1]), x_scale.view(-1, weight.shape[-1] // 128)),
            (w_fp8, w_scale),
            out.view(-1, out_features),
        )
    else:
        deep_gemm.fp8_einsum(
            "bhr,hdr->bhd",
            (x_fp8.view(-1, groups, weight.shape[-1]), x_scale.view(-1, groups, weight.shape[-1] // 128)),
            (w_fp8.view(groups, out_features, -1), w_scale.view(groups, -1, w_scale.shape[-1])),
            out.view(-1, groups, out_features),
            recipe=(1, 128, 128),
        )
    return out


@fp8_qat_linear.register_fake
def _linear_fake(x, weight, groups, rounding_seed=None, rounding_salt=None, cos_sin_cache=None, position_ids=None):
    prefix = (*x.shape[:-2], groups) if position_ids is not None else x.shape[:-1]
    return x.new_empty((*prefix, weight.shape[0] // groups))


def _linear_setup_context(ctx, inputs, output):
    x, weight, ctx.groups, rounding_seed, rounding_salt, cos_sin_cache, position_ids = inputs
    ctx.save_for_backward(x, weight, rounding_seed, rounding_salt, cos_sin_cache, position_ids)


def _linear_backward(ctx, grad):
    x, weight, rounding_seed, rounding_salt, cos_sin_cache, position_ids = ctx.saved_tensors
    original_shape = x.shape
    if position_ids is not None:
        from prime_rl.trainer.models.kernels.deepseek_v4.dsv4_rope import dsv4_rope

        x = dsv4_rope(x, cos_sin_cache, position_ids, inverse=True).view(*x.shape[:-2], ctx.groups, weight.shape[-1])
    rounded_weight = (
        _rounded_weight(weight.detach(), rounding_seed, rounding_salt, 0) if ctx.needs_input_grad[0] else None
    )
    groups = ctx.groups
    if groups == 1:
        dy = grad.reshape(-1, grad.shape[-1])
        dx = (dy @ rounded_weight).view_as(x) if ctx.needs_input_grad[0] else None
        dw = dy.T @ x.reshape(-1, x.shape[-1]) if ctx.needs_input_grad[1] else None
    else:
        dy = grad.reshape(-1, groups, grad.shape[-1]).transpose(0, 1)
        dx = (
            torch.bmm(dy, rounded_weight.view(groups, -1, x.shape[-1])).transpose(0, 1).reshape_as(x)
            if ctx.needs_input_grad[0]
            else None
        )
        x_grouped = x.reshape(-1, groups, x.shape[-1]).transpose(0, 1)
        dw = torch.bmm(dy.transpose(1, 2), x_grouped).reshape_as(weight) if ctx.needs_input_grad[1] else None
    if position_ids is not None and dx is not None:
        dx = dsv4_rope(dx.reshape(original_shape), cos_sin_cache, position_ids)
    return dx, dw, None, None, None, None, None


fp8_qat_linear.register_autograd(_linear_backward, setup_context=_linear_setup_context)


@torch.library.custom_op("prime_rl::fp8_qat_grouped_mm", mutates_args=())
def fp8_qat_grouped_mm(
    x: torch.Tensor,
    weight: torch.Tensor,
    offs: torch.Tensor,
    rounding_seed: torch.Tensor | None = None,
    rounding_salt: torch.Tensor | None = None,
    rounding_offset: int = 0,
) -> torch.Tensor:
    import deep_gemm

    total_m, padded_m, layout, blocks, _, starts, lengths, block_starts = build_grouped_layout(offs, total_m=x.shape[0])
    x_fp8 = grouped_per_token_cast_to_fp8_triton(
        x, padded_m, blocks, starts, lengths, block_starts, True, GROUP_ALIGNMENT, scale_floor=1e-10
    )
    w_fp8 = cast_qat_weight(
        weight.transpose(1, 2),
        True,
        rounding_seed=rounding_seed,
        rounding_salt=rounding_salt,
        rounding_offset=rounding_offset,
    )
    out = x.new_empty((padded_m, weight.shape[-1]))
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(x_fp8, w_fp8, out, layout, use_psum_layout=False)
    return unpack_rows_triton(out, total_m, blocks, starts, lengths, block_starts)


@fp8_qat_grouped_mm.register_fake
def _grouped_fake(x, weight, offs, rounding_seed=None, rounding_salt=None, rounding_offset=0):
    return x.new_empty((x.shape[0], weight.shape[-1]))


def _grouped_setup_context(ctx, inputs, output):
    x, weight, offs, rounding_seed, rounding_salt, ctx.rounding_offset = inputs
    ctx.save_for_backward(x, weight, offs, rounding_seed, rounding_salt)


def _grouped_backward(ctx, grad):
    x, weight, offs, rounding_seed, rounding_salt = ctx.saved_tensors
    grad = grad.contiguous()
    dx = None
    if ctx.needs_input_grad[0]:
        rounded_weight = _rounded_weight(
            weight.detach().transpose(1, 2), rounding_seed, rounding_salt, ctx.rounding_offset
        )
        dx = torch._grouped_mm(grad, rounded_weight, offs=offs)
        del rounded_weight
    dw = torch._grouped_mm(x.T, grad, offs=offs) if ctx.needs_input_grad[1] else None
    return dx, dw, None, None, None, None


fp8_qat_grouped_mm.register_autograd(_grouped_backward, setup_context=_grouped_setup_context)


@torch.library.custom_op("prime_rl::fp8_qat_rounded_weight", mutates_args=())
def _rounded_weight(
    weight: torch.Tensor,
    rounding_seed: torch.Tensor | None,
    rounding_salt: torch.Tensor | None,
    rounding_offset: int,
) -> torch.Tensor:
    return cast_qat_weight(
        weight,
        True,
        rounding_seed=rounding_seed,
        rounding_salt=rounding_salt,
        rounding_offset=rounding_offset,
        dequantize=True,
    )[0]


@_rounded_weight.register_fake
def _rounded_weight_fake(weight, rounding_seed, rounding_salt, rounding_offset):
    return weight.new_empty(weight.shape, dtype=torch.bfloat16)


class FP8QATLinear(Float8BlockwiseLinear):
    """Bias-free dense or grouped FP8 projection with BF16 backward GEMMs."""

    def __init__(self, *args, groups: int = 1, stochastic_rounding: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.groups = groups
        self.stochastic_rounding = stochastic_rounding

    def forward(self, x, cos_sin_cache=None, position_ids=None):
        return fp8_qat_linear(
            x,
            self.weight,
            self.groups,
            self.fp8_rounding.seed if self.fp8_rounding is not None else None,
            self.rounding_salt,
            cos_sin_cache,
            position_ids,
        )


def replace_linear_with_fp8_qat(model: nn.Module, ignore_patterns: list[str], stochastic_rounding: bool) -> None:
    replaced = []
    replacements = {}
    for name, module in list(model.named_modules(remove_duplicate=False)):
        if not isinstance(module, nn.Linear) or any(re.search(pattern, name) for pattern in ignore_patterns):
            continue
        groups = module.n_groups if isinstance(module, GroupedLinear) else 1
        if module.bias is not None or module.in_features % 128 or (module.out_features // groups) % 128:
            continue
        with torch.device("meta"):
            replacement = FP8QATLinear(
                module.in_features,
                module.out_features,
                bias=False,
                groups=groups,
                stochastic_rounding=stochastic_rounding,
            )
        replacement = replacements.setdefault(id(module), replacement)
        replacement.weight = module.weight
        replacement.train(module.training)
        parent_name, attr = name.rsplit(".", 1) if "." in name else ("", name)
        setattr(model.get_submodule(parent_name), attr, replacement)
        replaced.append(name)
    get_logger().info(f"Enabled FP8 QAT on {len(replaced)} projections (stochastic_rounding={stochastic_rounding})")


class FP8QATExpertCompute(DeepGemmFP8ExpertCompute):
    def _gemm(self, experts, name, x, weight, offsets):
        return fp8_qat_grouped_mm(
            x,
            weight,
            offsets,
            experts.fp8_rounding.seed if self.stochastic_rounding else None,
            experts.rounding_salts[name] if self.stochastic_rounding else None,
            experts.fp8_expert_offset * weight.shape[1] * weight.shape[2] if self.stochastic_rounding else 0,
        )
