"""FP8 forward matmuls with a BF16 straight-through backward surrogate."""

import hashlib
import re

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from prime_rl.trainer.models.kernels.fp8_qat import cast_qat_weight
from prime_rl.trainer.models.kernels.fp8_utils import (
    GROUP_ALIGNMENT,
    build_grouped_layout,
    grouped_per_token_cast_to_fp8_triton,
    per_token_cast_to_fp8_triton,
    unpack_rows_triton,
)
from prime_rl.trainer.models.layers.expert_compute import GroupedGemmExpertCompute
from prime_rl.trainer.models.layers.grouped_linear import GroupedLinear
from prime_rl.utils.logger import get_logger


@torch.library.custom_op("prime_rl::fp8_qat_linear", mutates_args=())
def fp8_qat_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    groups: int,
    stochastic_rounding: bool,
    step: torch.Tensor,
    seed: int,
    weight_key: torch.Tensor,
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
    w_fp8, w_scale = cast_qat_weight(weight, True, stochastic_rounding, step=step, seed=seed, weight_key=weight_key)
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
def _linear_fake(x, weight, groups, stochastic_rounding, step, seed, weight_key, cos_sin_cache=None, position_ids=None):
    prefix = (*x.shape[:-2], groups) if position_ids is not None else x.shape[:-1]
    return x.new_empty((*prefix, weight.shape[0] // groups))


def _linear_setup_context(ctx, inputs, output):
    x, weight, ctx.groups, ctx.sr, step, ctx.seed, weight_key, cos_sin_cache, position_ids = inputs
    ctx.save_for_backward(x, weight, step.clone(), weight_key, cos_sin_cache, position_ids)


def _linear_backward(ctx, grad):
    x, weight, step, weight_key, cos_sin_cache, position_ids = ctx.saved_tensors
    original_shape = x.shape
    if position_ids is not None:
        from prime_rl.trainer.models.kernels.deepseek_v4.dsv4_rope import dsv4_rope

        x = dsv4_rope(x, cos_sin_cache, position_ids, inverse=True).view(*x.shape[:-2], ctx.groups, weight.shape[-1])
    rounded_weight = _rounded_weight(weight.detach(), step, ctx.seed, weight_key, 0, ctx.sr)
    groups = ctx.groups
    if groups == 1:
        dy = grad.reshape(-1, grad.shape[-1])
        dx = (dy @ rounded_weight).view_as(x) if ctx.needs_input_grad[0] else None
        dw = dy.T @ x.reshape(-1, x.shape[-1]) if ctx.needs_input_grad[1] else None
    else:
        dy = grad.reshape(-1, groups, grad.shape[-1]).transpose(0, 1)
        w = rounded_weight.view(groups, -1, x.shape[-1])
        dx = torch.bmm(dy, w).transpose(0, 1).reshape_as(x) if ctx.needs_input_grad[0] else None
        x_grouped = x.reshape(-1, groups, x.shape[-1]).transpose(0, 1)
        dw = torch.bmm(dy.transpose(1, 2), x_grouped).reshape_as(weight) if ctx.needs_input_grad[1] else None
    if position_ids is not None and dx is not None:
        dx = dsv4_rope(dx.reshape(original_shape), cos_sin_cache, position_ids)
    return dx, dw, None, None, None, None, None, None, None


fp8_qat_linear.register_autograd(_linear_backward, setup_context=_linear_setup_context)


@torch.library.custom_op("prime_rl::fp8_qat_grouped_mm", mutates_args=())
def fp8_qat_grouped_mm(
    x: torch.Tensor,
    weight: torch.Tensor,
    offs: torch.Tensor,
    stochastic_rounding: bool,
    step: torch.Tensor,
    seed: int,
    weight_key: torch.Tensor,
    expert_start: int,
) -> torch.Tensor:
    import deep_gemm

    total_m, padded_m, layout, blocks, _, starts, lengths, block_starts = build_grouped_layout(offs, total_m=x.shape[0])
    x_fp8 = grouped_per_token_cast_to_fp8_triton(
        x, padded_m, blocks, starts, lengths, block_starts, True, GROUP_ALIGNMENT, scale_floor=1e-10
    )
    w_fp8 = cast_qat_weight(
        weight.transpose(1, 2),
        True,
        stochastic_rounding,
        step=step,
        seed=seed,
        weight_key=weight_key,
        expert_start=expert_start,
    )
    out = x.new_empty((padded_m, weight.shape[-1]))
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(x_fp8, w_fp8, out, layout, use_psum_layout=False)
    return unpack_rows_triton(out, total_m, blocks, starts, lengths, block_starts)


@fp8_qat_grouped_mm.register_fake
def _grouped_fake(x, weight, offs, stochastic_rounding, step, seed, weight_key, expert_start):
    return x.new_empty((x.shape[0], weight.shape[-1]))


def _grouped_setup_context(ctx, inputs, output):
    x, weight, offs, ctx.sr, step, ctx.seed, weight_key, ctx.expert_start = inputs
    ctx.save_for_backward(x, weight, offs, step.clone(), weight_key)


def _grouped_backward(ctx, grad):
    x, weight, offs, step, weight_key = ctx.saved_tensors
    grad = grad.contiguous()
    rounded_weight = _rounded_weight(
        weight.detach().transpose(1, 2), step, ctx.seed, weight_key, ctx.expert_start, ctx.sr
    )
    dx = torch._grouped_mm(grad, rounded_weight, offs=offs) if ctx.needs_input_grad[0] else None
    dw = torch._grouped_mm(x.T, grad, offs=offs) if ctx.needs_input_grad[1] else None
    return dx, dw, None, None, None, None, None, None


fp8_qat_grouped_mm.register_autograd(_grouped_backward, setup_context=_grouped_setup_context)


@torch.library.custom_op("prime_rl::fp8_qat_rounded_weight", mutates_args=())
def _rounded_weight(
    weight: torch.Tensor, step: torch.Tensor, seed: int, weight_key: torch.Tensor, expert_start: int, sr: bool
) -> torch.Tensor:
    return cast_qat_weight(
        weight, True, sr, step=step, seed=seed, weight_key=weight_key, expert_start=expert_start, dequantize=True
    )[0]


@_rounded_weight.register_fake
def _rounded_weight_fake(weight, step, seed, weight_key, expert_start, sr):
    return weight.new_empty(weight.shape, dtype=torch.bfloat16)


def logical_weight_keys(model: nn.Module) -> dict[int, int]:
    return {
        id(weight): int.from_bytes(hashlib.blake2b(name.encode(), digest_size=8).digest(), "little") & ((1 << 63) - 1)
        for name, weight in model.named_parameters()
    }


class FP8QATLinear(nn.Linear):
    """Bias-free dense or grouped FP8 projection retaining trainable high-precision weights."""

    def __init__(
        self, *args, groups: int = 1, stochastic_rounding: bool = False, seed: int = 0, weight_key: int = 0, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.groups = groups
        self.stochastic_rounding = stochastic_rounding
        self.seed = seed
        self.weight_key = weight_key
        self.register_buffer("rounding_key", None, persistent=False)
        self.rounding_step = None

    def forward(self, x, cos_sin_cache=None, position_ids=None):
        return fp8_qat_linear(
            x,
            self.weight,
            self.groups,
            self.stochastic_rounding,
            self.rounding_step,
            self.seed,
            self.rounding_key,
            cos_sin_cache,
            position_ids,
        )


def replace_linear_with_fp8_qat(
    model: nn.Module, ignore_patterns: list[str], stochastic_rounding: bool, seed: int
) -> None:
    replaced = []
    keys = logical_weight_keys(model)
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
                seed=seed,
                weight_key=keys[id(module.weight)],
            )
        replacement = replacements.setdefault(id(module), replacement)
        replacement.weight = module.weight
        replacement.train(module.training)
        parent_name, attr = name.rsplit(".", 1) if "." in name else ("", name)
        setattr(model.get_submodule(parent_name), attr, replacement)
        replaced.append(name)
    get_logger().info(f"Enabled FP8 QAT on {len(replaced)} projections (stochastic_rounding={stochastic_rounding})")


class FP8QATExpertCompute(GroupedGemmExpertCompute):
    def __init__(self, stochastic_rounding: bool, seed: int):
        super().__init__(fp8_qat_grouped_mm)
        self.stochastic_rounding = stochastic_rounding
        self.seed = seed

    def linear(self, experts, name, x, weight, offsets):
        weight_key = getattr(experts, f"_fp8_rounding_key_{name}")
        if isinstance(weight_key, DTensor):
            weight_key = weight_key.to_local()
        return fp8_qat_grouped_mm(
            x,
            weight,
            offsets,
            self.stochastic_rounding,
            experts.rounding_step,
            self.seed,
            weight_key,
            experts.fp8_qat_expert_start,
        )


def initialize_qat_rounding(model: nn.Module) -> None:
    step = torch.zeros((), device="cuda", dtype=torch.int64)
    model.register_buffer("_fp8_rounding_step", step, persistent=False)
    for module in model.modules():
        if isinstance(module, FP8QATLinear):
            module.rounding_key = torch.tensor(module.weight_key, device="cuda", dtype=torch.int64)
            module.rounding_step = step
        elif isinstance(getattr(module, "compute", None), FP8QATExpertCompute):
            for name, key in module.fp8_qat_keys.items():
                module.register_buffer(
                    f"_fp8_rounding_key_{name}", torch.tensor(key, device="cuda", dtype=torch.int64), persistent=False
                )
            module.rounding_step = step
