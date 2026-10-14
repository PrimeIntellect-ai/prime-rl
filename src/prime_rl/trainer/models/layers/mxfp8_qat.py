"""Native MXFP8 forward with a BF16 straight-through backward surrogate."""

import re

import torch
from torch import nn

from prime_rl.trainer.models.layers.expert_compute import GroupedGemmExpertCompute
from prime_rl.trainer.models.layers.grouped_linear import GroupedLinear
from prime_rl.utils.logger import get_logger


def _kernel():
    import prime_kernels

    return prime_kernels.load("mxfp8")


def _quantize(x, rounding_seed=None, rounding_salt=None, rounding_offset=0):
    q, scales = _kernel().quantize(
        x, rounding_seed=rounding_seed, rounding_salt=rounding_salt, rounding_offset=rounding_offset
    )
    return q, scales.view(torch.float8_e8m0fnu).float()


@torch.library.custom_op("prime_rl::mxfp8_qat_linear", mutates_args=())
def mxfp8_qat_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    groups: int = 1,
    rounding_seed: torch.Tensor | None = None,
    rounding_salt: torch.Tensor | None = None,
) -> torch.Tensor:
    import deep_gemm

    output = x.new_empty((*x.shape[:-1], weight.shape[0] // groups))
    if x.numel() == 0:
        return output
    xq, xs = _quantize(x.reshape(-1, x.shape[-1]))
    wq, ws = _quantize(weight, rounding_seed, rounding_salt)
    if groups == 1:
        deep_gemm.fp8_gemm_nt((xq, xs), (wq, ws), output.view(-1, weight.shape[0]), recipe_a=(1, 32), recipe_b=(1, 32))
    else:
        deep_gemm.fp8_einsum(
            "bhr,hdr->bhd",
            (xq.view(-1, groups, x.shape[-1]), xs.view(-1, groups, xs.shape[-1])),
            (wq.view(groups, -1, weight.shape[-1]), ws.view(groups, -1, ws.shape[-1])),
            output.view(-1, groups, weight.shape[0] // groups),
            recipe=(1, 1, 32),
        )
    return output


@mxfp8_qat_linear.register_fake
def _linear_fake(x, weight, groups=1, rounding_seed=None, rounding_salt=None):
    return x.new_empty((*x.shape[:-1], weight.shape[0] // groups))


def _linear_context(ctx, inputs, output):
    x, weight, ctx.groups, rounding_seed, rounding_salt = inputs
    ctx.save_for_backward(x, weight, rounding_seed, rounding_salt)


@torch.library.custom_op("prime_rl::mxfp8_qat_rounded_weight", mutates_args=())
def _rounded_weight(
    weight: torch.Tensor,
    rounding_seed: torch.Tensor | None = None,
    rounding_salt: torch.Tensor | None = None,
    rounding_offset: int = 0,
) -> torch.Tensor:
    kernel = _kernel()
    return kernel.dequantize(
        *kernel.quantize(
            weight, rounding_seed=rounding_seed, rounding_salt=rounding_salt, rounding_offset=rounding_offset
        )
    )


@_rounded_weight.register_fake
def _rounded_weight_fake(weight, rounding_seed=None, rounding_salt=None, rounding_offset=0):
    return torch.empty_like(weight, memory_format=torch.contiguous_format)


def _linear_backward(ctx, grad):
    x, weight, rounding_seed, rounding_salt = ctx.saved_tensors
    groups = ctx.groups
    rounded = _rounded_weight(weight.detach(), rounding_seed, rounding_salt) if ctx.needs_input_grad[0] else None
    if groups == 1:
        dy = grad.reshape(-1, grad.shape[-1])
        dx = (dy @ rounded).view_as(x) if ctx.needs_input_grad[0] else None
        dw = dy.T @ x.reshape(-1, x.shape[-1]) if ctx.needs_input_grad[1] else None
    else:
        dy = grad.reshape(-1, groups, grad.shape[-1]).transpose(0, 1)
        dx = (
            torch.bmm(dy, rounded.view(groups, -1, x.shape[-1])).transpose(0, 1).reshape_as(x)
            if ctx.needs_input_grad[0]
            else None
        )
        dw = (
            torch.bmm(dy.transpose(1, 2), x.reshape(-1, groups, x.shape[-1]).transpose(0, 1)).reshape_as(weight)
            if ctx.needs_input_grad[1]
            else None
        )
    return dx, dw, None, None, None


mxfp8_qat_linear.register_autograd(_linear_backward, setup_context=_linear_context)


@torch.library.custom_op("prime_rl::mxfp8_qat_grouped_mm", mutates_args=())
def mxfp8_qat_grouped_mm(
    x: torch.Tensor,
    weight: torch.Tensor,
    offs: torch.Tensor,
    rounding_seed: torch.Tensor | None = None,
    rounding_salt: torch.Tensor | None = None,
    rounding_offset: int = 0,
) -> torch.Tensor:
    import deep_gemm

    output = x.new_zeros((x.shape[0], weight.shape[-1]))
    if x.shape[0] == 0:
        return output
    # Dispatch aligns expert groups and may reserve unused rows after the last group.
    layout = torch.searchsorted(offs, torch.arange(x.shape[0], device=x.device, dtype=offs.dtype), right=True)
    layout = torch.where(layout < offs.numel(), layout, -1).to(torch.int32)
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        _quantize(x),
        _quantize(weight.transpose(1, 2), rounding_seed, rounding_salt, rounding_offset),
        output,
        layout,
        recipe_a=(1, 32),
        recipe_b=(1, 32),
        use_psum_layout=False,
    )
    return output


@mxfp8_qat_grouped_mm.register_fake
def _grouped_fake(x, weight, offs, rounding_seed=None, rounding_salt=None, rounding_offset=0):
    return x.new_empty((x.shape[0], weight.shape[-1]))


def _grouped_context(ctx, inputs, output):
    x, weight, offs, rounding_seed, rounding_salt, ctx.rounding_offset = inputs
    ctx.save_for_backward(x, weight, offs, rounding_seed, rounding_salt)


def _grouped_backward(ctx, grad):
    x, weight, offs, rounding_seed, rounding_salt = ctx.saved_tensors
    if x.shape[0] == 0:
        return torch.zeros_like(x), torch.zeros_like(weight), None, None, None, None
    grad = grad.contiguous()
    dx = (
        torch._grouped_mm(
            grad,
            _rounded_weight(weight.detach().transpose(1, 2), rounding_seed, rounding_salt, ctx.rounding_offset),
            offs=offs,
        )
        if ctx.needs_input_grad[0]
        else None
    )
    dw = torch._grouped_mm(x.T, grad, offs=offs) if ctx.needs_input_grad[1] else None
    return dx, dw, None, None, None, None


mxfp8_qat_grouped_mm.register_autograd(_grouped_backward, setup_context=_grouped_context)


class MXFP8QATLinear(nn.Linear):
    def __init__(self, *args, groups=1, stochastic_rounding=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.groups = groups
        self.stochastic_rounding = stochastic_rounding
        self.fp8_rounding = None
        self.rounding_salt = None

    def forward(self, x):
        if self.stochastic_rounding and self.fp8_rounding is None:
            raise RuntimeError("Initialize MXFP8 stochastic rounding with setup_fp8_rounding before forward")
        result = mxfp8_qat_linear(
            x,
            self.weight,
            self.groups,
            self.fp8_rounding.seed if self.fp8_rounding is not None else None,
            self.rounding_salt,
        )
        if self.bias is not None:
            result = result + (self.bias.view(self.groups, -1) if self.groups > 1 else self.bias)
        return result


def replace_linear_with_mxfp8_qat(
    model: nn.Module,
    ignore_patterns: list[str],
    stochastic_rounding: bool = False,
) -> None:
    _kernel()
    replaced = []
    replacements = {}
    for name, module in list(model.named_modules(remove_duplicate=False)):
        if not isinstance(module, nn.Linear) or any(re.search(pattern, name) for pattern in ignore_patterns):
            continue
        groups = module.n_groups if isinstance(module, GroupedLinear) else 1
        if module.in_features % 128 or (module.out_features // groups) % 128:
            raise ValueError(f"MXFP8 QAT projection {name} requires input/output sizes divisible by 128")
        with torch.device("meta"):
            replacement = MXFP8QATLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                groups=groups,
                stochastic_rounding=stochastic_rounding,
            )
        replacement = replacements.setdefault(id(module), replacement)
        replacement.weight, replacement.bias = module.weight, module.bias
        replacement.train(module.training)
        parent_name, attr = name.rsplit(".", 1) if "." in name else ("", name)
        setattr(model.get_submodule(parent_name) if parent_name else model, attr, replacement)
        replaced.append(name)
    get_logger().info(f"Enabled MXFP8 QAT on {len(replaced)} projections")


class MXFP8QATExpertCompute(GroupedGemmExpertCompute):
    def __init__(self, stochastic_rounding=False):
        _kernel()
        super().__init__(mxfp8_qat_grouped_mm, token_group_alignment=128)
        self.stochastic_rounding = stochastic_rounding

    def _gemm(self, experts, name, x, weight, offsets):
        return mxfp8_qat_grouped_mm(
            x,
            weight,
            offsets,
            experts.fp8_rounding.seed if self.stochastic_rounding else None,
            experts.rounding_salts[name] if self.stochastic_rounding else None,
            experts.fp8_expert_offset * weight.shape[1] * weight.shape[2] if self.stochastic_rounding else 0,
        )
