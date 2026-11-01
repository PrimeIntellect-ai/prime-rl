"""Native MXFP8 forward with a BF16 straight-through backward surrogate."""

import re

import torch
from torch import nn

from prime_rl.trainer.models.layers.expert_compute import GroupedGemmExpertCompute
from prime_rl.trainer.models.layers.grouped_linear import GroupedLinear
from prime_rl.utils.logger import get_logger


def _kernel():
    # Vendored from prime-kernels feat/mxfp8-qat-rounding (6806817): the pinned prime-kernels lacks the MXFP8 quantizer
    from prime_rl.trainer.models.kernels import mxfp8_quant

    return mxfp8_quant


# The current optimizer step's weight casts, by rounding salt (one per parameter). The stochastic stream is fixed for
# the step, so the step's micro steps, activation-checkpoint recomputes and backward all draw the same cast. Each entry
# keeps its salt alive so no other salt can take its address within the step.
_weight_casts: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def clear_weight_casts() -> None:
    _weight_casts.clear()


def _cast(x, rounding_seed=None, rounding_salt=None, rounding_offset=0, cache=False):
    """E4M3 values and UE8M0 scale bytes of ``x``; with ``cache``, cast once per optimizer step."""
    if cache and rounding_salt.data_ptr() in _weight_casts:
        return _weight_casts[rounding_salt.data_ptr()][1:]
    q, scales = _kernel().quantize(
        x, rounding_seed=rounding_seed, rounding_salt=rounding_salt, rounding_offset=rounding_offset
    )
    if cache:
        _weight_casts[rounding_salt.data_ptr()] = (rounding_salt, q, scales)
    return q, scales


def _quantize(x, rounding_seed=None, rounding_salt=None, rounding_offset=0, packed=True, cache=False):
    q, scales = _cast(x, rounding_seed, rounding_salt, rounding_offset, cache)
    if not packed:
        return q, scales.view(torch.float8_e8m0fnu).float()
    return q, _packed_scales(scales)


def _packed_scales(scales: torch.Tensor) -> torch.Tensor:
    """UE8M0 bytes ``[..., MN, K / 32]`` -> the int32 scales DeepGEMM takes as they are: four per word along K, MN-major,
    MN padded to 4. Unlike fp32 scales they keep code 0 (all-zero blocks, e.g. padding rows), which DeepGEMM's fp32
    packing rejects as a subnormal."""
    packed = scales.view(torch.int32)
    mn, packed_k = packed.shape[-2:]
    storage = packed.new_empty((*packed.shape[:-2], packed_k, -(-mn // 4) * 4))
    storage[..., :mn].copy_(packed.transpose(-1, -2))
    return storage.transpose(-1, -2)[..., :mn, :]


@torch.library.custom_op("prime_rl::mxfp8_qat_linear", mutates_args=())
def mxfp8_qat_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    groups: int = 1,
    rounding_seed: torch.Tensor | None = None,
    rounding_salt: torch.Tensor | None = None,
    cache_weight: bool = False,
) -> torch.Tensor:
    import deep_gemm

    output = x.new_empty((*x.shape[:-1], weight.shape[0] // groups))
    if x.numel() == 0:
        return output
    # The grouped einsum reshapes the scales, so it takes them as fp32
    xq, xs = _quantize(x.reshape(-1, x.shape[-1]), packed=groups == 1)
    wq, ws = _quantize(weight, rounding_seed, rounding_salt, packed=groups == 1, cache=cache_weight)
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
def _linear_fake(x, weight, groups=1, rounding_seed=None, rounding_salt=None, cache_weight=False):
    return x.new_empty((*x.shape[:-1], weight.shape[0] // groups))


def _linear_context(ctx, inputs, output):
    x, weight, ctx.groups, rounding_seed, rounding_salt, ctx.cache_weight = inputs
    ctx.save_for_backward(x, weight, rounding_seed, rounding_salt)


@torch.library.custom_op("prime_rl::mxfp8_qat_rounded_weight", mutates_args=())
def _rounded_weight(
    weight: torch.Tensor,
    rounding_seed: torch.Tensor | None = None,
    rounding_salt: torch.Tensor | None = None,
    rounding_offset: int = 0,
    cache_weight: bool = False,
) -> torch.Tensor:
    return _kernel().dequantize(*_cast(weight, rounding_seed, rounding_salt, rounding_offset, cache_weight))


@_rounded_weight.register_fake
def _rounded_weight_fake(weight, rounding_seed=None, rounding_salt=None, rounding_offset=0, cache_weight=False):
    return torch.empty_like(weight, memory_format=torch.contiguous_format)


def _linear_backward(ctx, grad):
    x, weight, rounding_seed, rounding_salt = ctx.saved_tensors
    groups = ctx.groups
    rounded = (
        _rounded_weight(weight.detach(), rounding_seed, rounding_salt, 0, ctx.cache_weight)
        if ctx.needs_input_grad[0]
        else None
    )
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
    return dx, dw, None, None, None, None


mxfp8_qat_linear.register_autograd(_linear_backward, setup_context=_linear_context)


@torch.library.custom_op("prime_rl::mxfp8_qat_grouped_mm", mutates_args=())
def mxfp8_qat_grouped_mm(
    x: torch.Tensor,
    weight: torch.Tensor,
    offs: torch.Tensor,
    rounding_seed: torch.Tensor | None = None,
    rounding_salt: torch.Tensor | None = None,
    rounding_offset: int = 0,
    cache_weight: bool = False,
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
        _quantize(weight.transpose(1, 2), rounding_seed, rounding_salt, rounding_offset, cache=cache_weight),
        output,
        layout,
        recipe_a=(1, 32),
        recipe_b=(1, 32),
        use_psum_layout=False,
    )
    return output


@mxfp8_qat_grouped_mm.register_fake
def _grouped_fake(x, weight, offs, rounding_seed=None, rounding_salt=None, rounding_offset=0, cache_weight=False):
    return x.new_empty((x.shape[0], weight.shape[-1]))


def _grouped_context(ctx, inputs, output):
    x, weight, offs, rounding_seed, rounding_salt, ctx.rounding_offset, ctx.cache_weight = inputs
    ctx.save_for_backward(x, weight, offs, rounding_seed, rounding_salt)


def _grouped_backward(ctx, grad):
    x, weight, offs, rounding_seed, rounding_salt = ctx.saved_tensors
    if x.shape[0] == 0:
        return torch.zeros_like(x), torch.zeros_like(weight), None, None, None, None, None
    grad = grad.contiguous()
    dx = (
        torch._grouped_mm(
            grad,
            _rounded_weight(
                weight.detach().transpose(1, 2), rounding_seed, rounding_salt, ctx.rounding_offset, ctx.cache_weight
            ),
            offs=offs,
        )
        if ctx.needs_input_grad[0]
        else None
    )
    dw = torch._grouped_mm(x.T, grad, offs=offs) if ctx.needs_input_grad[1] else None
    return dx, dw, None, None, None, None, None


mxfp8_qat_grouped_mm.register_autograd(_grouped_backward, setup_context=_grouped_context)


class MXFP8QATLinear(nn.Linear):
    def __init__(self, *args, groups=1, stochastic_rounding=False, cache_weight=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.groups = groups
        self.stochastic_rounding = stochastic_rounding
        self.cache_weight = cache_weight
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
            self.cache_weight,
        )
        if self.bias is not None:
            result = result + (self.bias.view(self.groups, -1) if self.groups > 1 else self.bias)
        return result


def replace_linear_with_mxfp8_qat(
    model: nn.Module,
    ignore_patterns: list[str],
    stochastic_rounding: bool = False,
    cache_weights: bool = False,
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
                cache_weight=cache_weights,
            )
        replacement = replacements.setdefault(id(module), replacement)
        replacement.weight, replacement.bias = module.weight, module.bias
        replacement.train(module.training)
        parent_name, attr = name.rsplit(".", 1) if "." in name else ("", name)
        setattr(model.get_submodule(parent_name) if parent_name else model, attr, replacement)
        replaced.append(name)
    get_logger().info(f"Enabled MXFP8 QAT on {len(replaced)} projections")


class MXFP8QATExpertCompute(GroupedGemmExpertCompute):
    def __init__(self, stochastic_rounding=False, cache_weights=False):
        _kernel()
        super().__init__(mxfp8_qat_grouped_mm, token_group_alignment=128)
        self.stochastic_rounding = stochastic_rounding
        self.cache_weights = cache_weights

    def _gemm(self, experts, name, x, weight, offsets):
        return mxfp8_qat_grouped_mm(
            x,
            weight,
            offsets,
            experts.fp8_rounding.seed if self.stochastic_rounding else None,
            experts.rounding_salts[name] if self.stochastic_rounding else None,
            experts.fp8_expert_offset * weight.shape[1] * weight.shape[2] if self.stochastic_rounding else 0,
            self.cache_weights,
        )
