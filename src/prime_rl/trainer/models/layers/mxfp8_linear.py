from __future__ import annotations

import re

import torch
from torch import nn
from torchao.prototype.mx_formats import ScaleCalculationMode
from torchao.prototype.mx_formats import mx_linear as tao_mx_linear
from torchao.prototype.mx_formats.mx_linear import _to_mxfp8_then_scaled_mm
from torchao.quantization.quantize_.common import KernelPreference

from prime_rl.configs.trainer import MXFP8Recipe
from prime_rl.utils.logger import get_logger


def _cache_mxfp8_dim0_weight_across_checkpoint_recompute() -> None:
    if getattr(tao_mx_linear.mx_mm, "_prime_rl_dim0_cached", False):
        return

    MXTensor = tao_mx_linear.MXTensor
    cache: dict[int, tuple[int, object]] = {}

    @staticmethod
    def forward(
        ctx,
        input_hp,
        weight_hp,
        in_elem_dtype,
        w_elem_dtype,
        grad_elem_dtype,
        block_size,
        kernel_preference,
        mxfp8_dim0_cast_kernel_choice,
        mxfp8_dim1_cast_kernel_choice,
        scale_calculation_mode,
        wgrad_with_hp,
    ):
        ctx.save_for_backward(input_hp, weight_hp)
        ctx.in_elem_dtype = in_elem_dtype
        ctx.w_elem_dtype = w_elem_dtype
        ctx.grad_elem_dtype = grad_elem_dtype
        ctx.block_size = block_size
        ctx.kernel_preference = kernel_preference
        ctx.wgrad_with_hp = wgrad_with_hp
        ctx.mxfp8_dim0_cast_kernel_choice = mxfp8_dim0_cast_kernel_choice
        ctx.mxfp8_dim1_cast_kernel_choice = mxfp8_dim1_cast_kernel_choice
        ctx.scale_calculation_mode = scale_calculation_mode
        input_orig_shape = input_hp.shape
        input_hp_r = input_hp.reshape(-1, input_orig_shape[-1])
        input_mx_r_dim0 = MXTensor.to_mx(
            input_hp_r,
            in_elem_dtype,
            block_size,
            scale_calculation_mode,
            kernel_preference,
            mxfp8_dim0_cast_kernel_choice=mxfp8_dim0_cast_kernel_choice,
        )
        cache_key = id(weight_hp)
        cached = cache.get(cache_key)
        if cached is not None and cached[0] == weight_hp._version:
            weight_mx_dim0 = cached[1]
        else:
            weight_mx_dim0 = MXTensor.to_mx(
                weight_hp,
                w_elem_dtype,
                block_size,
                scale_calculation_mode,
                kernel_preference,
                mxfp8_dim0_cast_kernel_choice=mxfp8_dim0_cast_kernel_choice,
            )
            cache[cache_key] = (weight_hp._version, weight_mx_dim0)

        output = torch.mm(input_mx_r_dim0, weight_mx_dim0.t())
        output = output.reshape(*input_orig_shape[:-1], output.shape[-1])
        return output

    tao_mx_linear.mx_mm.forward = forward
    tao_mx_linear.mx_mm._prime_rl_dim0_cached = True


class MXFP8Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        kernel_preference: KernelPreference = KernelPreference.AUTO,
        wgrad_with_hp: bool = False,
        scale_calculation_mode: ScaleCalculationMode = ScaleCalculationMode.RCEIL,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)
        self.kernel_preference = kernel_preference
        self.wgrad_with_hp = wgrad_with_hp
        self.scale_calculation_mode = scale_calculation_mode

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = _to_mxfp8_then_scaled_mm(
            input,
            self.weight,
            self.kernel_preference,
            self.scale_calculation_mode,
            wgrad_with_hp=self.wgrad_with_hp,
        )
        if self.bias is not None:
            output = output + self.bias
        return output

    @classmethod
    def from_linear(
        cls,
        mod: nn.Linear,
        *,
        kernel_preference: KernelPreference = KernelPreference.AUTO,
        wgrad_with_hp: bool = False,
        scale_calculation_mode: ScaleCalculationMode = ScaleCalculationMode.RCEIL,
    ) -> "MXFP8Linear":
        with torch.device("meta"):
            new_mod = cls(
                mod.in_features,
                mod.out_features,
                bias=mod.bias is not None,
                kernel_preference=kernel_preference,
                wgrad_with_hp=wgrad_with_hp,
                scale_calculation_mode=scale_calculation_mode,
            )
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        return new_mod


def replace_linear_with_mxfp8_linear(model: nn.Module, recipe: MXFP8Recipe, ignore_modules: list[str]) -> None:
    _cache_mxfp8_dim0_weight_across_checkpoint_recompute()
    wgrad_with_hp = recipe == "mxfp8_rceil_wgrad_with_hp"
    logger = get_logger()
    logger.info(f"Replacing linear layers with MXFP8 linear layers (recipe={recipe}, ignore={ignore_modules})")
    replaced_modules: list[str] = []
    skipped_modules: list[str] = []
    skipped_unaligned: list[str] = []
    for name, module in dict(model.named_modules()).items():
        if not isinstance(module, nn.Linear):
            continue
        if any(re.search(pattern, name) for pattern in ignore_modules):
            skipped_modules.append(name)
            continue
        if (module.in_features & 31) != 0 or (module.out_features & 31) != 0:
            skipped_unaligned.append(f"{name}({module.in_features}->{module.out_features})")
            continue
        parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(
            parent,
            attr_name,
            MXFP8Linear.from_linear(module, kernel_preference=KernelPreference.AUTO, wgrad_with_hp=wgrad_with_hp),
        )
        replaced_modules.append(name)

    logger.info(
        f"Replaced {len(replaced_modules)} linear layers with MXFP8 linear "
        f"(skipped {len(skipped_modules)} by name, {len(skipped_unaligned)} by 32-div); "
        f"first replaced={replaced_modules[:3]}, "
        f"first skipped(name)={skipped_modules[:3]}, "
        f"first skipped(unaligned)={skipped_unaligned[:3]}"
    )
