from __future__ import annotations

import re

import torch
from torch import nn

from torchao.prototype.mx_formats import ScaleCalculationMode
from torchao.prototype.mx_formats.mx_linear import _to_mxfp8_then_scaled_mm
from torchao.quantization.quantize_.common import KernelPreference

from prime_rl.configs.trainer import MXFP8Recipe
from prime_rl.utils.logger import get_logger


def _recipe_params(recipe: MXFP8Recipe) -> tuple[KernelPreference, bool]:
    emulated = recipe == "mxfp8_emulated_rceil"
    wgrad_with_hp = recipe == "mxfp8_rceil_wgrad_with_hp"
    kernel_preference = KernelPreference.EMULATED if emulated else KernelPreference.AUTO
    return kernel_preference, wgrad_with_hp


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
        sm_ver = (10, 0)
        if kernel_preference == KernelPreference.EMULATED:
            raise RuntimeError(f"MXFP8 requires SM{sm_ver}, emulated kernels are unsupported")
        cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
        if cap is None or cap < sm_ver:
            raise RuntimeError(f"MXFP8 requires SM{sm_ver} but device is SM{cap}")
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
    kernel_preference, wgrad_with_hp = _recipe_params(recipe)
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
        if (module.in_features % 32) != 0 or (module.out_features % 32) != 0:
            skipped_unaligned.append(f"{name}({module.in_features}->{module.out_features})")
            continue
        parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(
            parent,
            attr_name,
            MXFP8Linear.from_linear(module, kernel_preference=kernel_preference, wgrad_with_hp=wgrad_with_hp),
        )
        replaced_modules.append(name)

    logger.info(
        f"Replaced {len(replaced_modules)} linear layers with MXFP8 linear "
        f"(skipped {len(skipped_modules)} by name, {len(skipped_unaligned)} by 32-div); "
        f"first replaced={replaced_modules[:3]}, "
        f"first skipped(name)={skipped_modules[:3]}, "
        f"first skipped(unaligned)={skipped_unaligned[:3]}"
    )
