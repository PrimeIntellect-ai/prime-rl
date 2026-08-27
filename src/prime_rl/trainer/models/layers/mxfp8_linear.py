from __future__ import annotations

import torch
from torch import nn
from torchao.prototype.mx_formats import ScaleCalculationMode
from torchao.prototype.mx_formats.mx_linear import _to_mxfp8_then_scaled_mm
from torchao.quantization.quantize_.common import KernelPreference


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
