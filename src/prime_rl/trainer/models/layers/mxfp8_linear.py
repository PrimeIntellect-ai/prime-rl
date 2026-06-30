from __future__ import annotations

import re

import torch
from torch import nn

try:
    from torchao.prototype.mx_formats.config import KernelPreference, ScaleCalculationMode
    from torchao.prototype.mx_formats.mx_linear import _to_mxfp8_then_scaled_mm
except ImportError:
    # torchao is an x86_64-only optional dependency; MXFP8 paths are GPU/SM100-only
    # at runtime, so leaving these None is safe — only the forward bodies call them.
    KernelPreference = ScaleCalculationMode = None
    _to_mxfp8_then_scaled_mm = None

from prime_rl.utils.logger import get_logger

# Reuse the same default ignore list as the config-level default. These are layers
# that must stay in bf16: routers, gates, LM head, MTP/indexer projections, and
# hybrid-Mamba input projections.
DEFAULT_MXFP8_IGNORE_PATTERNS: list[str] = [
    "lm_head",
    "router",
    r"mlp\.gate\.",
    "shared_expert_gate",
    "eh_proj",
    "weights_proj",
    "in_proj_a",
    "in_proj_b",
]


def _scaling_mode(name: str):
    return {"rceil": ScaleCalculationMode.RCEIL, "floor": ScaleCalculationMode.FLOOR}[name]


def _kernel_preference(name: str):
    return {"auto": KernelPreference.AUTO, "emulated": KernelPreference.EMULATED}[name]


class MXFP8Linear(nn.Linear):
    """nn.Linear replacement using torchao MXFP8 (microscaling FP8, 1x32 e8m0-scaled) matmul.

    Forward and both backward GEMMs dynamically quantize their operands to MXFP8 via
    torchao's ``mx_mm`` autograd function. Requires SM100 (Blackwell) and bfloat16 weights.
    """

    def __init__(self, *args, scaling_mode: str = "rceil", kernel: str = "auto", **kwargs):
        super().__init__(*args, **kwargs)
        self._mx_scaling_mode = _scaling_mode(scaling_mode)
        self._mx_kernel = _kernel_preference(kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = _to_mxfp8_then_scaled_mm(x.contiguous(), self.weight, self._mx_kernel, self._mx_scaling_mode)
        if self.bias is not None:
            out = out + self.bias
        return out

    @classmethod
    def from_linear(cls, mod: nn.Linear, scaling_mode: str = "rceil", kernel: str = "auto") -> "MXFP8Linear":
        with torch.device("meta"):
            new_mod = cls(
                mod.in_features,
                mod.out_features,
                bias=mod.bias is not None,
                scaling_mode=scaling_mode,
                kernel=kernel,
            )
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        return new_mod


def replace_linear_with_mxfp8_linear(
    model: nn.Module,
    ignore_modules: list[str] | None = None,
    scaling_mode: str = "rceil",
    kernel: str = "auto",
) -> None:
    """Replace nn.Linear in `model` with MXFP8Linear, skipping any module whose qualified
    name matches an ignore pattern (substring or regex).

    Independently of the name-based ignore list, we skip any nn.Linear whose in_features or
    out_features is not a multiple of 16 (the float8 tensorcore requirement, matching
    torchtitan's MXFP8 module filter) and keep it in bf16.
    """
    if ignore_modules is None:
        ignore_modules = list(DEFAULT_MXFP8_IGNORE_PATTERNS)
    logger = get_logger()
    logger.info(f"Replacing linear layers with MXFP8 linear layers (ignore={ignore_modules})")
    replaced_modules: list[str] = []
    skipped_modules: list[str] = []
    skipped_unaligned: list[str] = []
    named_modules = dict(model.named_modules())
    for name, module in named_modules.items():
        if not isinstance(module, nn.Linear):
            continue
        if any(re.search(pattern, name) for pattern in ignore_modules):
            skipped_modules.append(name)
            continue
        if module.in_features % 16 != 0 or module.out_features % 16 != 0:
            skipped_unaligned.append(f"{name}({module.in_features}->{module.out_features})")
            continue
        parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr_name, MXFP8Linear.from_linear(module, scaling_mode=scaling_mode, kernel=kernel))
        replaced_modules.append(name)

    logger.info(
        f"Replaced {len(replaced_modules)} linear layers with MXFP8 linear "
        f"(skipped {len(skipped_modules)} by name, {len(skipped_unaligned)} by 16-divisibility); "
        f"first replaced={replaced_modules[:3]}, "
        f"first skipped(name)={skipped_modules[:3]}, "
        f"first skipped(unaligned)={skipped_unaligned[:3]}"
    )
