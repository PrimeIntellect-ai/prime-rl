"""Inductor heuristics that compile FP8 group quantization as a single-read persistent reduction with a fixed launch config."""

from __future__ import annotations

from typing import Any

import torch
from quant_torch import BLOCK
from torch._inductor.choices import InductorChoices
from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
from torch._inductor.codegen.triton import FixedTritonConfig
from torch._inductor.compile_fx import compile_fx, compile_fx_inner
from torch._inductor.dependencies import MemoryDep
from torch._inductor.virtualized import V

STRIDED_GROUP_QUANT_CONFIG = {"XBLOCK": 128, "num_warps": 4}
CONTIGUOUS_GROUP_QUANT_CONFIG = {"XBLOCK": 16, "num_warps": 4}


def fp8_group_quant_layout(features: SIMDKernelFeatures, kernel_kwargs: dict[str, Any]) -> str | None:
    """Group layout by the reads' stride along the reduction, if the only reductions are 128-wide amaxes and FP8 is stored."""
    if not features.is_reduction() or kernel_kwargs.get("mix_order_reduction"):
        return None
    if not V.graph.sizevars.statically_known_equals(features.reduction_numel, BLOCK):
        return None
    reductions = features.reduction_nodes()
    if any(node.node.get_reduction_type() != "max" for node in reductions):
        return None
    if not any(
        buffer.node.get_dtype() == torch.float8_e4m3fn
        for node in features.scheduler_nodes()
        for buffer in node.get_outputs()
    ):
        return None
    contiguous = {
        dep.stride1_for_last_dim(result_for_complex_expression=False)
        for node in reductions
        for dep in node.read_writes.reads
        if isinstance(dep, MemoryDep) and dep.var_names and dep.var_names[-1] in dep.index.free_symbols
    }
    return {frozenset({True}): "contiguous", frozenset({False}): "strided"}.get(frozenset(contiguous))


class Fp8GroupQuantChoices(InductorChoices):
    """Make strided (128x1) FP8 group quant kernels persistent with a fixed launch config, other kernels unchanged."""

    configs: dict[str, dict[str, int]] = {"strided": STRIDED_GROUP_QUANT_CONFIG}

    def triton_kernel_kwargs(self, kernel_cls, features, groups, kernel_kwargs):
        config = self.configs.get(fp8_group_quant_layout(features, kernel_kwargs))
        if config is None:
            return kernel_kwargs
        return {**kernel_kwargs, "override_persistent_reduction": True, "fixed_config": FixedTritonConfig(dict(config))}

    def uuid(self) -> str:
        return f"fp8-group-quant-v2-{sorted(self.configs.items())}"


class Fp8GroupQuantChoicesBothLayouts(Fp8GroupQuantChoices):
    """Also make contiguous (1x128) FP8 group quant kernels persistent with a fixed launch config."""

    configs = {"strided": STRIDED_GROUP_QUANT_CONFIG, "contiguous": CONTIGUOUS_GROUP_QUANT_CONFIG}


def fp8_group_quant_backend(choices_class: type[Fp8GroupQuantChoices] = Fp8GroupQuantChoices):
    """Inductor backend that installs `choices_class` around every forward and backward graph compile, on any thread."""

    def backend(gm: torch.fx.GraphModule, example_inputs, options: dict[str, Any] | None = None):
        def inner_compile(*args, **kwargs):
            with V.set_choices_handler(choices_class()):
                return compile_fx_inner(*args, **kwargs)

        config_patches = {**(options or {}), "inductor_choices_class": choices_class}
        return compile_fx(gm, example_inputs, inner_compile=inner_compile, config_patches=config_patches)

    return backend
