"""Inductor post-grad pass that moves FP8 activation and gradient quantization out of the DeepGEMM ops into torch ops."""

from __future__ import annotations

import functools
import inspect
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch._inductor import config as inductor_config
from torch._inductor.choices import InductorChoices
from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
from torch._inductor.compile_fx import compile_fx, compile_fx_inner
from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files
from torch._inductor.pattern_matcher import Arg, CallFunction, Match, PatternMatcherPass, register_graph_pattern
from torch._inductor.virtualized import V

from prime_rl.configs.trainer import FP8WgradQuant
from prime_rl.trainer.models.kernels import fp8_torch_quant
from prime_rl.trainer.models.kernels.fp8_torch_quant import BLOCK, quant_1x128, quant_128x1_tp
from prime_rl.trainer.models.layers import fp8_linear
from prime_rl.utils.logger import get_logger

PatternRegistrar = Callable[[PatternMatcherPass, FP8WgradQuant], None]

FP8_MM = torch.ops.prime_rl.fp8_blockwise_mm.default
FP8_MM_BACKWARD = torch.ops.prime_rl.fp8_blockwise_mm_backward.default


def register_fp8_linear_patterns(patterns: PatternMatcherPass, wgrad_quant: FP8WgradQuant) -> None:
    """Register rewrites of `prime_rl::fp8_blockwise_mm` and its backward into torch-op quants and pre-quantized GEMMs."""

    @register_graph_pattern(CallFunction(FP8_MM, Arg(), Arg(), Arg()), pass_dict=patterns)
    def rewrite_fp8_mm(match: Match, x: torch.fx.Node, weight: torch.fx.Node, block_size: int) -> None:
        assert block_size == BLOCK, f"the torch-op quantizers use {BLOCK}-element groups, got block_size={block_size}"

        def replacement(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            x_fp8, x_scale = quant_1x128(x.reshape(-1, x.shape[-1]))
            out = torch.ops.prime_rl.fp8_blockwise_mm_prequant(x_fp8, x_scale, weight, block_size)
            return out.reshape(*x.shape[:-1], weight.shape[0])

        match.replace_by_example(replacement, [x, weight])

    @register_graph_pattern(CallFunction(FP8_MM_BACKWARD, Arg(), Arg(), Arg(), Arg(), Arg(), Arg()), pass_dict=patterns)
    def rewrite_fp8_mm_backward(
        match: Match,
        grad_output: torch.fx.Node,
        x: torch.fx.Node,
        weight: torch.fx.Node,
        block_size: int,
        needs_grad_x: bool,
        needs_grad_weight: bool,
    ) -> None:
        assert block_size == BLOCK, f"the torch-op quantizers use {BLOCK}-element groups, got block_size={block_size}"

        def replacement(
            grad_output: torch.Tensor, x: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
            grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
            x_2d = x.reshape(-1, x.shape[-1])
            grad_x = grad_weight = None
            if needs_grad_x:
                grad_output_fp8, grad_output_scale = quant_1x128(grad_output_2d)
                grad_x = torch.ops.prime_rl.fp8_blockwise_mm_dgrad_prequant(
                    grad_output_fp8, grad_output_scale, weight, block_size
                ).reshape(x.shape)
            if needs_grad_weight and wgrad_quant == "torch":
                grad_output_t_fp8, grad_output_t_scale = quant_128x1_tp(grad_output_2d)
                x_t_fp8, x_t_scale = quant_128x1_tp(x_2d)
                grad_weight = torch.ops.prime_rl.fp8_blockwise_mm_wgrad_prequant(
                    grad_output_t_fp8, grad_output_t_scale, x_t_fp8, x_t_scale, weight
                )
            elif needs_grad_weight:
                grad_weight = torch.ops.prime_rl.fp8_blockwise_mm_wgrad(grad_output_2d, x_2d, weight, block_size)
            return grad_x, grad_weight

        match.replace_by_example(replacement, [grad_output, x, weight])


PATTERN_REGISTRARS: tuple[PatternRegistrar, ...] = (register_fp8_linear_patterns,)
REWRITTEN_OPS = (FP8_MM, FP8_MM_BACKWARD)


@functools.cache
def _patterns(wgrad_quant: FP8WgradQuant) -> PatternMatcherPass:
    patterns = PatternMatcherPass(pass_name=f"fp8_quant_pass_{wgrad_quant}")
    for register in PATTERN_REGISTRARS:
        register(patterns, wgrad_quant)
    return patterns


def _hashable(value: object) -> object:
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _hashable(item)) for key, item in value.items()))
    return value


def _dedupe(graph: torch.fx.Graph, candidates: set[torch.fx.Node]) -> int:
    """Merge identical pure nodes among `candidates`, so repeated quants of one tensor run once."""
    canonical: dict[tuple, torch.fx.Node] = {}
    removed = 0
    for node in list(graph.nodes):
        if node not in candidates or node.op != "call_function" or node.is_impure():
            continue
        key = (node.target, _hashable(node.args), _hashable(node.kwargs))
        if key in canonical:
            node.replace_all_uses_with(canonical[key])
            graph.erase_node(node)
            removed += 1
        else:
            canonical[key] = node
    return removed


class Fp8QuantPass(CustomGraphPass):
    """Rewrites FP8 GEMM ops into torch-op quants, which Inductor fuses into their producers, and pre-quantized GEMMs."""

    def __init__(self, wgrad_quant: FP8WgradQuant) -> None:
        self.wgrad_quant = wgrad_quant

    def __call__(self, graph: torch.fx.Graph) -> None:
        nodes_before = set(graph.nodes)
        ops_before = Counter(
            str(node.target) for node in graph.nodes if node.op == "call_function" and node.target in REWRITTEN_OPS
        )
        rewritten = _patterns(self.wgrad_quant).apply(graph)
        if not rewritten:
            return
        deduped = _dedupe(graph, set(graph.nodes) - nodes_before)
        graph.eliminate_dead_code()
        get_logger().info(
            f"FP8 quant pass ({self.wgrad_quant} wgrad quant) rewrote {rewritten} ops {dict(ops_before)}, "
            f"deduplicated {deduped} repeated quant nodes"
        )

    def uuid(self) -> bytes:
        source_files = {__file__, fp8_torch_quant.__file__, fp8_linear.__file__}
        source_files.update(inspect.getfile(register) for register in PATTERN_REGISTRARS)
        return get_hash_for_files(tuple(sorted(source_files)), extra=self.wgrad_quant)


class PersistentGroupReductionChoices(InductorChoices):
    """Inductor heuristics that make every 128-element reduction persistent, including the strided 128x1 quant amax."""

    @staticmethod
    def should_use_persistent_reduction(features: SIMDKernelFeatures, cooperative_reduction: bool) -> bool:
        if inductor_config.triton.persistent_reductions and V.graph.sizevars.statically_known_equals(
            features.reduction_numel, BLOCK
        ):
            return True
        return InductorChoices.should_use_persistent_reduction(features, cooperative_reduction)

    def uuid(self) -> str:
        return get_hash_for_files((__file__, fp8_torch_quant.__file__)).hex()


class InductorChoicesBackend:
    """`torch.compile` backend that runs Inductor with `choices_class` installed around every graph compile, on any thread."""

    def __init__(self, choices_class: type[InductorChoices]) -> None:
        self.choices_class = choices_class

    def __call__(
        self, gm: torch.fx.GraphModule, example_inputs: Sequence[Any], options: dict[str, Any] | None = None
    ) -> Callable:
        def inner_compile(*args, **kwargs):
            with V.set_choices_handler(self.choices_class()):
                return compile_fx_inner(*args, **kwargs)

        config_patches = {**(options or {}), "inductor_choices_class": self.choices_class}
        return compile_fx(gm, example_inputs, inner_compile=inner_compile, config_patches=config_patches)
