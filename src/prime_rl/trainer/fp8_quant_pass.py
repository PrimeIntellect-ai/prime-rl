"""Inductor post-grad pass that moves FP8 activation and gradient quantization out of the DeepGEMM ops into torch ops."""

from __future__ import annotations

import functools
import inspect
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any, Literal

import sympy
import torch
from torch._inductor.choices import InductorChoices
from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
from torch._inductor.codegen.triton import FixedTritonConfig, TritonKernel
from torch._inductor.compile_fx import compile_fx, compile_fx_inner
from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files
from torch._inductor.dependencies import MemoryDep
from torch._inductor.pattern_matcher import Arg, CallFunction, Match, PatternMatcherPass, register_graph_pattern
from torch._inductor.scheduler import BaseSchedulerNode, Scheduler
from torch._inductor.utils import sympy_product
from torch._inductor.virtualized import V

from prime_rl.configs.trainer import FP8QuantDivision, FP8QuantPassTarget, FP8WgradQuant
from prime_rl.trainer.models.kernels import fp8_torch_quant
from prime_rl.trainer.models.kernels.fp8_torch_quant import (
    BLOCK,
    QuantizerDivision,
    quant_1x128,
    quant_128x1_tp,
    quant_grouped_128x1_kmajor,
)
from prime_rl.trainer.models.layers import fp8_grouped_gemm, fp8_linear
from prime_rl.utils.logger import get_logger

PatternRegistrar = Callable[[PatternMatcherPass, FP8WgradQuant, QuantizerDivision, QuantizerDivision], None]

FP8_MM = torch.ops.prime_rl.fp8_blockwise_mm.default
FP8_MM_BACKWARD = torch.ops.prime_rl.fp8_blockwise_mm_backward.default
GROUPED_FP8_GEMM = torch.ops.prime_rl.grouped_fp8_gemm.default
GROUPED_FP8_GEMM_BACKWARD = torch.ops.prime_rl.grouped_fp8_gemm_backward.default
# Tuned on H200 against the Triton quantizers, keyed by whether the 128-element groups are contiguous or strided in memory.
FP8_GROUP_QUANT_LAUNCH_CONFIGS = {
    "strided": {"XBLOCK": 128, "num_warps": 4},
    "contiguous": {"XBLOCK": 16, "num_warps": 4},
}


def register_fp8_linear_patterns(
    patterns: PatternMatcherPass,
    wgrad_quant: FP8WgradQuant,
    forward_division: QuantizerDivision,
    backward_division: QuantizerDivision,
) -> None:
    """Register rewrites of `prime_rl::fp8_blockwise_mm` and its backward into torch-op quants and pre-quantized GEMMs."""

    @register_graph_pattern(CallFunction(FP8_MM, Arg(), Arg(), Arg()), pass_dict=patterns)
    def rewrite_fp8_mm(match: Match, x: torch.fx.Node, weight: torch.fx.Node, block_size: int) -> None:
        assert block_size == BLOCK, f"the torch-op quantizers use {BLOCK}-element groups, got block_size={block_size}"

        def replacement(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            x_fp8, x_scale = quant_1x128(x.reshape(-1, x.shape[-1]), forward_division)
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
                grad_output_fp8, grad_output_scale = quant_1x128(grad_output_2d, backward_division)
                grad_x = torch.ops.prime_rl.fp8_blockwise_mm_dgrad_prequant(
                    grad_output_fp8, grad_output_scale, weight, block_size
                ).reshape(x.shape)
            if needs_grad_weight and wgrad_quant == "torch":
                grad_output_t_fp8, grad_output_t_scale = quant_128x1_tp(grad_output_2d, backward_division)
                x_t_fp8, x_t_scale = quant_128x1_tp(x_2d, backward_division)
                grad_weight = torch.ops.prime_rl.fp8_blockwise_mm_wgrad_prequant(
                    grad_output_t_fp8, grad_output_t_scale, x_t_fp8, x_t_scale, weight
                )
            elif needs_grad_weight:
                grad_weight = torch.ops.prime_rl.fp8_blockwise_mm_wgrad(grad_output_2d, x_2d, weight, block_size)
            return grad_x, grad_weight

        match.replace_by_example(replacement, [grad_output, x, weight])


def register_fp8_grouped_gemm_patterns(
    patterns: PatternMatcherPass,
    wgrad_quant: FP8WgradQuant,
    forward_division: QuantizerDivision,
    backward_division: QuantizerDivision,
) -> None:
    """Register rewrites of `prime_rl::grouped_fp8_gemm` and its backward into torch-op quants and pre-quantized grouped GEMMs."""

    @register_graph_pattern(CallFunction(GROUPED_FP8_GEMM, Arg(), Arg(), Arg()), pass_dict=patterns)
    def rewrite_grouped_fp8_gemm(match: Match, x: torch.fx.Node, weight: torch.fx.Node, offs: torch.fx.Node) -> None:
        def replacement(x: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
            x_fp8, x_scale = quant_1x128(x, forward_division)
            return torch.ops.prime_rl.grouped_fp8_gemm_prequant(x_fp8, x_scale, weight, offs)

        match.replace_by_example(replacement, [x, weight, offs])

    @register_graph_pattern(
        CallFunction(GROUPED_FP8_GEMM_BACKWARD, Arg(), Arg(), Arg(), Arg(), Arg(), Arg(), Arg()), pass_dict=patterns
    )
    def rewrite_grouped_fp8_gemm_backward(
        match: Match,
        grad_output: torch.fx.Node,
        x: torch.fx.Node,
        weight: torch.fx.Node,
        offs: torch.fx.Node,
        needs_grad_x: bool,
        needs_grad_weight: bool,
        grad_weight_transposed: bool,
    ) -> None:
        def replacement(
            grad_output: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            grad_x = torch.empty_like(x)
            grad_weight = torch.empty_like(weight)
            if needs_grad_x:
                grad_output_fp8, grad_output_scale = quant_1x128(grad_output, backward_division)
                grad_x = torch.ops.prime_rl.grouped_fp8_gemm_dgrad_prequant(
                    grad_output_fp8, grad_output_scale, weight, offs
                )
            if needs_grad_weight and wgrad_quant == "torch":
                x_t_fp8, x_t_scale = quant_grouped_128x1_kmajor(x, offs, backward_division)
                grad_output_t_fp8, grad_output_t_scale = quant_grouped_128x1_kmajor(
                    grad_output, offs, backward_division
                )
                grad_weight = torch.ops.prime_rl.grouped_fp8_gemm_wgrad_prequant(
                    x_t_fp8, x_t_scale, grad_output_t_fp8, grad_output_t_scale, weight, offs, grad_weight_transposed
                )
            elif needs_grad_weight:
                grad_weight = torch.ops.prime_rl.grouped_fp8_gemm_wgrad(
                    grad_output, x, weight, offs, grad_weight_transposed
                )
            return grad_x, grad_weight

        match.replace_by_example(replacement, [grad_output, x, weight, offs])


PATTERN_REGISTRARS: dict[FP8QuantPassTarget, PatternRegistrar] = {
    "linear": register_fp8_linear_patterns,
    "grouped_gemm": register_fp8_grouped_gemm_patterns,
}
REWRITTEN_OPS = (FP8_MM, FP8_MM_BACKWARD, GROUPED_FP8_GEMM, GROUPED_FP8_GEMM_BACKWARD)


def _quantizer_divisions(division: FP8QuantDivision) -> tuple[QuantizerDivision, QuantizerDivision]:
    """Division of the forward activation casts, which the forward GEMM rewrites emit in both graphs, and of the gradient and weight-gradient casts."""
    if division == "forward_exact":
        return "exact", "approximate"
    return division, division


@functools.cache
def _patterns(
    wgrad_quants: tuple[tuple[FP8QuantPassTarget, FP8WgradQuant], ...], division: FP8QuantDivision
) -> PatternMatcherPass:
    name = "_".join(f"{target}_{wgrad_quant}" for target, wgrad_quant in wgrad_quants)
    patterns = PatternMatcherPass(pass_name=f"fp8_quant_pass_{name}_{division}")
    forward_division, backward_division = _quantizer_divisions(division)
    for target, wgrad_quant in wgrad_quants:
        PATTERN_REGISTRARS[target](patterns, wgrad_quant, forward_division, backward_division)
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

    def __init__(
        self,
        wgrad_quant: FP8WgradQuant,
        targets: Sequence[FP8QuantPassTarget],
        division: FP8QuantDivision,
        grouped_gemm_wgrad_quant: FP8WgradQuant | None = None,
    ) -> None:
        wgrad_quant_overrides = {"grouped_gemm": grouped_gemm_wgrad_quant}
        self.wgrad_quants = tuple(
            (target, wgrad_quant_overrides.get(target) or wgrad_quant) for target in sorted(set(targets))
        )
        self.division = division

    def __call__(self, graph: torch.fx.Graph) -> None:
        nodes_before = set(graph.nodes)
        ops_before = Counter(
            str(node.target) for node in graph.nodes if node.op == "call_function" and node.target in REWRITTEN_OPS
        )
        rewritten = _patterns(self.wgrad_quants, self.division).apply(graph)
        if not rewritten:
            return
        deduped = _dedupe(graph, set(graph.nodes) - nodes_before)
        graph.eliminate_dead_code()
        get_logger().info(
            f"FP8 quant pass (wgrad quant by target {dict(self.wgrad_quants)}, {self.division} division) "
            f"rewrote {rewritten} ops "
            f"{dict(ops_before)}, deduplicated {deduped} repeated quant nodes"
        )

    def uuid(self) -> bytes:
        source_files = {__file__, fp8_torch_quant.__file__, fp8_linear.__file__, fp8_grouped_gemm.__file__}
        source_files.update(inspect.getfile(register) for register in PATTERN_REGISTRARS.values())
        wgrad_quants = "-".join(f"{target}:{wgrad_quant}" for target, wgrad_quant in self.wgrad_quants)
        return get_hash_for_files(tuple(sorted(source_files)), extra=f"{wgrad_quants}-{self.division}")


def _fp8_group_quant_layout(
    features: SIMDKernelFeatures, kernel_kwargs: dict[str, Any]
) -> Literal["contiguous", "strided"] | None:
    """Layout of the groups an FP8 group quant kernel reduces, by its reads' stride along the reduction, or None for any other kernel."""
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
    contiguous_reads = {
        dep.stride1_for_last_dim(result_for_complex_expression=False)
        for node in reductions
        for dep in node.read_writes.reads
        if isinstance(dep, MemoryDep) and dep.var_names and dep.var_names[-1] in dep.index.free_symbols
    }
    if contiguous_reads == {True}:
        return "contiguous"
    if contiguous_reads == {False}:
        return "strided"
    return None


def _is_group_amax(node: BaseSchedulerNode) -> bool:
    """Whether `node` has reductions and all are 128-element maxes, like the FP8 group amax."""
    reductions = [reduction for reduction in node.get_nodes() if reduction.is_reduction()]
    return bool(reductions) and all(
        reduction.node.get_reduction_type() == "max"
        and V.graph.sizevars.statically_known_equals(sympy_product(reduction.node.get_reduction_size()), BLOCK)
        for reduction in reductions
    )


def _reads_unbacked_output(producer: BaseSchedulerNode, consumer: BaseSchedulerNode) -> bool:
    produced = producer.get_buffer_names()
    return any(dep.name in produced and dep.has_unbacked_symbols() for dep in consumer.read_writes.reads)


class Fp8QuantFusionChoices(InductorChoices):
    """Inductor heuristics that fuse an FP8 group amax with its producer, scale and cast even when its rows are unbacked, like the routed-expert rows sized by the all-to-all, other fusions unchanged."""

    @staticmethod
    def can_fuse(
        scheduler: Scheduler, node1: BaseSchedulerNode, node2: BaseSchedulerNode, shared_data_score: int
    ) -> bool:
        # Inductor scores shared data from size hints, which are zero for unbacked sizes, and never fuses a reduction without it.
        if (
            shared_data_score == 0
            and (_is_group_amax(node1) or _is_group_amax(node2))
            and _reads_unbacked_output(node1, node2)
        ):
            shared_data_score = 1
        return InductorChoices.can_fuse(scheduler, node1, node2, shared_data_score)

    def uuid(self) -> str:
        return get_hash_for_files((__file__, fp8_torch_quant.__file__)).hex()


class Fp8GroupQuantChoices(Fp8QuantFusionChoices):
    """Inductor heuristics that also compile FP8 group quant kernels as persistent reductions with a fixed launch config, other kernels unchanged."""

    def triton_kernel_kwargs(
        self,
        kernel_cls: type[TritonKernel],
        features: SIMDKernelFeatures,
        groups: list[sympy.Expr],
        kernel_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        layout = _fp8_group_quant_layout(features, kernel_kwargs)
        if layout is None:
            return kernel_kwargs
        fixed_config = FixedTritonConfig(dict(FP8_GROUP_QUANT_LAUNCH_CONFIGS[layout]))
        return {**kernel_kwargs, "override_persistent_reduction": True, "fixed_config": fixed_config}


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
