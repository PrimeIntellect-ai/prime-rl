"""Inductor post-grad pass that splits DeepGEMM FP8 linear ops into torch-op quantizers and pre-quantized GEMMs."""

from __future__ import annotations

import functools
import sys
import threading
from collections import Counter
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch._inductor import config as inductor_config
from torch._inductor.choices import InductorChoices
from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files
from torch._inductor.pattern_matcher import Arg, CallFunction, Match, PatternMatcherPass, register_graph_pattern
from torch._inductor.virtualized import V

import prime_rl.trainer.models.layers.fp8_linear  # noqa: F401
from prime_rl.trainer.models.kernels.fp8_utils import (
    per_block_cast_to_fp8_tp_triton,
    per_block_cast_to_fp8_triton,
    per_token_cast_to_fp8_tp_triton,
    ue8m0_for_device,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quant_torch  # noqa: E402
from quant_torch import quant_1x128, quant_128x1_tp  # noqa: E402

FP8_MM = torch.ops.prime_rl.fp8_blockwise_mm.default
FP8_MM_BACKWARD = torch.ops.prime_rl.fp8_blockwise_mm_backward.default
SM90_ONLY = "the torch-op quantizer emits fp32 scales, which DeepGEMM accepts on SM90 only"


@torch.library.custom_op("prime_rl::fp8_blockwise_mm_prequant", mutates_args=())
def fp8_blockwise_mm_prequant(
    x_fp8: torch.Tensor, x_scale: torch.Tensor, weight: torch.Tensor, block_size: int
) -> torch.Tensor:
    import deep_gemm

    use_ue8m0 = ue8m0_for_device(x_fp8.device)
    assert not use_ue8m0, SM90_ONLY
    weight_fp8 = per_block_cast_to_fp8_triton(weight, use_ue8m0, block_size)
    out = torch.empty((x_fp8.size(0), weight.size(0)), device=x_fp8.device, dtype=torch.bfloat16)
    deep_gemm.fp8_gemm_nt((x_fp8, x_scale), weight_fp8, out)
    return out


@fp8_blockwise_mm_prequant.register_fake
def _fp8_blockwise_mm_prequant_fake(
    x_fp8: torch.Tensor, x_scale: torch.Tensor, weight: torch.Tensor, block_size: int
) -> torch.Tensor:
    return x_fp8.new_empty((x_fp8.shape[0], weight.shape[0]), dtype=torch.bfloat16)


@torch.library.custom_op("prime_rl::fp8_blockwise_mm_dgrad_prequant", mutates_args=())
def fp8_blockwise_mm_dgrad_prequant(
    grad_output_fp8: torch.Tensor, grad_output_scale: torch.Tensor, weight: torch.Tensor, block_size: int
) -> torch.Tensor:
    import deep_gemm

    use_ue8m0 = ue8m0_for_device(weight.device)
    assert not use_ue8m0, SM90_ONLY
    weight_dx_fp8 = per_block_cast_to_fp8_tp_triton(weight, use_ue8m0, block_size)
    grad_x = weight.new_empty((grad_output_fp8.size(0), weight.size(1)))
    deep_gemm.fp8_gemm_nt((grad_output_fp8, grad_output_scale), weight_dx_fp8, grad_x)
    return grad_x


@fp8_blockwise_mm_dgrad_prequant.register_fake
def _fp8_blockwise_mm_dgrad_prequant_fake(
    grad_output_fp8: torch.Tensor, grad_output_scale: torch.Tensor, weight: torch.Tensor, block_size: int
) -> torch.Tensor:
    return weight.new_empty((grad_output_fp8.shape[0], weight.shape[1]))


def _wgrad_gemm(
    grad_output_t: tuple[torch.Tensor, torch.Tensor], x_t: tuple[torch.Tensor, torch.Tensor], weight: torch.Tensor
) -> torch.Tensor:
    import deep_gemm

    grad_weight_fp32 = torch.zeros(weight.shape, device=weight.device, dtype=torch.float32)
    deep_gemm.fp8_gemm_nt(
        grad_output_t, x_t, grad_weight_fp32, c=grad_weight_fp32, recipe=(1, 1, 128), compiled_dims="mn"
    )
    return grad_weight_fp32.to(weight.dtype)


@torch.library.custom_op("prime_rl::fp8_blockwise_mm_wgrad_prequant", mutates_args=())
def fp8_blockwise_mm_wgrad_prequant(
    grad_output_t_fp8: torch.Tensor,
    grad_output_t_scale: torch.Tensor,
    x_t_fp8: torch.Tensor,
    x_t_scale: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    assert not ue8m0_for_device(weight.device), SM90_ONLY
    return _wgrad_gemm((grad_output_t_fp8, grad_output_t_scale), (x_t_fp8, x_t_scale), weight)


@fp8_blockwise_mm_wgrad_prequant.register_fake
def _fp8_blockwise_mm_wgrad_prequant_fake(
    grad_output_t_fp8: torch.Tensor,
    grad_output_t_scale: torch.Tensor,
    x_t_fp8: torch.Tensor,
    x_t_scale: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return weight.new_empty(weight.shape)


@torch.library.custom_op("prime_rl::fp8_blockwise_mm_wgrad", mutates_args=())
def fp8_blockwise_mm_wgrad(
    grad_output: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, block_size: int
) -> torch.Tensor:
    """Weight gradient with the Triton 128x1 quantizers inside, as `prime_rl::fp8_blockwise_mm_backward` computes it."""
    use_ue8m0 = ue8m0_for_device(weight.device)
    num_tokens = grad_output.size(0)
    padded_tokens = (num_tokens + block_size - 1) // block_size * block_size
    if padded_tokens != num_tokens:
        grad_output = F.pad(grad_output, (0, 0, 0, padded_tokens - num_tokens))
        x = F.pad(x, (0, 0, 0, padded_tokens - num_tokens))
    grad_output_t = per_token_cast_to_fp8_tp_triton(grad_output, use_ue8m0, block_size)
    x_t = per_token_cast_to_fp8_tp_triton(x, use_ue8m0, block_size)
    return _wgrad_gemm(grad_output_t, x_t, weight)


@fp8_blockwise_mm_wgrad.register_fake
def _fp8_blockwise_mm_wgrad_fake(
    grad_output: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, block_size: int
) -> torch.Tensor:
    return weight.new_empty(weight.shape)


PASS_RECORDS: list[dict] = []


@functools.cache
def _build_patterns(wgrad_quant: str) -> tuple[PatternMatcherPass, list[torch.fx.Node]]:
    """Build the rewrite patterns for one weight-gradient quantizer choice, plus the list their replacements add to."""
    patterns = PatternMatcherPass(pass_name=f"fp8_quant_pass_{wgrad_quant}")
    added_nodes: list[torch.fx.Node] = []

    @register_graph_pattern(CallFunction(FP8_MM, Arg(), Arg(), Arg()), pass_dict=patterns)
    def rewrite_fp8_mm(match: Match, x: torch.fx.Node, weight: torch.fx.Node, block_size: int) -> None:
        assert block_size == quant_torch.BLOCK

        def replacement(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            x_fp8, x_scale = quant_1x128(x.reshape(-1, x.shape[-1]))
            out = torch.ops.prime_rl.fp8_blockwise_mm_prequant(x_fp8, x_scale, weight, block_size)
            return out.reshape(*x.shape[:-1], weight.shape[0])

        added_nodes.extend(match.replace_by_example(replacement, [x, weight]))

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
        assert block_size == quant_torch.BLOCK

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

        added_nodes.extend(match.replace_by_example(replacement, [grad_output, x, weight]))

    return patterns, added_nodes


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


def _count_targets(graph: torch.fx.Graph, targets: tuple) -> Counter:
    return Counter(str(node.target) for node in graph.nodes if node.op == "call_function" and node.target in targets)


class Fp8QuantPass(CustomGraphPass):
    """Rewrites FP8 linear ops into torch-op quants and pre-quantized GEMMs, with wgrad quants in torch or Triton."""

    def __init__(self, wgrad_quant: Literal["torch", "triton"] = "torch") -> None:
        self.wgrad_quant = wgrad_quant

    def __call__(self, graph: torch.fx.Graph) -> None:
        amax = (torch.ops.aten.amax.default,)
        before = _count_targets(graph, (FP8_MM, FP8_MM_BACKWARD))
        patterns, added_nodes = _build_patterns(self.wgrad_quant)
        added_nodes.clear()
        rewritten = patterns.apply(graph)
        quants_before_dedupe = _count_targets(graph, amax)["aten.amax.default"]
        removed = _dedupe(graph, set(added_nodes))
        added_nodes.clear()
        graph.eliminate_dead_code()
        new_ops = (
            torch.ops.prime_rl.fp8_blockwise_mm_prequant.default,
            torch.ops.prime_rl.fp8_blockwise_mm_dgrad_prequant.default,
            torch.ops.prime_rl.fp8_blockwise_mm_wgrad_prequant.default,
            torch.ops.prime_rl.fp8_blockwise_mm_wgrad.default,
        )
        PASS_RECORDS.append(
            {
                "graph": "backward" if before[str(FP8_MM_BACKWARD)] else "forward",
                "thread": threading.current_thread().name,
                "wgrad_quant": self.wgrad_quant,
                "rewritten": rewritten,
                "ops_before": dict(before),
                "ops_after": dict(_count_targets(graph, (FP8_MM, FP8_MM_BACKWARD) + new_ops)),
                "quants_before_dedupe": quants_before_dedupe,
                "quants_after_dedupe": _count_targets(graph, amax)["aten.amax.default"],
                "deduped_nodes": removed,
                "config_seen": {
                    "post_grad_custom_pre_pass": type(inductor_config.post_grad_custom_pre_pass).__name__,
                    "nested_reduction": inductor_config.triton.nested_reduction,
                    "mix_order_reduction_non_strict_mode": inductor_config.triton.mix_order_reduction_non_strict_mode,
                    "emulate_precision_casts": inductor_config.emulate_precision_casts,
                    "inductor_choices": type(V.choices).__name__,
                },
            }
        )

    def uuid(self) -> bytes:
        return get_hash_for_files((__file__, quant_torch.__file__), extra=self.wgrad_quant)


class PersistentGroupReductionChoices(InductorChoices):
    """Inductor heuristics that make 128-element reductions persistent even when they are not along the inner axis."""

    @staticmethod
    def should_use_persistent_reduction(features: SIMDKernelFeatures, cooperative_reduction: bool) -> bool:
        if inductor_config.triton.persistent_reductions and V.graph.sizevars.statically_known_equals(
            features.reduction_numel, quant_torch.BLOCK
        ):
            return True
        return InductorChoices.should_use_persistent_reduction(features, cooperative_reduction)

    def uuid(self) -> str:
        return "persistent-group-reduction-v1"
