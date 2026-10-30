"""Torch-op quantizers, pre-quantized ops and post-grad rewrites for DeepGEMM's FP8 routed-expert grouped GEMMs (SM90)."""

from __future__ import annotations

import functools
import sys
import threading
from collections import Counter
from pathlib import Path
from typing import Literal

import torch
from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files
from torch._inductor.pattern_matcher import Arg, CallFunction, Match, PatternMatcherPass, register_graph_pattern

import prime_rl.trainer.models.layers.fp8_grouped_gemm  # noqa: F401
from prime_rl.trainer.models.kernels.fp8_utils import (
    GROUP_ALIGNMENT,
    build_grouped_layout,
    grouped_per_block_cast_to_fp8_triton,
    ue8m0_for_device,
)
from prime_rl.trainer.models.layers.fp8_grouped_gemm import _compute_grad_weight, _empty_grad_weight, _grouped_layout

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quant_torch  # noqa: E402
from quant_torch import AMAX_FLOOR, BLOCK, FP8_MAX, quant_1x128  # noqa: E402

GROUPED_FP8_GEMM = torch.ops.prime_rl.grouped_fp8_gemm.default
GROUPED_FP8_GEMM_BACKWARD = torch.ops.prime_rl.grouped_fp8_gemm_backward.default
WORD_BYTES = 8
WgradQuant = Literal["torch", "torch-inline", "triton"]
SM90_ONLY = (
    "the torch-op quantizers emit fp32 scales and the SM90 K-major wgrad layout, which DeepGEMM takes on SM90 only"
)


def stored(x: torch.Tensor) -> torch.Tensor:
    """Return `x` through a pair of no-op dtype views, which make Inductor store it to memory instead of recomputing it in each reader."""
    return x.contiguous().view(torch.int32).view(x.dtype)


def quant_grouped_128x1_kmajor(
    x: torch.Tensor, offs: torch.Tensor, store_input: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel cast in 128-row blocks in `grouped_per_channel_cast_to_fp8_sm90_kmajor_triton`'s layout, with rows past `offs[-1]` as one extra trailing group."""
    rows, channels = x.shape
    row_blocks = rows // BLOCK
    x_float = (stored(x) if store_input else x).float()
    amax = x_float.reshape(row_blocks, BLOCK, channels).abs().amax(dim=1).clamp(min=AMAX_FLOOR)
    scale_t = (amax / FP8_MAX).t()
    x_t_blocks = x_float.t().reshape(channels, row_blocks, BLOCK)
    x_fp8_t = (x_t_blocks / scale_t.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    # The int64 view realizes x_fp8_t in the amax kernel, so the group scatter below moves coalesced 8-byte words.
    x_fp8_t_words = x_fp8_t.contiguous().view(torch.int64)

    group_ends = torch.cat((offs, offs.new_full((1,), rows))).long()
    group_starts = torch.cat((offs.new_zeros(1), offs)).long()
    word = torch.arange(rows // WORD_BYTES, device=x.device)
    word_group = torch.bucketize(word * WORD_BYTES, group_ends, right=True)
    group_start = group_starts[word_group] // WORD_BYTES
    group_size = group_ends[word_group] // WORD_BYTES - group_start
    channel = torch.arange(channels, device=x.device).unsqueeze(-1)
    destination = group_start * channels + channel * group_size + (word - group_start)
    destination = destination.view(channels, row_blocks, BLOCK // WORD_BYTES)
    out = torch.index_put(x_fp8_t_words.new_empty(rows * channels // WORD_BYTES), (destination,), x_fp8_t_words)
    return out.view(torch.float8_e4m3fn), scale_t


def _wgrad_ks(offs: torch.Tensor, rows: int) -> tuple[torch.Tensor, list[int]]:
    torch._assert_async(torch.all(offs % GROUP_ALIGNMENT == 0))
    ks_tensor = build_grouped_layout(offs, rows)[2]
    return ks_tensor, ks_tensor.tolist()


@torch.library.custom_op("prime_rl::grouped_fp8_gemm_prequant", mutates_args=())
def grouped_fp8_gemm_prequant(
    x_fp8: torch.Tensor, x_scale: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    import deep_gemm

    use_ue8m0 = ue8m0_for_device(x_fp8.device)
    assert not use_ue8m0, SM90_ONLY
    assert x_fp8.size(0) % GROUP_ALIGNMENT == 0, f"expected rows padded to {GROUP_ALIGNMENT}, got {x_fp8.size(0)}"
    torch._assert_async(torch.all(offs % GROUP_ALIGNMENT == 0))
    grouped_layout = build_grouped_layout(offs, x_fp8.size(0))[0]
    weight_fp8 = grouped_per_block_cast_to_fp8_triton(weight.transpose(1, 2), use_ue8m0, GROUP_ALIGNMENT)
    out = torch.empty((x_fp8.size(0), weight.size(2)), device=x_fp8.device, dtype=torch.bfloat16)
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous((x_fp8, x_scale), weight_fp8, out, grouped_layout, use_psum_layout=False)
    return out


@grouped_fp8_gemm_prequant.register_fake
def _grouped_fp8_gemm_prequant_fake(
    x_fp8: torch.Tensor, x_scale: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    return x_fp8.new_empty((x_fp8.shape[0], weight.shape[2]), dtype=torch.bfloat16)


@torch.library.custom_op("prime_rl::grouped_fp8_gemm_dgrad_prequant", mutates_args=())
def grouped_fp8_gemm_dgrad_prequant(
    grad_output_fp8: torch.Tensor, grad_output_scale: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    import deep_gemm

    use_ue8m0 = ue8m0_for_device(weight.device)
    assert not use_ue8m0, SM90_ONLY
    rows = grad_output_fp8.size(0)
    assert rows % GROUP_ALIGNMENT == 0, f"expected rows padded to {GROUP_ALIGNMENT}, got {rows}"
    torch._assert_async(torch.all(offs % GROUP_ALIGNMENT == 0))
    grouped_layout = build_grouped_layout(offs, rows)[0]
    weight_dx_fp8 = grouped_per_block_cast_to_fp8_triton(weight, use_ue8m0, GROUP_ALIGNMENT)
    grad_x = torch.empty((rows, weight.size(1)), device=weight.device, dtype=torch.bfloat16)
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (grad_output_fp8, grad_output_scale), weight_dx_fp8, grad_x, grouped_layout, use_psum_layout=False
    )
    return grad_x


@grouped_fp8_gemm_dgrad_prequant.register_fake
def _grouped_fp8_gemm_dgrad_prequant_fake(
    grad_output_fp8: torch.Tensor, grad_output_scale: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    return grad_output_fp8.new_empty((grad_output_fp8.shape[0], weight.shape[1]), dtype=torch.bfloat16)


@torch.library.custom_op("prime_rl::grouped_fp8_gemm_wgrad_prequant", mutates_args=())
def grouped_fp8_gemm_wgrad_prequant(
    x_t_fp8: torch.Tensor,
    x_t_scale: torch.Tensor,
    grad_output_t_fp8: torch.Tensor,
    grad_output_t_scale: torch.Tensor,
    weight: torch.Tensor,
    offs: torch.Tensor,
    grad_weight_transposed: bool,
) -> torch.Tensor:
    """Weight gradient from `quant_grouped_128x1_kmajor` outputs over all rows, sliced to the grouped rows here."""
    import deep_gemm

    assert not ue8m0_for_device(weight.device), SM90_ONLY
    in_features, out_features = weight.shape[1], weight.shape[2]
    rows = x_t_scale.size(1) * GROUP_ALIGNMENT
    ks_tensor, aligned_ms = _wgrad_ks(offs, rows)
    grouped_rows = sum(aligned_ms)
    grouped_blocks = grouped_rows // GROUP_ALIGNMENT
    x_t = (x_t_fp8[: grouped_rows * in_features], x_t_scale[:, :grouped_blocks])
    grad_output_t = (grad_output_t_fp8[: grouped_rows * out_features], grad_output_t_scale[:, :grouped_blocks])
    if grad_weight_transposed:
        lhs, rhs, grad_shape = grad_output_t, x_t, weight.transpose(1, 2).shape
    else:
        lhs, rhs, grad_shape = x_t, grad_output_t, weight.shape
    grad_weight = torch.zeros(grad_shape, device=weight.device, dtype=torch.float32)
    deep_gemm.k_grouped_fp8_gemm_nt_contiguous(lhs, rhs, grad_weight, aligned_ms, ks_tensor, grad_weight)
    grad_weight = grad_weight.to(weight.dtype)
    return grad_weight.transpose(1, 2) if grad_weight_transposed else grad_weight


@grouped_fp8_gemm_wgrad_prequant.register_fake
def _grouped_fp8_gemm_wgrad_prequant_fake(
    x_t_fp8: torch.Tensor,
    x_t_scale: torch.Tensor,
    grad_output_t_fp8: torch.Tensor,
    grad_output_t_scale: torch.Tensor,
    weight: torch.Tensor,
    offs: torch.Tensor,
    grad_weight_transposed: bool,
) -> torch.Tensor:
    return _empty_grad_weight(weight, grad_weight_transposed)


@torch.library.custom_op("prime_rl::grouped_fp8_gemm_wgrad", mutates_args=())
def grouped_fp8_gemm_wgrad(
    grad_output: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor, grad_weight_transposed: bool
) -> torch.Tensor:
    """Weight gradient with the Triton per-channel quantizers inside, as `prime_rl::grouped_fp8_gemm_backward` computes it."""
    _, block_to_group, ks_tensor, starts_tensor, actual_ms_tensor, block_starts_tensor = _grouped_layout(x, offs)
    aligned_ms = ks_tensor.tolist()
    grouped_rows = sum(aligned_ms)
    return _compute_grad_weight(
        x,
        grad_output.contiguous(),
        weight,
        grouped_rows,
        block_to_group[: grouped_rows // GROUP_ALIGNMENT],
        ks_tensor,
        starts_tensor,
        actual_ms_tensor,
        block_starts_tensor,
        aligned_ms,
        grad_weight_transposed,
    )


@grouped_fp8_gemm_wgrad.register_fake
def _grouped_fp8_gemm_wgrad_fake(
    grad_output: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor, grad_weight_transposed: bool
) -> torch.Tensor:
    return _empty_grad_weight(weight, grad_weight_transposed)


PASS_RECORDS: list[dict] = []
NEW_OPS = (
    torch.ops.prime_rl.grouped_fp8_gemm_prequant.default,
    torch.ops.prime_rl.grouped_fp8_gemm_dgrad_prequant.default,
    torch.ops.prime_rl.grouped_fp8_gemm_wgrad_prequant.default,
    torch.ops.prime_rl.grouped_fp8_gemm_wgrad.default,
)


def register_grouped_patterns(patterns: PatternMatcherPass, wgrad_quant: WgradQuant) -> None:
    """Register the rewrites of `grouped_fp8_gemm` and its backward into `patterns`, with wgrad quants in torch (inputs stored or inlined) or Triton."""

    @register_graph_pattern(CallFunction(GROUPED_FP8_GEMM, Arg(), Arg(), Arg()), pass_dict=patterns)
    def rewrite_grouped_fp8_gemm(match: Match, x: torch.fx.Node, weight: torch.fx.Node, offs: torch.fx.Node) -> None:
        def replacement(x: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
            x_fp8, x_scale = quant_1x128(x)
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
            if needs_grad_x:
                grad_output_fp8, grad_output_scale = quant_1x128(grad_output)
                grad_x = torch.ops.prime_rl.grouped_fp8_gemm_dgrad_prequant(
                    grad_output_fp8, grad_output_scale, weight, offs
                )
            else:
                grad_x = x.new_empty(x.shape)
            if needs_grad_weight and wgrad_quant != "triton":
                store_input = wgrad_quant == "torch"
                x_t_fp8, x_t_scale = quant_grouped_128x1_kmajor(x, offs, store_input)
                grad_output_t_fp8, grad_output_t_scale = quant_grouped_128x1_kmajor(grad_output, offs, store_input)
                grad_weight = torch.ops.prime_rl.grouped_fp8_gemm_wgrad_prequant(
                    x_t_fp8, x_t_scale, grad_output_t_fp8, grad_output_t_scale, weight, offs, grad_weight_transposed
                )
            elif needs_grad_weight:
                grad_weight = torch.ops.prime_rl.grouped_fp8_gemm_wgrad(
                    grad_output, x, weight, offs, grad_weight_transposed
                )
            else:
                grad_weight = _empty_grad_weight(weight, grad_weight_transposed)
            return grad_x, grad_weight

        match.replace_by_example(replacement, [grad_output, x, weight, offs])


@functools.cache
def _patterns(wgrad_quant: str) -> PatternMatcherPass:
    patterns = PatternMatcherPass(pass_name=f"grouped_fp8_quant_pass_{wgrad_quant}")
    register_grouped_patterns(patterns, wgrad_quant)
    return patterns


def _count_targets(graph: torch.fx.Graph, targets: tuple) -> Counter:
    return Counter(str(node.target) for node in graph.nodes if node.op == "call_function" and node.target in targets)


class GroupedFp8QuantPass(CustomGraphPass):
    """Rewrites FP8 grouped GEMM ops into torch-op quants and pre-quantized grouped GEMMs, with wgrad quants in torch or Triton."""

    def __init__(self, wgrad_quant: WgradQuant = "torch") -> None:
        self.wgrad_quant = wgrad_quant

    def __call__(self, graph: torch.fx.Graph) -> None:
        old_ops = (GROUPED_FP8_GEMM, GROUPED_FP8_GEMM_BACKWARD)
        before = _count_targets(graph, old_ops)
        rewritten = _patterns(self.wgrad_quant).apply(graph)
        graph.eliminate_dead_code()
        PASS_RECORDS.append(
            {
                "graph": "backward" if before[str(GROUPED_FP8_GEMM_BACKWARD)] else "forward",
                "thread": threading.current_thread().name,
                "wgrad_quant": self.wgrad_quant,
                "rewritten": rewritten,
                "ops_before": dict(before),
                "ops_after": dict(_count_targets(graph, old_ops + NEW_OPS)),
                "amax_after": _count_targets(graph, (torch.ops.aten.amax.default,))["aten.amax.default"],
            }
        )

    def uuid(self) -> bytes:
        return get_hash_for_files((__file__, quant_torch.__file__), extra=self.wgrad_quant)
