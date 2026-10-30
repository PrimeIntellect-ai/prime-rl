"""Check the grouped FP8 torch-op quantizers, pre-quantized ops and pass, then run a routed-expert toy block with and without the pass."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import re
import statistics
import sys
from pathlib import Path

import torch
import triton
from torch import nn
from torch._inductor.choices import InductorChoices
from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
from torch._inductor.graph import GraphLowering
from torch._inductor.utils import run_and_get_code
from torch._inductor.virtualized import V
from torch.autograd import DeviceType

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grouped_prequant import PASS_RECORDS, GroupedFp8QuantPass, quant_grouped_128x1_kmajor  # noqa: E402
from quant_torch import BLOCK, quant_1x128  # noqa: E402

from prime_rl.configs.trainer import ActivationCheckpointConfig  # noqa: E402
from prime_rl.trainer.activation_checkpointing import get_activation_checkpoint_wrapper  # noqa: E402
from prime_rl.trainer.distributed.token_dispatcher import _GatherPermutedRows  # noqa: E402
from prime_rl.trainer.models.kernels.fp8_utils import (  # noqa: E402
    GROUP_ALIGNMENT,
    build_grouped_layout,
    grouped_per_channel_cast_to_fp8_sm90_kmajor_triton,
    grouped_per_token_cast_to_fp8_triton,
)
from prime_rl.trainer.models.layers.activations import ClampedSilu  # noqa: E402
from prime_rl.trainer.models.layers.fp8_grouped_gemm import grouped_fp8_gemm  # noqa: E402

SWIGLU_LIMIT = 10.0
WEIGHT_STD = 0.02
ARMS = {"nopass": None, "pass-torch": "torch", "pass-torch-inline": "torch-inline", "pass-triton": "triton"}


class PersistentGroupReductionChoices(InductorChoices):
    """Inductor heuristics that make 128-element reductions persistent even when they are not along the inner axis."""

    @staticmethod
    def should_use_persistent_reduction(features: SIMDKernelFeatures, cooperative_reduction: bool) -> bool:
        from torch._inductor import config as inductor_config

        if inductor_config.triton.persistent_reductions and V.graph.sizevars.statically_known_equals(
            features.reduction_numel, BLOCK
        ):
            return True
        return InductorChoices.should_use_persistent_reduction(features, cooperative_reduction)

    def uuid(self) -> str:
        return "persistent-group-reduction-v1"


def grouped_rows(counts: list[int], tail_rows: int, channels: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Random bf16 rows for 128-aligned groups of `counts` rows (zero allowed) plus a tail, and their int32 offsets."""
    assert all(count % GROUP_ALIGNMENT == 0 for count in counts)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    rows = sum(counts) + tail_rows
    magnitudes = torch.logspace(-3, 3, rows, device="cuda").unsqueeze(-1)
    x = (torch.randn(rows, channels, device="cuda", generator=generator) * magnitudes).to(torch.bfloat16)
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    return x, offs


def ulp_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float().contiguous().view(torch.int32).long() - b.float().contiguous().view(torch.int32).long()).abs()


def compare_quant(fp8: torch.Tensor, scale: torch.Tensor, ref_fp8: torch.Tensor, ref_scale: torch.Tensor) -> dict:
    assert fp8.shape == ref_fp8.shape and scale.shape == ref_scale.shape, (
        fp8.shape,
        ref_fp8.shape,
        scale.shape,
        ref_scale.shape,
    )
    scale_ulps = ulp_diff(scale, ref_scale)
    fp8_bytes_differ = (fp8 != ref_fp8).float().mean().item()
    return {
        "scale_strides": [list(scale.stride()), list(ref_scale.stride())],
        "scales_bitwise_equal": (scale_ulps == 0).float().mean().item(),
        "max_scale_ulp": scale_ulps.max().item(),
        "fp8_bytes_differ": fp8_bytes_differ,
    }


def check_quantizers(args: argparse.Namespace) -> dict:
    """Layout and closeness of the torch-op quantizers against the Triton ones, eager and compiled with dynamic rows."""
    cases = {
        "balanced": ([1024] * 8, 384),
        "ragged_empty": ([0, 2048, 128, 0, 640, 256, 1152, 0], 256),
        "no_tail": ([512, 128, 1408, 256], 0),
    }
    compiled_1x128 = torch.compile(quant_1x128, fullgraph=True)
    compiled_kmajor = torch.compile(quant_grouped_128x1_kmajor, fullgraph=True)
    report = {}
    for channels in (4096, 2048):
        for name, (counts, tail_rows) in cases.items():
            x, offs = grouped_rows(counts, tail_rows, channels, seed=channels + len(name))
            torch._dynamo.mark_dynamic(x, 0)
            rows = x.shape[0]
            grouped_layout, block_to_group, ks_tensor, starts, actual_ms, block_starts = build_grouped_layout(
                offs, rows
            )
            ref_fp8, ref_scale = grouped_per_token_cast_to_fp8_triton(
                x, rows, block_to_group, starts, actual_ms, block_starts, False, GROUP_ALIGNMENT
            )
            valid = grouped_layout >= 0
            grouped_total = sum(ks_tensor.tolist())
            ref_t_fp8, ref_t_scale = grouped_per_channel_cast_to_fp8_sm90_kmajor_triton(
                x,
                grouped_total,
                block_to_group[: grouped_total // GROUP_ALIGNMENT],
                starts,
                actual_ms,
                ks_tensor,
                block_starts,
                False,
                GROUP_ALIGNMENT,
            )
            for mode, quant, quant_t in (
                ("eager", quant_1x128, quant_grouped_128x1_kmajor),
                ("compiled", compiled_1x128, compiled_kmajor),
            ):
                fp8, scale = quant(x)
                t_fp8, t_scale = quant_t(x, offs)
                blocks = grouped_total // GROUP_ALIGNMENT
                report[f"{mode} C={channels} {name}"] = {
                    "1x128_valid_rows": compare_quant(
                        fp8.view(torch.uint8)[valid], scale[valid], ref_fp8.view(torch.uint8)[valid], ref_scale[valid]
                    ),
                    "1x128_layout": [list(fp8.shape), list(scale.shape), list(scale.stride())]
                    == [list(ref_fp8.shape), list(ref_scale.shape), list(ref_scale.stride())],
                    "kmajor_valid": compare_quant(
                        t_fp8.view(torch.uint8)[: grouped_total * channels],
                        t_scale[:, :blocks],
                        ref_t_fp8.view(torch.uint8),
                        ref_t_scale,
                    ),
                    "kmajor_layout": t_fp8.numel() == rows * channels
                    and t_scale.stride() == ref_t_scale.stride()
                    and t_scale.shape == (channels, rows // GROUP_ALIGNMENT),
                }
    return report


def kernel_launches(code: str) -> list[dict]:
    """List kernel and extern launches in an Inductor wrapper's call() in order, with their origin comment."""
    kernel_sources = dict(re.findall(r"^(triton_\w+) = async_compile\.triton\('\w+', '''(.*?)'''", code, re.S | re.M))
    call_body = code[code.index("def call(") :]
    launches = []
    origin = ""
    for line in call_body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# Topologically Sorted") or stripped.startswith("# Unsorted"):
            origin = stripped.split(":", 1)[-1].strip()
        if match := re.match(r"(triton_\w+)\.run\(", stripped):
            source = kernel_sources.get(match.group(1), "")
            launches.append({"kernel": match.group(1), "fp8_casts": source.count("tl.float8e4nv"), "origin": origin})
            origin = ""
        elif match := re.search(r"torch\.ops\.(\w+\.\w+)\.default\(", stripped):
            launches.append({"kernel": match.group(1), "origin": origin})
            origin = ""
    return launches


def time_ms(fn, *args) -> float:
    return triton.testing.do_bench(lambda: fn(*args), warmup=25, rep=100, return_mode="median")


def bench_quantizers(args: argparse.Namespace, out_dir: Path) -> dict:
    """Kernel count and time of the compiled torch-op quantizers against the Triton ones at the routed shapes."""
    report = {}
    num_experts = args.experts
    for channels in (args.hidden, args.expert_dim):
        rows = args.bench_rows
        group_rows = rows // num_experts // GROUP_ALIGNMENT * GROUP_ALIGNMENT
        counts = [group_rows] * (num_experts - 1)
        counts.append(rows - sum(counts) - 2 * GROUP_ALIGNMENT)
        x, offs = grouped_rows(counts, 2 * GROUP_ALIGNMENT, channels, seed=channels)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.reset()
        compiled_1x128 = torch.compile(quant_1x128, fullgraph=True, options=args.options)
        compiled_kmajor = torch.compile(quant_grouped_128x1_kmajor, fullgraph=True, options=args.options)
        _, code_1x128 = run_and_get_code(compiled_1x128, x)
        _, code_kmajor = run_and_get_code(compiled_kmajor, x, offs)
        (out_dir / f"quant_1x128_C{channels}.py").write_text(code_1x128[0])
        (out_dir / f"quant_kmajor_C{channels}.py").write_text(code_kmajor[0])
        grouped_layout, block_to_group, ks_tensor, starts, actual_ms, block_starts = build_grouped_layout(offs, rows)
        grouped_total = sum(ks_tensor.tolist())

        def triton_1x128():
            return grouped_per_token_cast_to_fp8_triton(
                x, rows, block_to_group, starts, actual_ms, block_starts, False, GROUP_ALIGNMENT
            )

        def triton_kmajor():
            return grouped_per_channel_cast_to_fp8_sm90_kmajor_triton(
                x,
                grouped_total,
                block_to_group[: grouped_total // GROUP_ALIGNMENT],
                starts,
                actual_ms,
                ks_tensor,
                block_starts,
                False,
                GROUP_ALIGNMENT,
            )

        report[f"rows={rows} C={channels}"] = {
            "1x128_kernels": kernel_launches(code_1x128[0]),
            "kmajor_kernels": kernel_launches(code_kmajor[0]),
            "1x128_ms": {"torch": time_ms(compiled_1x128, x), "triton": time_ms(triton_1x128)},
            "kmajor_ms": {"torch": time_ms(compiled_kmajor, x, offs), "triton": time_ms(triton_kmajor)},
        }
    return report


def check_ops(args: argparse.Namespace) -> dict:
    """Pre-quantized ops fed by the Triton quantizers against the existing ops, bitwise, and fed by the torch-op quantizers, loosely."""
    num_experts, channels_in, channels_out = 8, 1024, 512
    counts = [256, 0, 1280, 128, 384, 0, 640, 128]
    tail_rows = 256
    report = {}
    for transposed in (True, False):
        x, offs = grouped_rows(counts, tail_rows, channels_in, seed=1)
        x = x / x.abs().amax(dim=-1, keepdim=True)
        rows = x.shape[0]
        generator = torch.Generator(device="cuda").manual_seed(2)
        weight = (
            torch.randn(num_experts, channels_out, channels_in, device="cuda", generator=generator) * WEIGHT_STD
        ).to(torch.bfloat16)
        weight = weight.transpose(1, 2) if transposed else weight.transpose(1, 2).contiguous()
        grad_output = torch.randn(rows, channels_out, device="cuda", generator=generator).to(torch.bfloat16)
        grouped_layout, block_to_group, ks_tensor, starts, actual_ms, block_starts = build_grouped_layout(offs, rows)
        valid = grouped_layout >= 0
        grouped_total = sum(ks_tensor.tolist())

        def triton_1x128(t):
            return grouped_per_token_cast_to_fp8_triton(
                t, rows, block_to_group, starts, actual_ms, block_starts, False, GROUP_ALIGNMENT
            )

        def triton_kmajor(t):
            return grouped_per_channel_cast_to_fp8_sm90_kmajor_triton(
                t,
                grouped_total,
                block_to_group[: grouped_total // GROUP_ALIGNMENT],
                starts,
                actual_ms,
                ks_tensor,
                block_starts,
                False,
                GROUP_ALIGNMENT,
            )

        ops = torch.ops.prime_rl
        out_ref = ops.grouped_fp8_gemm(x, weight, offs)
        grad_x_ref, grad_weight_ref = ops.grouped_fp8_gemm_backward(
            grad_output, x, weight, offs, True, True, transposed
        )

        out_triton = ops.grouped_fp8_gemm_prequant(*triton_1x128(x), weight, offs)
        tail_poisoned_fp8, tail_poisoned_scale = triton_1x128(x)
        tail_poisoned_fp8.view(torch.uint8)[~valid] = 0x7F
        tail_poisoned_scale[~valid] = float("nan")
        out_poisoned = ops.grouped_fp8_gemm_prequant(tail_poisoned_fp8, tail_poisoned_scale, weight, offs)
        grad_x_triton = ops.grouped_fp8_gemm_dgrad_prequant(*triton_1x128(grad_output), weight, offs)
        grad_weight_triton = ops.grouped_fp8_gemm_wgrad_prequant(
            *triton_kmajor(x), *triton_kmajor(grad_output), weight, offs, transposed
        )

        out_torch = ops.grouped_fp8_gemm_prequant(*quant_1x128(x), weight, offs)
        grad_x_torch = ops.grouped_fp8_gemm_dgrad_prequant(*quant_1x128(grad_output), weight, offs)
        grad_weight_torch = ops.grouped_fp8_gemm_wgrad_prequant(
            *quant_grouped_128x1_kmajor(x, offs),
            *quant_grouped_128x1_kmajor(grad_output, offs),
            weight,
            offs,
            transposed,
        )

        def bitwise(a: torch.Tensor, b: torch.Tensor) -> bool:
            return (
                a.shape == b.shape
                and a.stride() == b.stride()
                and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
            )

        report[f"weight_transposed={transposed}"] = {
            "triton_fed_bitwise": {
                "out_valid_rows": bitwise(out_triton[valid], out_ref[valid]),
                "out_valid_rows_with_nan_tail": bitwise(out_poisoned[valid], out_ref[valid]),
                "grad_x_valid_rows": bitwise(grad_x_triton[valid], grad_x_ref[valid]),
                "grad_weight": bitwise(grad_weight_triton, grad_weight_ref),
                "grad_weight_triton_wgrad_op": bitwise(
                    ops.grouped_fp8_gemm_wgrad(grad_output, x, weight, offs, transposed), grad_weight_ref
                ),
            },
            "torch_fed_rel_err": {
                "out": rel_err(out_torch[valid], out_ref[valid]),
                "grad_x": rel_err(grad_x_torch[valid], grad_x_ref[valid]),
                "grad_weight": rel_err(grad_weight_torch, grad_weight_ref),
            },
        }
    return report


class RoutedExpertBlock(nn.Module):
    """Gather into 128-aligned expert groups, grouped gate_up, ClampedSilu, grouped down, then scatter back to tokens."""

    def __init__(self, num_experts: int, hidden: int, expert_dim: int, gemm) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * expert_dim, hidden))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden, expert_dim))
        self.activation = ClampedSilu(SWIGLU_LIMIT)
        self.gemm = gemm

    def forward(
        self, x: torch.Tensor, permuted_indices: torch.Tensor, num_tokens_per_expert: torch.Tensor
    ) -> torch.Tensor:
        x_padded = torch.vstack((x, x.new_zeros((1, x.shape[-1]))))
        routed = _GatherPermutedRows.apply(x_padded, permuted_indices)
        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        gate_up = self.gemm(routed, self.gate_up_proj.transpose(-2, -1), offsets)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = self.activation.apply(gate, up)
        routed_out = self.gemm(hidden, self.down_proj.transpose(-2, -1), offsets)
        out = routed_out.new_empty(x_padded.shape)
        out[permuted_indices] = routed_out
        return out[:-1]


def bf16_grouped_mm(x: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    return torch._grouped_mm(x, weight, offs=offs)


def build_block(args: argparse.Namespace) -> RoutedExpertBlock:
    block = RoutedExpertBlock(args.experts, args.hidden, args.expert_dim, grouped_fp8_gemm).cuda()
    with torch.no_grad():
        for param in block.parameters():
            nn.init.trunc_normal_(param, std=WEIGHT_STD)
    return block.to(torch.bfloat16)


def make_inputs(args: argparse.Namespace, num_routed: int, dynamic: bool) -> dict[str, torch.Tensor]:
    """Routed rows, a gather order that groups them by expert with 128-row padding, padded group sizes and a probe gradient."""
    from torchtitan.experiments.kernels.moe.indices import generate_permute_indices

    generator = torch.Generator(device="cuda").manual_seed(num_routed)
    x = torch.randn(num_routed, args.hidden, device="cuda", generator=generator).to(torch.bfloat16)
    expert_of_row = torch.randint(0, args.experts, (num_routed,), device="cuda", generator=generator)
    tokens_per_expert = torch.bincount(expert_of_row, minlength=args.experts)
    max_len = (num_routed + args.experts * GROUP_ALIGNMENT + GROUP_ALIGNMENT - 1) // GROUP_ALIGNMENT * GROUP_ALIGNMENT
    permuted_indices, num_tokens_per_expert, _ = generate_permute_indices(
        tokens_per_expert, args.experts, 1, max_len, GROUP_ALIGNMENT
    )
    grad_output = torch.randn(num_routed, args.hidden, device="cuda", generator=generator).to(torch.bfloat16)
    if dynamic:
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(permuted_indices, 0)
    return {
        "x": x.requires_grad_(),
        "permuted_indices": permuted_indices,
        "num_tokens_per_expert": num_tokens_per_expert,
        "grad_output": grad_output,
    }


def run_step(module: nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    for param in module.parameters():
        param.grad = None
    inputs["x"].grad = None
    out = module(inputs["x"], inputs["permuted_indices"], inputs["num_tokens_per_expert"])
    out.backward(inputs["grad_output"])
    return out


def collect(module: nn.Module, inputs: dict[str, torch.Tensor], out: torch.Tensor) -> dict[str, torch.Tensor]:
    grads = {
        f"{name.removeprefix('_checkpoint_wrapped_module.')}.grad": p.grad for name, p in module.named_parameters()
    }
    return {"out": out.detach(), "x.grad": inputs["x"].grad} | grads


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def compile_block(block: RoutedExpertBlock, wgrad_quant: str | None, options: dict) -> nn.Module:
    wrapped = get_activation_checkpoint_wrapper(ActivationCheckpointConfig(mode="full"))(block)
    options = dict(options)
    if wgrad_quant is not None:
        options["post_grad_custom_pre_pass"] = GroupedFp8QuantPass(wgrad_quant)
    wrapped.compile(fullgraph=True, options=options)
    return wrapped


def run_arm(args: argparse.Namespace, base: RoutedExpertBlock, arm: str, out_dir: Path) -> tuple[dict, dict, nn.Module]:
    torch._dynamo.reset()
    PASS_RECORDS.clear()
    unique_graphs_before = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    module = compile_block(copy.deepcopy(base), ARMS[arm], args.options)
    sources: list[str] = []
    GraphLowering.save_output_code = sources.append
    results = {}
    for num_routed in args.routed:
        inputs = make_inputs(args, num_routed, dynamic=True)
        out = run_step(module, inputs)
        torch.cuda.synchronize()
        results[num_routed] = {name: t.clone() for name, t in collect(module, inputs, out).items()}
    GraphLowering.save_output_code = None
    launches = {}
    for index, code in enumerate(sources):
        path = out_dir / f"{arm}_graph{index}.py"
        path.write_text(code)
        launches[path.name] = kernel_launches(code)
    report = {
        "pass_records": list(PASS_RECORDS),
        "compiled_graphs": len(sources),
        "dynamo_unique_graphs": torch._dynamo.utils.counters["stats"]["unique_graphs"] - unique_graphs_before,
        "launches": launches,
    }
    return report, results, module


def profile_steps(module: nn.Module, inputs: dict[str, torch.Tensor], steps: int) -> dict:
    """Per-step kernel launches and GPU time by kernel category, averaged over `steps` forward plus backward steps."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            run_step(module, inputs)
        torch.cuda.synchronize()
    categories: dict[str, dict] = {}
    for event in prof.key_averages():
        if event.device_type != DeviceType.CUDA:
            continue
        name = event.key
        if name.startswith("triton_"):
            category = "inductor_triton"
        elif "deep_gemm" in name:
            category = "deepgemm"
        elif name.startswith("_per_") or name.startswith("_grouped_"):
            category = "triton_quant_eager"
        else:
            category = "other"
        bucket = categories.setdefault(category, {"calls": 0, "ms": 0.0, "kernels": {}})
        bucket["calls"] += event.count / steps
        bucket["ms"] += event.self_device_time_total / 1e3 / steps
        bucket["kernels"][name] = {
            "calls": event.count / steps,
            "ms": round(event.self_device_time_total / 1e3 / steps, 4),
        }
    return categories


def time_steps(module: nn.Module, inputs: dict[str, torch.Tensor], iters: int) -> dict[str, float]:
    for _ in range(3):
        run_step(module, inputs)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        module(inputs["x"], inputs["permuted_indices"], inputs["num_tokens_per_expert"])
    end.record()
    torch.cuda.synchronize()
    forward_ms = start.elapsed_time(end) / iters
    start.record()
    for _ in range(iters):
        run_step(module, inputs)
    end.record()
    torch.cuda.synchronize()
    return {"forward_ms": forward_ms, "forward_backward_ms": start.elapsed_time(end) / iters}


def run_block(args: argparse.Namespace, out_dir: Path) -> dict:
    torch.manual_seed(0)
    base = build_block(args)
    reports, results, modules = {}, {}, {}
    for arm in args.arms:
        reports[arm], results[arm], modules[arm] = run_arm(args, base, arm, out_dir)
    if args.bench_routed:
        inputs = make_inputs(args, args.bench_routed, dynamic=True)
        samples: dict[str, list[dict[str, float]]] = {arm: [] for arm in modules}
        for _ in range(args.reps):
            for arm, module in modules.items():
                samples[arm].append(time_steps(module, inputs, args.iters))
        for arm, module in modules.items():
            reports[arm]["timing"] = {key: statistics.median(s[key] for s in samples[arm]) for key in samples[arm][0]}
            reports[arm]["timing_reps"] = samples[arm]
            reports[arm]["profile"] = profile_steps(module, inputs, args.iters)
            reports[arm]["routed_rows_padded"] = inputs["permuted_indices"].numel()
    reference_block = copy.deepcopy(base)
    reference_block.gemm = bf16_grouped_mm
    closeness = {}
    for num_routed in args.routed:
        inputs = make_inputs(args, num_routed, dynamic=False)
        reference = collect(reference_block, inputs, run_step(reference_block, inputs))
        rows = {}
        for name, ref in reference.items():
            row = {f"{arm}_vs_bf16": rel_err(results[arm][num_routed][name], ref) for arm in results}
            for arm in results:
                if arm != "nopass" and "nopass" in results:
                    row[f"{arm}_vs_nopass"] = rel_err(
                        results[arm][num_routed][name], results["nopass"][num_routed][name]
                    )
            rows[name] = row
        closeness[num_routed] = rows
    return {"arms": reports, "closeness": closeness}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--sections", default="quant,ops,block", help="comma-separated subset of quant,bench,ops,block")
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--expert-dim", type=int, default=2048)
    parser.add_argument("--routed", type=int, nargs="+", default=[20000, 61000, 100000, 186368])
    parser.add_argument("--bench-routed", type=int, default=186368, help="routed rows before padding, 0 to skip timing")
    parser.add_argument("--bench-rows", type=int, default=190464, help="padded rows for the quantizer benchmark")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument(
        "--option", action="append", default=[], help="Inductor option KEY=VALUE, e.g. emulate_precision_casts=True"
    )
    parser.add_argument("--persistent-group-reductions", action="store_true")
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    args = parser.parse_args()

    args.options = {key: ast.literal_eval(value) for key, value in (option.split("=", 1) for option in args.option)}
    if args.persistent_group_reductions:
        args.options["inductor_choices_class"] = PersistentGroupReductionChoices
        V.set_choices_handler(PersistentGroupReductionChoices())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sections = args.sections.split(",")
    summary: dict = {"options": {key: str(value) for key, value in args.options.items()}}
    if "quant" in sections:
        summary["quant"] = check_quantizers(args)
        print(
            "== quant: fraction of scales bitwise equal, max scale ULP, fraction of FP8 bytes differing, layout match"
        )
        for case, row in summary["quant"].items():
            print(
                f"   {case}: 1x128 {row['1x128_valid_rows']['scales_bitwise_equal']:.3f} {row['1x128_valid_rows']['max_scale_ulp']}"
                f" {row['1x128_valid_rows']['fp8_bytes_differ']:.2e} {row['1x128_layout']} |"
                f" kmajor {row['kmajor_valid']['scales_bitwise_equal']:.3f} {row['kmajor_valid']['max_scale_ulp']}"
                f" {row['kmajor_valid']['fp8_bytes_differ']:.2e} {row['kmajor_layout']}",
                flush=True,
            )
    if "bench" in sections:
        summary["bench"] = bench_quantizers(args, args.out_dir)
        print("== bench", json.dumps(summary["bench"], indent=1), flush=True)
    if "ops" in sections:
        summary["ops"] = check_ops(args)
        print("== ops", json.dumps(summary["ops"], indent=1), flush=True)
    if "block" in sections:
        summary["block"] = run_block(args, args.out_dir)
        block = summary["block"]
        for arm, report in block["arms"].items():
            print(
                f"== {arm}: {report['compiled_graphs']} compiled graphs, dynamo unique graphs {report['dynamo_unique_graphs']}"
            )
            for record in report["pass_records"]:
                print(f"   pass record: {json.dumps(record)}")
            if "timing" in report:
                print(f"   timing at {report['routed_rows_padded']} padded rows: {report['timing']}")
                for category, bucket in sorted(report["profile"].items()):
                    print(f"   kernels {category}: {bucket['calls']:.0f} launches, {bucket['ms']:.3f} ms per step")
            for name, launch_list in report["launches"].items():
                print(f"   {name}:")
                for launch in launch_list:
                    extra = f" fp8_casts={launch['fp8_casts']}" if "fp8_casts" in launch else ""
                    print(f"      {launch['kernel']}{extra}  {launch['origin'][:200]}")
        print("== closeness (relative error norm)")
        for num_routed, rows in block["closeness"].items():
            for name, row in rows.items():
                print(
                    f"   routed={num_routed} {name}: " + ", ".join(f"{key}={value:.3e}" for key, value in row.items())
                )
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
