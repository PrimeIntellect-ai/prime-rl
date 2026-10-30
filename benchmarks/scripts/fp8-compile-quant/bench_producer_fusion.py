"""Compile each FP8 GEMM input producer together with its torch-op quantizer and compare it against the compiled producer followed by today's Triton quant kernel."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch._dynamo.utils import counters
from torch._inductor.choices import InductorChoices
from torch._inductor.utils import run_and_get_code
from torch._inductor.virtualized import V
from torch.profiler import ProfilerActivity, profile
from triton.testing import do_bench

from prime_rl.trainer.models.kernels.fp8_utils import per_token_cast_to_fp8_tp_triton, per_token_cast_to_fp8_triton
from prime_rl.trainer.models.layers.activations import ClampedSilu
from prime_rl.trainer.models.layers.norms import get_quack_rmsnorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_quant import PersistentGroupReductions  # noqa: E402
from quant_torch import quant_1x128, quant_128x1_tp  # noqa: E402

GROUP = 128
SWIGLU_LIMIT = 10.0
RMS_NORM_EPS = 1e-6
DENSE_TOKENS = (32768, 31007)
ROUTED_TOKENS = 190464
LOCAL_EXPERTS = 32
EP_RANKS = 8
GATHER_REAL_TOKENS = (ROUTED_TOKENS - LOCAL_EXPERTS * GROUP, 185001)
FP8_E4M3_RELATIVE_SPACING = 2.0**-3
FP8_E4M3_SUBNORMAL_SPACING = 2.0**-9
PROFILED_CALLS = 3
DEFAULT_OUT_DIR = Path.home() / "tmp/fp8-compile-quant/fusion"

clamped_silu = ClampedSilu(SWIGLU_LIMIT)


def swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    return clamped_silu.apply(gate, up)


def torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """RMSNorm in quack's fp32 order: sum of squares, rsqrt, then `x * rstd * weight` rounded to bf16 once."""
    x_fp32 = x.float()
    rstd = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + RMS_NORM_EPS)
    return (x_fp32 * rstd * weight.float()).to(x.dtype)


def quack_rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return get_quack_rmsnorm()(x, weight, eps=RMS_NORM_EPS)


def gather_permuted(x: torch.Tensor, permuted_indices: torch.Tensor) -> torch.Tensor:
    """Forward of `permute_for_grouped_gemm`: append a zero row that the -1 padding slots of the gather read."""
    x = torch.vstack((x, x.new_zeros((1, x.shape[-1]))))
    return x[permuted_indices]


def triton_1x128(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return per_token_cast_to_fp8_triton(y, use_ue8m0=False)


def triton_128x1_tp(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad tokens to a multiple of 128 and cast `y.T`, as `fp8_blockwise_mm_backward` does for wgrad."""
    tokens = y.size(0)
    padded_tokens = (tokens + GROUP - 1) // GROUP * GROUP
    if padded_tokens != tokens:
        y = F.pad(y, (0, 0, 0, padded_tokens - tokens))
    return per_token_cast_to_fp8_tp_triton(y, use_ue8m0=False)


TORCH_QUANTS = {"1x128": quant_1x128, "128x1_tp": quant_128x1_tp}
TRITON_QUANTS = {"1x128": triton_1x128, "128x1_tp": triton_128x1_tp}


def random_bf16(tokens: int, hidden: int, std: float = 1.0) -> torch.Tensor:
    return (std * torch.randn(tokens, hidden, device="cuda")).to(torch.bfloat16)


def plain_inputs(hidden: int, std: float = 1.0) -> Callable[[int], tuple[torch.Tensor, ...]]:
    return lambda tokens: (random_bf16(tokens, hidden, std),)


def norm_inputs(hidden: int) -> Callable[[int], tuple[torch.Tensor, ...]]:
    def make(tokens: int) -> tuple[torch.Tensor, ...]:
        weight = (1.0 + 0.1 * torch.randn(hidden, device="cuda")).to(torch.bfloat16)
        return random_bf16(tokens, hidden), weight

    return make


def gather_inputs(real_tokens: int) -> tuple[torch.Tensor, ...]:
    """Routed rows plus `generate_permute_indices` output for 32 local experts receiving from 8 ranks, as in `permute_for_grouped_gemm`."""
    from torchtitan.experiments.kernels.moe.indices import generate_permute_indices

    splits = torch.randint(0, real_tokens + 1, (LOCAL_EXPERTS * EP_RANKS - 1,)).sort().values
    tokens_per_expert_group = torch.diff(splits, prepend=torch.tensor([0]), append=torch.tensor([real_tokens]))
    max_len = (real_tokens + LOCAL_EXPERTS * GROUP + GROUP - 1) // GROUP * GROUP
    permuted_indices, _, _ = generate_permute_indices(
        tokens_per_expert_group.cuda(), LOCAL_EXPERTS, EP_RANKS, max_len, GROUP
    )
    return random_bf16(real_tokens, 4096), permuted_indices


@dataclass(frozen=True)
class ShapeGroup:
    label: str
    tokens: tuple[int, ...]
    make_inputs: Callable[[int], tuple[torch.Tensor, ...]]
    dynamic_dims: tuple[tuple[int, int], ...] = ((0, 0),)


@dataclass(frozen=True)
class Baseline:
    name: str
    producer: Callable | None
    compile_producer: bool


@dataclass(frozen=True)
class Case:
    title: str
    producer: Callable | None
    quants: tuple[str, ...]
    baselines: tuple[Baseline, ...]
    groups: tuple[ShapeGroup, ...]
    variants: dict[str, dict]


PRECISION_VARIANTS = {"default": {}, "emulate_casts": {"emulate_precision_casts": True}}
NESTED_VARIANTS = {
    "nested_off": {"triton.nested_reduction": False},
    "nested_on": {"triton.nested_reduction": True},
    "nested_off+emulate": {"triton.nested_reduction": False, "emulate_precision_casts": True},
    "nested_on+emulate": {"triton.nested_reduction": True, "emulate_precision_casts": True},
}
MIX_ORDER_VARIANTS = {
    "mix_order_on": {"triton.mix_order_reduction": True},
    "mix_order_off": {"triton.mix_order_reduction": False},
}
TUNED = {"inductor_choices_class": PersistentGroupReductions, "coordinate_descent_tuning": True}
TUNING_VARIANTS = {
    "multi_kernel": {"triton.multi_kernel": 1},
    "coord_descent": {"coordinate_descent_tuning": True},
    "tuned": TUNED,
}
SWIGLU_BASELINE = Baseline("inductor+triton", swiglu, True)
GATE_UP_DENSE = ShapeGroup("gate_up 4096->2048", DENSE_TOKENS, plain_inputs(4096, std=4.0))

CASES = {
    "a": Case(
        "ClampedSilu then quant_1x128",
        swiglu,
        ("1x128",),
        (SWIGLU_BASELINE,),
        (ShapeGroup("gate_up 4096->2048", DENSE_TOKENS + (ROUTED_TOKENS,), plain_inputs(4096, std=4.0)),),
        {**PRECISION_VARIANTS, **TUNING_VARIANTS},
    ),
    "b": Case(
        "RMSNorm then quant_1x128",
        torch_rmsnorm,
        ("1x128",),
        (Baseline("quack+triton", quack_rmsnorm, False), Baseline("inductor+triton", torch_rmsnorm, True)),
        (ShapeGroup("x 4096", DENSE_TOKENS, norm_inputs(4096)), ShapeGroup("x 1024", DENSE_TOKENS, norm_inputs(1024))),
        NESTED_VARIANTS,
    ),
    "c": Case(
        "permutation gather then quant_1x128",
        gather_permuted,
        ("1x128",),
        (Baseline("inductor+triton", gather_permuted, True),),
        (ShapeGroup("gather 4096", GATHER_REAL_TOKENS, gather_inputs, ((0, 0), (1, 0))),),
        {"default": {}, **TUNING_VARIANTS},
    ),
    "d": Case(
        "dY then quant_1x128 and quant_128x1_tp",
        None,
        ("1x128", "128x1_tp"),
        (Baseline("triton", None, False),),
        (
            ShapeGroup("dY 4096", DENSE_TOKENS, plain_inputs(4096)),
            ShapeGroup("dY 2048", DENSE_TOKENS, plain_inputs(2048)),
        ),
        {**MIX_ORDER_VARIANTS, "tuned": TUNED},
    ),
    "e": Case(
        "recomputed ClampedSilu then quant_1x128 and quant_128x1_tp",
        swiglu,
        ("1x128", "128x1_tp"),
        (SWIGLU_BASELINE,),
        (GATE_UP_DENSE,),
        {
            **PRECISION_VARIANTS,
            "mix_order_off": {"triton.mix_order_reduction": False},
            **TUNING_VARIANTS,
            "emulate_casts+tuned": {"emulate_precision_casts": True, **TUNED},
        },
    ),
}


def fused_fn(producer: Callable | None, quants: tuple[str, ...]) -> Callable:
    def fused(*inputs: torch.Tensor):
        y = inputs[0] if producer is None else producer(*inputs)
        return tuple(TORCH_QUANTS[quant](y) for quant in quants)

    return fused


def unfused_fn(producer: Callable | None, quants: tuple[str, ...]) -> Callable:
    def unfused(*inputs: torch.Tensor):
        y = inputs[0] if producer is None else producer(*inputs)
        return tuple(TRITON_QUANTS[quant](y) for quant in quants)

    return unfused


KERNEL_SOURCE = re.compile(r"(\w+) = async_compile\.triton\('\w+', '''(.*?)'''", re.S)
MULTI_KERNEL = re.compile(r"(\w+) = async_compile\.multi_kernel\('\w+', \[(.*?)\]", re.S)
LAUNCH = re.compile(r"\b(\w+)\.run\(")
EXTERN_CALL = re.compile(r"\b(extern_kernels\.\w+|torch\.ops\.\w+\.\w+)")
INDIRECT_LOAD = re.compile(r"tl\.load\(in_ptr\d+ \+ \([^)]*tmp\d+[^)]*\)")
REDUCTION = re.compile(r"tl\.sum\(|max2\(|tl\.max\(")


def kernel_traits(source: str) -> list[str]:
    body = source[source.index("def triton") :]
    if body.count("def triton") > 1:
        return [f"multi_kernel of {body.count('def triton')}"]
    traits = []
    if "tl.float8e4nv" in body:
        traits.append("fp8 cast")
    if "sigmoid" in body or "libdevice.exp(" in body:
        traits.append("silu exp")
    if "rsqrt" in body:
        traits.append("rsqrt")
    if INDIRECT_LOAD.search(body):
        traits.append("indirect load")
    traits.append(f"{len(REDUCTION.findall(body))} reductions")
    traits.append(f"{body.count('tl.load(')} loads")
    traits.append(f"{body.count('tl.store(')} stores")
    return traits


def summarize_code(code: str) -> dict:
    sources = {name: source for name, source in KERNEL_SOURCE.findall(code)}
    for name, members in MULTI_KERNEL.findall(code):
        sources[name] = "\n".join(sources[member] for member in re.findall(r"\w+", members))
    wrapper = "\n".join(line for line in KERNEL_SOURCE.sub("", code).splitlines() if not line.lstrip().startswith("#"))
    launches = Counter(name for name in LAUNCH.findall(wrapper) if name in sources)
    return {
        "launches": sum(launches.values()),
        "kernels": [
            {"name": name, "launches": count, "traits": kernel_traits(sources[name])}
            for name, count in launches.items()
        ],
        "extern_calls": sorted({call for call in EXTERN_CALL.findall(wrapper) if "_alloc_from_pool" not in call}),
    }


def profiled_kernels(fn: Callable) -> dict[str, dict[str, float]]:
    """Launches and GPU microseconds per call of each CUDA kernel `fn` runs."""
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(PROFILED_CALLS):
            fn()
        torch.cuda.synchronize()
    kernels: dict[str, dict[str, float]] = {}
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            stats = kernels.setdefault(event.name, {"calls": 0.0, "us": 0.0})
            stats["calls"] += 1 / PROFILED_CALLS
            stats["us"] += event.time_range.elapsed_us() / PROFILED_CALLS
    return kernels


def kernels_per_call(kernels: dict[str, dict[str, float]]) -> str:
    return f"{sum(stats['calls'] for stats in kernels.values()):g}"


def kernel_ms(kernels: dict[str, dict[str, float]]) -> str:
    return f"{sum(stats['us'] for stats in kernels.values()) / 1e3:.3f}" if kernels else "n/a"


def bench_ms(fn: Callable) -> float:
    return do_bench(fn, warmup=25, rep=200, return_mode="median")


def ordered_e4m3(q: torch.Tensor) -> torch.Tensor:
    bits = q.contiguous().view(torch.uint8).int()
    magnitude = bits & 0x7F
    return torch.where(bits >= 0x80, -magnitude, magnitude)


def ordered_bf16(t: torch.Tensor) -> torch.Tensor:
    bits = t.contiguous().view(torch.int16).int()
    magnitude = bits & 0x7FFF
    return torch.where(bits < 0, -magnitude, magnitude)


def bf16_ulp_histogram(test: torch.Tensor, ref: torch.Tensor) -> str:
    ulps = (ordered_bf16(test) - ordered_bf16(ref)).abs()
    fractions = [(ulps == n).float().mean().item() for n in (0, 1, 2)]
    beyond = (ulps > 2).float().mean().item()
    return (
        f"0 ulp {100 * fractions[0]:.3f}%, 1 ulp {100 * fractions[1]:.3f}%, 2 ulp {100 * fractions[2]:.4f}%, "
        f">2 ulp {100 * beyond:.4f}%, max {int(ulps.max())}"
    )


def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    rows, cols = q.shape
    return q.float().view(rows, cols // GROUP, GROUP) * scale.float().contiguous().unsqueeze(-1)


def closeness(test: tuple[torch.Tensor, torch.Tensor], ref: tuple[torch.Tensor, torch.Tensor]) -> dict:
    """Compare (fp8, scale) pairs: scale fp32 ULPs, fp8 byte flips, and dequantized error in units of one fp8 step."""
    (q, scale), (q_ref, scale_ref) = test, ref
    assert q.shape == q_ref.shape and scale.shape == scale_ref.shape, (q.shape, q_ref.shape, scale.shape)
    scale, scale_ref = scale.float().contiguous(), scale_ref.float().contiguous()
    scale_ulps = (scale.view(torch.int32).long() - scale_ref.view(torch.int32).long()).abs()
    deq_error = (dequantize(q, scale) - dequantize(q_ref, scale_ref)).abs()
    fp8_step = FP8_E4M3_RELATIVE_SPACING * dequantize(
        q_ref, scale_ref
    ).abs() + FP8_E4M3_SUBNORMAL_SPACING * scale_ref.unsqueeze(-1)
    return {
        "scale_max_ulp": int(scale_ulps.max()),
        "scale_frac_over_4_ulp": (scale_ulps > 4).float().mean().item(),
        "scale_max_rel": ((scale - scale_ref).abs() / scale_ref).max().item(),
        "fp8_diff_frac": (q.contiguous().view(torch.uint8) != q_ref.contiguous().view(torch.uint8))
        .float()
        .mean()
        .item(),
        "fp8_max_ulp": int((ordered_e4m3(q) - ordered_e4m3(q_ref)).abs().max()),
        "deq_error_over_fp8_step": (deq_error / fp8_step).max().item(),
    }


def mark_dynamic(inputs: tuple[torch.Tensor, ...], dynamic_dims: tuple[tuple[int, int], ...]) -> None:
    for index, dim in dynamic_dims:
        torch._dynamo.mark_dynamic(inputs[index], dim)


def ncu_probe(label: str, fn: Callable) -> None:
    fn()
    torch.cuda.synchronize()
    print(f"ncu probe: {label}", flush=True)
    torch.cuda.profiler.start()
    fn()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


def run_group(
    key: str, case: Case, variant: str, options: dict, group: ShapeGroup, out_dir: Path, ncu: bool
) -> list[dict]:
    rows = []
    torch.manual_seed(0)
    fused = torch.compile(fused_fn(case.producer, case.quants), fullgraph=True, options=options)
    first_inputs = group.make_inputs(group.tokens[0])
    mark_dynamic(first_inputs, group.dynamic_dims)
    _, codes = run_and_get_code(fused, *first_inputs)
    slug = re.sub(r"[^\w]+", "_", f"{key}_{variant}_{group.label}")
    (out_dir / f"{slug}.py").write_text("\n\n".join(codes))
    summary = summarize_code("\n".join(codes))
    print(
        f"\nkernels ({key}, {variant}, {group.label}): {summary['launches']} launches, externs {summary['extern_calls']}"
    )
    for kernel in summary["kernels"]:
        print(f"  {kernel['name']} x{kernel['launches']}: {', '.join(kernel['traits'])}")
    baselines = []
    for baseline in case.baselines:
        producer = baseline.producer
        if baseline.compile_producer:
            producer = torch.compile(producer, fullgraph=True, options=options)
        baselines.append((baseline.name, producer, unfused_fn(producer, case.quants)))
    for tokens in group.tokens:
        inputs = first_inputs if tokens == group.tokens[0] else group.make_inputs(tokens)
        mark_dynamic(inputs, group.dynamic_dims)
        graphs_before = counters["stats"]["unique_graphs"]
        fused_out = fused(*inputs)
        row = {
            "case": key,
            "variant": variant,
            "shape": group.label,
            "tokens": fused_out[0][0].shape[0],
            "fused_recompiles": counters["stats"]["unique_graphs"] - graphs_before,
            "code_launches": summary["launches"],
            "code_kernels": summary["kernels"],
            "baselines": [],
        }
        if ncu:
            ncu_probe(f"{key} {variant} {group.label} M={row['tokens']} fused", lambda: fused(*inputs))
        else:
            row["fused_profiler"] = profiled_kernels(lambda: fused(*inputs))
            row["fused_ms"] = bench_ms(lambda: fused(*inputs))
        for name, producer, unfused in baselines:
            ref_out = unfused(*inputs)
            result = {
                "name": name,
                "closeness": {quant: closeness(test, ref) for quant, test, ref in zip(case.quants, fused_out, ref_out)},
            }
            del ref_out
            if ncu:
                if name == case.baselines[0].name:
                    ncu_probe(f"{key} {variant} {group.label} M={row['tokens']} {name}", lambda: unfused(*inputs))
            else:
                result["profiler"] = profiled_kernels(lambda: unfused(*inputs))
                result["ms"] = bench_ms(lambda: unfused(*inputs))
                result["producer_ms"] = None if producer is None else bench_ms(lambda: producer(*inputs))
            row["baselines"].append(result)
        print(json.dumps(row), flush=True)
        rows.append(row)
    return rows


def run_case(key: str, case: Case, out_dir: Path, variants: set[str] | None, ncu: bool) -> list[dict]:
    rows = []
    for variant, options in case.variants.items():
        if variants is not None and variant not in variants:
            continue
        for group in case.groups:
            with V.set_choices_handler(options.get("inductor_choices_class", InductorChoices)()):
                rows.extend(run_group(key, case, variant, options, group, out_dir, ncu))
    return rows


def norm_extras() -> list[dict]:
    """bf16 ULP histogram of the compiled torch RMSNorm against quack, and backward time of both."""
    torch._functorch.config.donated_buffer = False
    results = []
    for hidden in (4096, 1024):
        for tokens in DENSE_TOKENS:
            # Without a reset, automatic dynamic shapes make the hidden size symbolic once a second size is seen.
            torch._dynamo.reset()
            torch.manual_seed(0)
            x, weight = norm_inputs(hidden)(tokens)
            compiled = torch.compile(torch_rmsnorm, fullgraph=True)
            torch._dynamo.mark_dynamic(x, 0)
            y_quack = quack_rmsnorm(x, weight)
            result = {
                "hidden": hidden,
                "tokens": tokens,
                "fwd_ulp_compiled_vs_quack": bf16_ulp_histogram(compiled(x, weight), y_quack),
                "fwd_ulp_eager_vs_quack": bf16_ulp_histogram(torch_rmsnorm(x, weight), y_quack),
            }
            x.requires_grad_()
            weight.requires_grad_()
            grad_out = torch.randn_like(x)
            y_torch = compiled(x, weight)
            y_quack = quack_rmsnorm(x, weight)

            def torch_backward():
                return torch.autograd.grad(y_torch, (x, weight), grad_out, retain_graph=True)

            def quack_backward():
                return torch.autograd.grad(y_quack, (x, weight), grad_out, retain_graph=True)

            (dx_torch, dw_torch), (dx_quack, dw_quack) = torch_backward(), quack_backward()
            result["dx_ulp_compiled_vs_quack"] = bf16_ulp_histogram(dx_torch, dx_quack)
            result["dw_max_rel_compiled_vs_quack"] = (
                ((dw_torch.float() - dw_quack.float()).abs() / dw_quack.float().abs().clamp(min=1e-6)).max().item()
            )
            result["bwd_torch_profiler"] = profiled_kernels(torch_backward)
            result["bwd_quack_profiler"] = profiled_kernels(quack_backward)
            result["bwd_torch_ms"] = bench_ms(torch_backward)
            result["bwd_quack_ms"] = bench_ms(quack_backward)
            print(json.dumps(result), flush=True)
            results.append(result)
    return results


def fmt_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def print_tables(key: str, case: Case, rows: list[dict]) -> None:
    print(f"\n## ({key}) {case.title}\n")
    print("Times in ms, lower is better. bench: triton.testing.do_bench median. kern: profiler sum of kernel times.")
    print("k: CUDA kernels per call, code launches/profiler.")
    print(
        "| variant | shape | M | fused k | fused bench | fused kern | baseline | base k | base bench | base kern | prod |"
    )
    print("|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|")
    for row in rows:
        fused_k = f"{row['code_launches']}/{kernels_per_call(row.get('fused_profiler', {}))}"
        for result in row["baselines"]:
            print(
                f"| {row['variant']} | {row['shape']} | {row['tokens']} | {fused_k} | {fmt_ms(row.get('fused_ms'))} "
                f"| {kernel_ms(row.get('fused_profiler', {}))} | {result['name']} "
                f"| {kernels_per_call(result.get('profiler', {}))} | {fmt_ms(result.get('ms'))} "
                f"| {kernel_ms(result.get('profiler', {}))} | {fmt_ms(result.get('producer_ms'))} |"
            )
    print(
        "\nCloseness, fused vs baseline. deq err: max |dequant diff| in units of one fp8 step at the reference value."
    )
    print("| variant | shape | M | out | vs | scale max ulp | scale >4 ulp % | fp8 diff % | fp8 max ulp | deq err |")
    print("|---|---|---:|---|---|---:|---:|---:|---:|---:|")
    for row in rows:
        for result in row["baselines"]:
            for quant, stats in result["closeness"].items():
                print(
                    f"| {row['variant']} | {row['shape']} | {row['tokens']} | {quant} | {result['name']} "
                    f"| {stats['scale_max_ulp']} | {100 * stats['scale_frac_over_4_ulp']:.3f} "
                    f"| {100 * stats['fp8_diff_frac']:.4f} | {stats['fp8_max_ulp']} "
                    f"| {stats['deq_error_over_fp8_step']:.3f} |"
                )
    recompiles = sum(row["fused_recompiles"] for row in rows)
    print(f"\nfused recompiles across token counts: {recompiles}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[*CASES, "all"], default="all")
    parser.add_argument("--variants", default=None, help="Comma-separated variant names to run, default all.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--ncu", action="store_true", help="Run one profiler-range call per variant for ncu, no timing."
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    variants = None if args.variants is None else set(args.variants.split(","))
    keys = list(CASES) if args.case == "all" else [args.case]
    for key in keys:
        case = CASES[key]
        rows = run_case(key, case, args.out_dir, variants, args.ncu)
        extras = norm_extras() if key == "b" and not args.ncu else []
        suffix = "_ncu" if args.ncu else ""
        (args.out_dir / f"results_{key}{suffix}.json").write_text(
            json.dumps({"rows": rows, "extras": extras}, indent=1)
        )
        print_tables(key, case, rows)
        for extra in extras:
            print(
                f"norm H={extra['hidden']} M={extra['tokens']}: fwd compiled vs quack {extra['fwd_ulp_compiled_vs_quack']}"
                f" | eager vs quack {extra['fwd_ulp_eager_vs_quack']}"
            )
            print(
                f"  bwd bench/kern ms (lower is better): compiled torch {extra['bwd_torch_ms']:.3f}"
                f"/{kernel_ms(extra['bwd_torch_profiler'])} ({kernels_per_call(extra['bwd_torch_profiler'])} kernels),"
                f" quack {extra['bwd_quack_ms']:.3f}"
                f"/{kernel_ms(extra['bwd_quack_profiler'])} ({kernels_per_call(extra['bwd_quack_profiler'])} kernels)"
                f" | dx {extra['dx_ulp_compiled_vs_quack']}"
                f" | dw max rel {extra['dw_max_rel_compiled_vs_quack']:.2e}"
            )


if __name__ == "__main__":
    main()
