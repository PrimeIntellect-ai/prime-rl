"""Compile DeepSeek V4 shared-expert toy blocks through the trainer's apply_ac and apply_compile, with and without the FP8 quant pass, then compare outputs, kernels and times."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import re
import statistics
import threading
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch._inductor import config as inductor_config
from torch._inductor.graph import GraphLowering
from torch._inductor.virtualized import V
from torch.autograd import DeviceType

import prime_rl.trainer.model as trainer_model
from prime_rl.configs.trainer import ActivationCheckpointConfig, CompileConfig, FP8QuantPassConfig
from prime_rl.trainer.fp8_quant_pass import FP8_MM, FP8_MM_BACKWARD, Fp8QuantPass
from prime_rl.trainer.model import apply_ac, apply_compile, apply_rms_norm_backend
from prime_rl.trainer.models.layers.activations import ClampedSilu
from prime_rl.trainer.models.layers.fp8_linear import Float8BlockwiseLinear
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig

HIDDEN = 4096
INTERMEDIATE = 2048
SWIGLU_LIMIT = 10.0
EPS = 1e-6
ARMS = {"nopass": None, "pass-triton": "triton", "pass-torch": "torch"}
PREQUANT_OPS = (
    torch.ops.prime_rl.fp8_blockwise_mm_prequant.default,
    torch.ops.prime_rl.fp8_blockwise_mm_dgrad_prequant.default,
    torch.ops.prime_rl.fp8_blockwise_mm_wgrad_prequant.default,
    torch.ops.prime_rl.fp8_blockwise_mm_wgrad.default,
)
PASS_RECORDS: list[dict] = []


def count_targets(graph: torch.fx.Graph, targets: tuple) -> Counter:
    return Counter(str(node.target) for node in graph.nodes if node.op == "call_function" and node.target in targets)


class RecordingFp8QuantPass(Fp8QuantPass):
    """The trainer's pass, plus a record of what it rewrote and the compile context it ran in."""

    def __call__(self, graph: torch.fx.Graph) -> None:
        before = count_targets(graph, (FP8_MM, FP8_MM_BACKWARD))
        amax = (torch.ops.aten.amax.default,)
        super().__call__(graph)
        PASS_RECORDS.append(
            {
                "graph": "backward" if before[str(FP8_MM_BACKWARD)] else "forward",
                "thread": threading.current_thread().name,
                "ops_before": dict(before),
                "ops_after": dict(count_targets(graph, (FP8_MM, FP8_MM_BACKWARD) + PREQUANT_OPS)),
                "amax_after_dedupe": count_targets(graph, amax)["aten.amax.default"],
                "inductor_choices": type(V.choices).__name__,
                "mix_order_reduction_non_strict_mode": inductor_config.triton.mix_order_reduction_non_strict_mode,
                "emulate_precision_casts": inductor_config.emulate_precision_casts,
            }
        )


trainer_model.Fp8QuantPass = RecordingFp8QuantPass


class SharedExpertBlock(nn.Module):
    """Pre-norm gated MLP with a residual, shaped like a DeepSeek V4 shared expert."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = RMSNorm(RMSNormConfig(HIDDEN, EPS))
        self.gate_proj = Float8BlockwiseLinear(HIDDEN, INTERMEDIATE, bias=False)
        self.up_proj = Float8BlockwiseLinear(HIDDEN, INTERMEDIATE, bias=False)
        self.down_proj = Float8BlockwiseLinear(INTERMEDIATE, HIDDEN, bias=False)
        self.activation = ClampedSilu(SWIGLU_LIMIT)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = self.norm(hidden_states)
        gate = self.gate_proj(normed)
        up = self.up_proj(normed)
        return hidden_states + self.down_proj(self.activation.apply(gate, up))


class ToyModel(nn.Module):
    """Stack of toy blocks under `model.layers`, where the trainer's `get_language_model` finds decoder layers."""

    def __init__(self, blocks: list[SharedExpertBlock]) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(blocks)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


def build_block() -> SharedExpertBlock:
    block = SharedExpertBlock().cuda()
    with torch.no_grad():
        for linear in (block.gate_proj, block.up_proj, block.down_proj):
            nn.init.trunc_normal_(linear.weight, std=0.02)
        block.norm.weight.copy_(1.0 + 0.1 * torch.randn_like(block.norm.weight))
    return block.to(torch.bfloat16)


def fp32_reference(model: ToyModel, x: torch.Tensor, grad_output: torch.Tensor) -> dict[str, torch.Tensor]:
    params = {name: param.detach().float().requires_grad_() for name, param in model.named_parameters()}
    x = x.detach().float().requires_grad_()
    hidden = x
    for index in range(len(model.model.layers)):
        prefix = f"model.layers.{index}."
        normed = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + EPS) * params[prefix + "norm.weight"]
        gate = F.linear(normed, params[prefix + "gate_proj.weight"])
        up = F.linear(normed, params[prefix + "up_proj.weight"])
        hidden = hidden + F.linear(ClampedSilu(SWIGLU_LIMIT).apply(gate, up), params[prefix + "down_proj.weight"])
    hidden.backward(grad_output.float())
    return {"out": hidden.detach(), "x.grad": x.grad} | {f"{name}.grad": p.grad for name, p in params.items()}


def compile_config(args: argparse.Namespace, wgrad_quant: str | None) -> CompileConfig:
    fp8_quant_pass = None
    if wgrad_quant is not None:
        fp8_quant_pass = FP8QuantPassConfig(
            wgrad_quant=wgrad_quant, persistent_group_reductions=args.persistent_group_reductions
        )
    return CompileConfig(fullgraph=True, fp8_quant_pass=fp8_quant_pass, inductor_options=args.options)


def build_model(args: argparse.Namespace, base: ToyModel, wgrad_quant: str | None) -> ToyModel:
    model = copy.deepcopy(base)
    if args.norm == "torch":
        apply_rms_norm_backend(model, "torch")
    apply_ac(model, ActivationCheckpointConfig(mode="full"))
    apply_compile(model, compile_config(args, wgrad_quant), args.norm)
    return model


def make_inputs(num_tokens: int, dynamic: bool) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(num_tokens)
    x = torch.randn(1, num_tokens, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=generator)
    grad_output = torch.randn(1, num_tokens, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=generator)
    if dynamic:
        torch._dynamo.mark_dynamic(x, 1)
    return x.requires_grad_(), grad_output


def run_step(module: nn.Module, x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    for param in module.parameters():
        param.grad = None
    x.grad = None
    out = module(x)
    out.backward(grad_output)
    return out


def collect(module: nn.Module, x: torch.Tensor, out: torch.Tensor) -> dict[str, torch.Tensor]:
    grads = {f"{name.replace('._checkpoint_wrapped_module', '')}.grad": p.grad for name, p in module.named_parameters()}
    return {"out": out.detach(), "x.grad": x.grad} | grads


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def kernel_launches(code: str) -> list[dict]:
    """List kernel and extern launches in an Inductor wrapper's call() in order, with their origin comment."""
    kernel_sources = dict(re.findall(r"^(triton_\w+) = async_compile\.triton\('\w+', '''(.*?)'''", code, re.S | re.M))
    call_body = code[code.index("def call(") :]
    launches = []
    origin = ""
    for line in call_body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# Topologically Sorted") or stripped.startswith("# Unsorted"):
            origin = stripped
        if match := re.match(r"(triton_\w+)\.run\(", stripped):
            source = kernel_sources.get(match.group(1), "")
            launches.append(
                {
                    "kernel": match.group(1),
                    "fp8_stores": source.count("tl.float8e4nv"),
                    "loads": source.count("tl.load("),
                    "stores": source.count("tl.store("),
                    "origin": origin,
                }
            )
            origin = ""
        elif match := re.search(r"torch\.ops\.(\w+\.\w+)\.default\(", stripped):
            launches.append({"kernel": match.group(1), "origin": origin})
            origin = ""
    return launches


def fp8_kernel_kinds(launches: list[dict]) -> dict[str, int]:
    """Count FP8-storing Inductor kernels by kind: `per` (persistent reduction), `red` (looped reduction), `poi`."""
    return dict(Counter(launch["kernel"].split("_")[1] for launch in launches if launch.get("fp8_stores")))


def profile_steps(module: nn.Module, x: torch.Tensor, grad_output: torch.Tensor, steps: int) -> dict:
    """Per-step kernel launches and GPU time by kernel category, averaged over `steps` forward plus backward steps."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            run_step(module, x, grad_output)
        torch.cuda.synchronize()
    categories: dict[str, dict] = {}
    for event in prof.key_averages():
        if event.device_type != DeviceType.CUDA:
            continue
        name = event.key
        if "fp8_gemm" in name or "sm90" in name.lower() or "deep_gemm" in name:
            category = "deepgemm"
        elif name.startswith("_per_") or name.startswith("_grouped_"):
            category = "triton_quant_eager"
        elif name.startswith("triton_"):
            category = "inductor_triton"
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


def time_steps(module: nn.Module, x: torch.Tensor, grad_output: torch.Tensor, iters: int) -> dict[str, float]:
    for _ in range(3):
        run_step(module, x, grad_output)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        module(x)
    end.record()
    torch.cuda.synchronize()
    forward_ms = start.elapsed_time(end) / iters
    start.record()
    for _ in range(iters):
        run_step(module, x, grad_output)
    end.record()
    torch.cuda.synchronize()
    return {"forward_ms": forward_ms, "forward_backward_ms": start.elapsed_time(end) / iters}


def initialize_default_choices_on_both_threads() -> None:
    """Compile an unrelated function forward and backward, so the main and autograd threads cache Inductor's default choices."""

    def unrelated(t: torch.Tensor) -> torch.Tensor:
        return (t.float().square().sum(-1, keepdim=True).sqrt() * t).to(t.dtype)

    t = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    torch.compile(unrelated, fullgraph=True, dynamic=False)(t).sum().backward()
    torch.cuda.synchronize()


def run_arm(args: argparse.Namespace, base: ToyModel, arm: str, out_dir: Path) -> tuple[dict, dict, nn.Module]:
    torch._dynamo.reset()
    PASS_RECORDS.clear()
    stats = torch._dynamo.utils.counters["stats"]
    unique_graphs_before = stats["unique_graphs"]
    module = build_model(args, base, ARMS[arm])
    sources: list[str] = []
    GraphLowering.save_output_code = sources.append
    results = {}
    for num_tokens in args.tokens:
        x, grad_output = make_inputs(num_tokens, not args.static)
        out = run_step(module, x, grad_output)
        torch.cuda.synchronize()
        results[num_tokens] = {name: t.clone() for name, t in collect(module, x, out).items()}
    GraphLowering.save_output_code = None
    code_files = []
    launches = {}
    for index, code in enumerate(sources):
        path = out_dir / f"{arm}_graph{index}.py"
        path.write_text(code)
        code_files.append(str(path))
        launches[path.name] = kernel_launches(code)
    report = {
        "arm": arm,
        "pass_records": list(PASS_RECORDS),
        "compiled_graphs": len(sources),
        "dynamo_unique_graphs": stats["unique_graphs"] - unique_graphs_before,
        "code_files": code_files,
        "launches": launches,
        "fp8_kernel_kinds": {name: fp8_kernel_kinds(launch_list) for name, launch_list in launches.items()},
    }
    return report, results, module


def benchmark(args: argparse.Namespace, modules: dict[str, nn.Module], reports: dict[str, dict]) -> None:
    """Time the arms in alternating blocks, so drift in GPU clocks over the run hits every arm alike."""
    print("== interleaved timing: arms compiled before the last reset recompile once here", flush=True)
    x, grad_output = make_inputs(args.bench_tokens, not args.static)
    samples: dict[str, list[dict[str, float]]] = {arm: [] for arm in modules}
    for _ in range(args.reps):
        for arm, module in modules.items():
            samples[arm].append(time_steps(module, x, grad_output, args.iters))
    for arm, module in modules.items():
        reports[arm]["timing"] = {
            key: statistics.median(sample[key] for sample in samples[arm]) for key in samples[arm][0]
        } | {"reps": samples[arm]}
        reports[arm]["profile"] = profile_steps(module, x, grad_output, args.iters)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("norm", choices=["quack", "torch"], help="model.rms_norm_backend")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1000, 4096, 5000, 32768])
    parser.add_argument("--bench-tokens", type=int, default=32768, help="0 skips timing")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--option", action="append", default=[], help="model.compile.inductor_options entry KEY=VALUE")
    parser.add_argument("--static", action="store_true", help="do not mark the token dim dynamic")
    parser.add_argument("--persistent-group-reductions", action="store_true")
    parser.add_argument(
        "--preinit-choices",
        action="store_true",
        help="first compile an unrelated function so both threads cache Inductor's default choices",
    )
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    args.options = {key: ast.literal_eval(value) for key, value in (option.split("=", 1) for option in args.option)}
    if args.preinit_choices:
        initialize_default_choices_on_both_threads()
    torch.manual_seed(0)
    base = ToyModel([build_block() for _ in range(args.layers)])

    reports, results, modules = {}, {}, {}
    for arm in args.arms:
        reports[arm], results[arm], modules[arm] = run_arm(args, base, arm, out_dir)
    if args.bench_tokens:
        benchmark(args, modules, reports)

    closeness = {}
    for num_tokens in args.tokens:
        x, grad_output = make_inputs(num_tokens, dynamic=False)
        reference = fp32_reference(base, x, grad_output)
        rows = {}
        for name, ref in reference.items():
            row = {f"{arm}_vs_fp32": rel_err(results[arm][num_tokens][name], ref) for arm in args.arms}
            for arm in args.arms:
                if arm != "nopass" and "nopass" in results:
                    row[f"{arm}_vs_nopass"] = rel_err(
                        results[arm][num_tokens][name], results["nopass"][num_tokens][name]
                    )
            rows[name] = row
        closeness[num_tokens] = rows

    summary = {
        "norm": args.norm,
        "layers": args.layers,
        "options": args.options,
        "static": args.static,
        "persistent_group_reductions": args.persistent_group_reductions,
        "preinit_choices": args.preinit_choices,
        "arms": reports,
        "closeness": closeness,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    for arm, report in reports.items():
        print(
            f"== {arm}: {report['compiled_graphs']} compiled graphs, dynamo unique graphs {report['dynamo_unique_graphs']}"
        )
        for record in report["pass_records"]:
            print(f"   pass record: {json.dumps(record)}")
        print(f"   FP8-storing kernels by kind: {json.dumps(report['fp8_kernel_kinds'])}")
        if "timing" in report:
            print(f"   timing at M={args.bench_tokens}: {report['timing']}")
            for category, bucket in sorted(report["profile"].items()):
                print(f"   kernels {category}: {bucket['calls']:.0f} launches, {bucket['ms']:.3f} ms per step")
        for name, launch_list in report["launches"].items():
            print(f"   {name}:")
            for launch in launch_list:
                extra = f" fp8_stores={launch['fp8_stores']}" if "fp8_stores" in launch else ""
                print(f"      {launch['kernel']}{extra}  {launch['origin'][:220]}")
    print("== closeness (relative error norm)")
    for num_tokens, rows in closeness.items():
        for name, row in rows.items():
            print(f"   M={num_tokens} {name}: " + ", ".join(f"{key}={value:.3e}" for key, value in row.items()))


if __name__ == "__main__":
    main()
