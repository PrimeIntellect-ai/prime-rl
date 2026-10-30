"""Compile a DeepSeek V4 shared-expert toy block with and without the FP8 quant pass, then compare outputs, kernels and times."""

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
import torch.nn.functional as F
from torch import nn
from torch._inductor.graph import GraphLowering
from torch._inductor.virtualized import V
from torch.autograd import DeviceType

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fp8_quant_pass import PASS_RECORDS, Fp8QuantPass, PersistentGroupReductionChoices  # noqa: E402

from prime_rl.configs.trainer import ActivationCheckpointConfig  # noqa: E402
from prime_rl.trainer.activation_checkpointing import get_activation_checkpoint_wrapper  # noqa: E402
from prime_rl.trainer.models.layers.activations import ClampedSilu  # noqa: E402
from prime_rl.trainer.models.layers.fp8_linear import Float8BlockwiseLinear  # noqa: E402
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig  # noqa: E402

HIDDEN = 4096
INTERMEDIATE = 2048
SWIGLU_LIMIT = 10.0
EPS = 1e-6
ARMS = {"nopass": None, "pass-torch": "torch", "pass-triton": "triton"}


class TorchRMSNorm(nn.Module):
    """RMSNorm in plain torch ops, visible to Inductor, with the weight multiply in fp32."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(hidden_states.dtype)


class SharedExpertBlock(nn.Module):
    """Pre-norm gated MLP with a residual, shaped like a DeepSeek V4 shared expert."""

    def __init__(self, norm: str) -> None:
        super().__init__()
        self.norm = RMSNorm(RMSNormConfig(HIDDEN, EPS)) if norm == "quack" else TorchRMSNorm(HIDDEN, EPS)
        self.gate_proj = Float8BlockwiseLinear(HIDDEN, INTERMEDIATE, bias=False)
        self.up_proj = Float8BlockwiseLinear(HIDDEN, INTERMEDIATE, bias=False)
        self.down_proj = Float8BlockwiseLinear(INTERMEDIATE, HIDDEN, bias=False)
        self.activation = ClampedSilu(SWIGLU_LIMIT)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = self.norm(hidden_states)
        gate = self.gate_proj(normed)
        up = self.up_proj(normed)
        return hidden_states + self.down_proj(self.activation.apply(gate, up))


def build_block(norm: str) -> SharedExpertBlock:
    block = SharedExpertBlock(norm).cuda()
    with torch.no_grad():
        for linear in (block.gate_proj, block.up_proj, block.down_proj):
            nn.init.trunc_normal_(linear.weight, std=0.02)
        block.norm.weight.copy_(1.0 + 0.1 * torch.randn_like(block.norm.weight))
    return block.to(torch.bfloat16)


def fp32_reference(block: SharedExpertBlock, x: torch.Tensor, grad_output: torch.Tensor) -> dict[str, torch.Tensor]:
    params = {name: param.detach().float().requires_grad_() for name, param in block.named_parameters()}
    x = x.detach().float().requires_grad_()
    normed = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS) * params["norm.weight"]
    gate = F.linear(normed, params["gate_proj.weight"])
    up = F.linear(normed, params["up_proj.weight"])
    out = x + F.linear(ClampedSilu(SWIGLU_LIMIT).apply(gate, up), params["down_proj.weight"])
    out.backward(grad_output.float())
    return {"out": out.detach(), "x.grad": x.grad} | {f"{name}.grad": p.grad for name, p in params.items()}


def compile_block(block: SharedExpertBlock, pass_mode: str | None, options: dict) -> nn.Module:
    wrapped = get_activation_checkpoint_wrapper(ActivationCheckpointConfig(mode="full"))(block)
    options = dict(options)
    if pass_mode is not None:
        options["post_grad_custom_pre_pass"] = Fp8QuantPass(wgrad_quant=pass_mode)
    wrapped.compile(fullgraph=True, options=options)
    return wrapped


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
    grads = {
        f"{name.removeprefix('_checkpoint_wrapped_module.')}.grad": p.grad for name, p in module.named_parameters()
    }
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


def run_arm(args: argparse.Namespace, base: SharedExpertBlock, arm: str, out_dir: Path) -> dict:
    torch._dynamo.reset()
    PASS_RECORDS.clear()
    unique_graphs_before = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    module = compile_block(copy.deepcopy(base), ARMS[arm], args.options)
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
        "dynamo_unique_graphs": torch._dynamo.utils.counters["stats"]["unique_graphs"] - unique_graphs_before,
        "code_files": code_files,
        "launches": launches,
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
    parser.add_argument("norm", choices=["quack", "torch"])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1000, 4096, 5000, 32768])
    parser.add_argument("--bench-tokens", type=int, default=32768)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument(
        "--option", action="append", default=[], help="Inductor option KEY=VALUE, e.g. triton.nested_reduction=True"
    )
    parser.add_argument("--static", action="store_true", help="do not mark the token dim dynamic")
    parser.add_argument("--persistent-group-reductions", action="store_true")
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    args = parser.parse_args()

    torch._dynamo.config.capture_scalar_outputs = True
    args.options = {key: ast.literal_eval(value) for key, value in (option.split("=", 1) for option in args.option)}
    if args.persistent_group_reductions:
        args.options["inductor_choices_class"] = PersistentGroupReductionChoices
        V.set_choices_handler(PersistentGroupReductionChoices())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    base = build_block(args.norm)

    reports, results, modules = {}, {}, {}
    for arm in args.arms:
        reports[arm], results[arm], modules[arm] = run_arm(args, base, arm, args.out_dir)
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
        "options": args.options,
        "static": args.static,
        "arms": reports,
        "closeness": closeness,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    for arm, report in reports.items():
        print(
            f"== {arm}: {report['compiled_graphs']} compiled graphs, dynamo unique graphs {report['dynamo_unique_graphs']}"
        )
        for record in report["pass_records"]:
            print(f"   pass record: {json.dumps(record)}")
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
