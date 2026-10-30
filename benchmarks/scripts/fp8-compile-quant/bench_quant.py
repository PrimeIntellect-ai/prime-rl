"""Check layout, closeness, recompiles and kernel count of the torch-op FP8 quantizers and time them against the Triton kernels."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch._dynamo.utils import counters
from torch._inductor.choices import InductorChoices
from torch._inductor.utils import run_and_get_code
from torch._inductor.virtualized import V
from triton.testing import do_bench

from prime_rl.trainer.models.kernels.fp8_utils import per_token_cast_to_fp8_tp_triton, per_token_cast_to_fp8_triton

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quant_torch import BLOCK, quant_1x128, quant_128x1_tp  # noqa: E402

SECTIONS = ("layout", "closeness", "recompiles", "kernels", "timing")
H200_PEAK_TB_PER_S = 4.8
FP8_E4M3_MANTISSA_BITS = 3
FP8_E4M3_MIN_NORMAL_EXPONENT = -6
LAYOUT_SHAPES = {
    "1x128": [(31007, 4096), (32768, 1024), (256, 128)],
    "128x1_tp": [(31007, 4096), (32768, 1024), (1000, 130)],
}
MAGNITUDES = ("1e-3", "1", "1e3", "mixed")
RECOMPILE_TOKENS = (31007, 32768, 4097, 1000, 256)
TIMING_SHAPES = {
    4096: (32768, 31007, 190464),
    1024: (32768, 31007),
    8192: (32768, 31007),
    2048: (32768, 31007, 190464),
}


def triton_1x128(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return per_token_cast_to_fp8_triton(x, use_ue8m0=False)


def triton_128x1_tp(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad tokens to a multiple of 128 and cast `x.T`, as `fp8_blockwise_mm_backward` does for wgrad."""
    tokens = x.size(0)
    padded_tokens = (tokens + BLOCK - 1) // BLOCK * BLOCK
    if padded_tokens != tokens:
        x = F.pad(x, (0, 0, 0, padded_tokens - tokens))
    return per_token_cast_to_fp8_tp_triton(x, use_ue8m0=False)


QUANTIZERS = {
    "1x128": (quant_1x128, triton_1x128),
    "128x1_tp": (quant_128x1_tp, triton_128x1_tp),
}


class PersistentGroupReductions(InductorChoices):
    """Inductor choices that make every reduction over at most one quantization group a persistent reduction."""

    @staticmethod
    def should_use_persistent_reduction(features, cooperative_reduction: bool) -> bool:
        return V.graph.sizevars.statically_known_leq(
            features.reduction_numel, BLOCK
        ) or InductorChoices.should_use_persistent_reduction(features, cooperative_reduction)

    def uuid(self) -> str:
        return "persistent-group-reductions"


COMPILE_VARIANTS = {
    "default": ({}, InductorChoices),
    "tuned": (
        {"inductor_choices_class": PersistentGroupReductions, "coordinate_descent_tuning": True},
        PersistentGroupReductions,
    ),
}


class MessageCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def random_bf16(tokens: int, channels: int, magnitude: str = "1") -> torch.Tensor:
    x = torch.randn(tokens, channels, device="cuda")
    if magnitude == "mixed":
        x *= 10 ** torch.empty(tokens, 1, device="cuda").uniform_(-4, 4)
    else:
        x *= float(magnitude)
    return x.bfloat16()


def compile_dynamic_tokens(fn, example: torch.Tensor, variant: str = "default"):
    torch._dynamo.mark_dynamic(example, 0)
    return torch.compile(fn, fullgraph=True, options=COMPILE_VARIANTS[variant][0])


def compiling_call_choices(variant: str):
    """Inductor caches its choices handler per thread on first use, so set it explicitly around the compiling call."""
    return V.set_choices_handler(COMPILE_VARIANTS[variant][1]())


def describe(t: torch.Tensor) -> str:
    return f"shape={tuple(t.shape)} dtype={t.dtype} stride={t.stride()}"


def check_layout() -> None:
    print("\n## Layout: torch-op outputs vs Triton outputs (shape, dtype, stride)\n")
    for name, (torch_fn, triton_fn) in QUANTIZERS.items():
        for tokens, channels in LAYOUT_SHAPES[name]:
            torch._dynamo.reset()
            x = random_bf16(tokens, channels)
            reference = triton_fn(x)
            compiled = compile_dynamic_tokens(torch_fn, x)
            for mode, outputs in (("eager", torch_fn(x)), ("compiled", compiled(x))):
                for label, got, want in zip(("fp8", "scale"), outputs, reference):
                    match = got.shape == want.shape and got.dtype == want.dtype and got.stride() == want.stride()
                    verdict = "MATCH" if match else "MISMATCH"
                    print(f"{name:8} M={tokens:6} K={channels:5} {mode:8} {label:5} {verdict}: {describe(got)}")
                    if not match:
                        print(f"{'':45} triton: {describe(want)}")


def fp8_ulp(fp8_values: torch.Tensor) -> torch.Tensor:
    exponent = torch.frexp(fp8_values.abs())[1] - 1
    return torch.exp2((exponent.clamp(min=FP8_E4M3_MIN_NORMAL_EXPONENT) - FP8_E4M3_MANTISSA_BITS).float())


def closeness(got: tuple[torch.Tensor, torch.Tensor], want: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
    (fp8_got, scale_got), (fp8_want, scale_want) = got, want
    scale_ulps = (scale_got.view(torch.int32).long() - scale_want.view(torch.int32).long()).abs()
    values_got = fp8_got.float()
    values_want = fp8_want.float()
    element_scale_got = scale_got.repeat_interleave(BLOCK, dim=1)
    element_scale_want = scale_want.repeat_interleave(BLOCK, dim=1)
    dequant_diff = (values_got * element_scale_got - values_want * element_scale_want).abs()
    ulp_at_magnitude = fp8_ulp(torch.maximum(values_got.abs(), values_want.abs())) * element_scale_want
    return {
        "scale_equal": (scale_ulps == 0).sum().item() / scale_ulps.numel(),
        "max_scale_ulps": scale_ulps.max().item(),
        "byte_diff": (fp8_got.view(torch.uint8) != fp8_want.view(torch.uint8)).sum().item() / fp8_got.numel(),
        "max_dequant_fp8_ulps": (dequant_diff / ulp_at_magnitude).max().item(),
    }


def check_closeness(tokens: int, channels: int) -> None:
    print(f"\n## Closeness vs Triton, M={tokens} K={channels}, random bf16 (higher scale_equal is better)\n")
    print("scale_equal: fraction of scales bitwise equal. max_scale_ulps: max |scale diff| in fp32 ULPs.")
    print("byte_diff: fraction of FP8 bytes that differ. max_deq_ulps: max |dequant diff| in FP8 ULPs at that value.\n")
    print("| quantizer | mode | magnitude | scale_equal | max_scale_ulps | byte_diff | max_deq_ulps |")
    print("|---|---|---|---:|---:|---:|---:|")
    for name, (torch_fn, triton_fn) in QUANTIZERS.items():
        torch._dynamo.reset()
        compiled = None
        for magnitude in MAGNITUDES:
            x = random_bf16(tokens, channels, magnitude)
            if compiled is None:
                compiled = compile_dynamic_tokens(torch_fn, x)
            reference = triton_fn(x)
            for mode, fn in (("eager", torch_fn), ("compiled", compiled)):
                stats = closeness(fn(x), reference)
                print(
                    f"| {name} | {mode} | {magnitude} | {stats['scale_equal']:.4f} | {stats['max_scale_ulps']} "
                    f"| {stats['byte_diff']:.2e} | {stats['max_dequant_fp8_ulps']:.3f} |"
                )


def check_recompiles(channels: int) -> None:
    print(f"\n## Recompiles under dynamic M, K={channels}, M in {RECOMPILE_TOKENS}\n")
    guard_log = logging.getLogger("torch._dynamo.guards.__guards")
    collector = MessageCollector()
    guard_log.addHandler(collector)
    for name, (torch_fn, _) in QUANTIZERS.items():
        torch._dynamo.reset()
        counters.clear()
        collector.messages.clear()
        first = random_bf16(RECOMPILE_TOKENS[0], channels)
        compiled = compile_dynamic_tokens(torch_fn, first)
        torch._logging.set_logs(guards=True)
        compiled(first)
        torch._logging.set_logs()
        with torch._dynamo.config.patch(error_on_recompile=True):
            for tokens in RECOMPILE_TOKENS[1:]:
                x = random_bf16(tokens, channels)
                for got, want in zip(compiled(x), QUANTIZERS[name][1](x)):
                    assert got.shape == want.shape and got.stride() == want.stride(), (name, tokens)
        print(f"{name}: unique_graphs={counters['stats']['unique_graphs']} after {len(RECOMPILE_TOKENS)} M values")
        token_guards = {
            line.strip()
            for message in collector.messages
            for line in message.splitlines()
            if "size()[0]" in line or "TENSOR_MATCH" in line
        }
        for line in sorted(token_guards):
            print(f"  guard: {line}")
    guard_log.removeHandler(collector)


def check_kernels(tokens: int, channels: int, out_dir: Path) -> None:
    print(f"\n## Inductor kernels, dynamic M, example M={tokens} K={channels}\n")
    for name, (torch_fn, _) in QUANTIZERS.items():
        for variant in COMPILE_VARIANTS:
            torch._dynamo.reset()
            x = random_bf16(tokens, channels)
            compiled = compile_dynamic_tokens(torch_fn, x, variant)
            with compiling_call_choices(variant), torch._inductor.config.patch(force_disable_caches=True):
                _, source_codes = run_and_get_code(compiled, x)
            code = "\n".join(source_codes)
            code_path = out_dir / f"output_code_{name}_{variant}.py"
            code_path.write_text(code)
            kernels = re.findall(r"^(triton_\w+) = async_compile\.triton\('\w+', '''(.*?)'''", code, re.S | re.M)
            launches = re.findall(r"\b(triton_\w+)\.run\(", code)
            print(f"{name} {variant}: {len(kernels)} kernel definitions, {len(launches)} launches, code in {code_path}")
            for kernel_name, source in kernels:
                heuristic = re.search(r"@triton_heuristics\.(\w+)", source).group(1)
                signature = re.search(r"def \w+\(([^)]*)\)", source).group(1)
                print(f"  {kernel_name}: {heuristic}({signature})")
                for line in source.splitlines():
                    if "tl.load(" in line or "tl.store(" in line or "numel =" in line:
                        print(f"    {line.strip()}")


def ideal_bytes(name: str, tokens: int, channels: int) -> int:
    bf16_read = tokens * channels * 2
    if name == "1x128":
        return bf16_read + tokens * channels + tokens * (channels // BLOCK) * 4
    padded_tokens = (tokens + BLOCK - 1) // BLOCK * BLOCK
    return bf16_read + padded_tokens * channels + padded_tokens // BLOCK * channels * 4


def check_timing() -> None:
    print("\n## Timing, do_bench median (lower ms is better, higher TB/s is better)\n")
    print("bytes = bf16 read + FP8 write + fp32 scale write (no padding copies). TB/s = bytes / time.")
    print(f"H200 peak ~{H200_PEAK_TB_PER_S} TB/s. default: torch.compile without options.")
    print("tuned: persistent reductions for <=128-wide reductions plus coordinate descent tuning.")
    print("Triton 128x1_tp includes the F.pad copy when M is not a multiple of 128, as in fp8_blockwise_mm_backward.\n")
    print("| quantizer | M | K | default ms | tuned ms | Triton ms | default TB/s | tuned TB/s | Triton TB/s |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for channels, token_counts in TIMING_SHAPES.items():
        for name, (torch_fn, triton_fn) in QUANTIZERS.items():
            torch._dynamo.reset()
            compiled = {}
            for tokens in token_counts:
                x = random_bf16(tokens, channels)
                if not compiled:
                    for variant in COMPILE_VARIANTS:
                        compiled[variant] = compile_dynamic_tokens(torch_fn, x, variant)
                        with compiling_call_choices(variant):
                            compiled[variant](x)
                default_ms = do_bench(lambda: compiled["default"](x), return_mode="median")
                tuned_ms = do_bench(lambda: compiled["tuned"](x), return_mode="median")
                triton_ms = do_bench(lambda: triton_fn(x), return_mode="median")
                total_bytes = ideal_bytes(name, tokens, channels)
                print(
                    f"| {name} | {tokens} | {channels} | {default_ms:.4f} | {tuned_ms:.4f} | {triton_ms:.4f} "
                    f"| {total_bytes / default_ms / 1e9:.2f} | {total_bytes / tuned_ms / 1e9:.2f} "
                    f"| {total_bytes / triton_ms / 1e9:.2f} |"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="directory for generated Inductor code")
    parser.add_argument("--sections", default=",".join(SECTIONS), help=f"comma-separated subset of {SECTIONS}")
    parser.add_argument("--tokens", type=int, default=31007, help="M for the closeness and kernel sections")
    parser.add_argument("--channels", type=int, default=4096, help="K for the closeness, recompile and kernel sections")
    args = parser.parse_args()
    sections = args.sections.split(",")
    unknown = set(sections) - set(SECTIONS)
    if unknown:
        parser.error(f"unknown sections {sorted(unknown)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"torch {torch.__version__}, device {torch.cuda.get_device_name()}")
    if "layout" in sections:
        check_layout()
    if "closeness" in sections:
        check_closeness(args.tokens, args.channels)
    if "recompiles" in sections:
        check_recompiles(args.channels)
    if "kernels" in sections:
        check_kernels(args.tokens, args.channels, args.out_dir)
    if "timing" in sections:
        check_timing()


if __name__ == "__main__":
    main()
