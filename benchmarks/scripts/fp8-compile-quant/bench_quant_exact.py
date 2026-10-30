"""Check the div_rn torch-op FP8 quantizers bit for bit against the Triton kernels, and time them."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch._inductor.choices import InductorChoices
from torch._inductor.virtualized import V
from triton.testing import do_bench

from prime_rl.trainer.models.layers.activations import ClampedSilu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_quant import random_bf16, triton_1x128, triton_128x1_tp  # noqa: E402
from bench_quant_tuning import captured_graphs, kernel_summaries  # noqa: E402
from quant_choices import fp8_group_quant_backend  # noqa: E402
from quant_exact import quant_1x128_exact, quant_128x1_tp_exact  # noqa: E402
from quant_torch import BLOCK, quant_1x128, quant_128x1_tp  # noqa: E402

SECTIONS = ("bitwise", "kernels", "timing", "producer")
EXACT = {
    "1x128": (quant_1x128_exact, triton_1x128, (4096, 1024)),
    "128x1_tp": (quant_128x1_tp_exact, triton_128x1_tp, (4096, 1024, 130)),
}
BITWISE_TOKENS = (31007, 32768, 1000)
MAGNITUDES = ("1e-3", "1", "1e3", "mixed", "zero+tiny groups")
TIMING_CHANNELS = (4096, 2048, 1024)
TIMING_TOKENS = (32768, 31007)
RECOMPILE_TOKENS = (31007, 32768, 4097, 1000)
SWIGLU_LIMIT = 10.0
TINY = 1e-14


def bitwise_input(tokens: int, channels: int, magnitude: str) -> torch.Tensor:
    """Random bf16, and for "zero+tiny groups" also all-zero and below-floor groups of both group shapes."""
    if magnitude != "zero+tiny groups":
        return random_bf16(tokens, channels, magnitude)
    x = random_bf16(tokens, channels, "mixed")
    x[:BLOCK] = 0
    x[BLOCK : 2 * BLOCK] = random_bf16(BLOCK, channels) * TINY
    x[2 * BLOCK : 3 * BLOCK, :BLOCK] = 0
    x[3 * BLOCK : 4 * BLOCK, BLOCK : 2 * BLOCK] = random_bf16(BLOCK, min(BLOCK, channels - BLOCK)) * TINY
    return x


def mismatches(got: tuple[torch.Tensor, torch.Tensor], want: tuple[torch.Tensor, torch.Tensor]) -> tuple[int, int]:
    """Number of differing FP8 bytes and differing fp32 scale bit patterns."""
    (fp8_got, scale_got), (fp8_want, scale_want) = got, want
    assert fp8_got.shape == fp8_want.shape and scale_got.shape == scale_want.shape
    assert fp8_got.stride() == fp8_want.stride() and scale_got.stride() == scale_want.stride()
    byte_diff = (fp8_got.view(torch.uint8) != fp8_want.view(torch.uint8)).sum().item()
    scale_diff = (scale_got.view(torch.int32) != scale_want.view(torch.int32)).sum().item()
    return byte_diff, scale_diff


def compile_dynamic(fn, example: torch.Tensor, **compile_kwargs):
    torch._dynamo.mark_dynamic(example, 0)
    return torch.compile(fn, fullgraph=True, **compile_kwargs)


def check_bitwise() -> None:
    print("\n## Bitwise vs Triton: differing FP8 bytes / differing scales (0 / 0 is bitwise equal)\n")
    print(
        f"Compiled: one dynamic-M graph per (quantizer, K), reused for M in {BITWISE_TOKENS} under error_on_recompile."
    )
    print(f"zero+tiny groups: mixed 1e-4..1e4 per token, plus all-zero and {TINY:g}-scaled groups of both shapes.\n")
    print("| quantizer | K | M | magnitude | eager | compiled |")
    print("|---|---:|---:|---|---|---|")
    total = {"eager": 0, "compiled": 0}
    for name, (exact_fn, triton_fn, channel_counts) in EXACT.items():
        for channels in channel_counts:
            torch._dynamo.reset()
            compiled = None
            for tokens in BITWISE_TOKENS:
                for magnitude in MAGNITUDES:
                    x = bitwise_input(tokens, channels, magnitude)
                    if compiled is None:
                        compiled = compile_dynamic(exact_fn, x)
                        compiled(x)
                    want = triton_fn(x)
                    with torch._dynamo.config.patch(error_on_recompile=True):
                        results = {"eager": mismatches(exact_fn(x), want), "compiled": mismatches(compiled(x), want)}
                    for mode, (byte_diff, scale_diff) in results.items():
                        total[mode] += byte_diff + scale_diff
                    cells = " | ".join(f"{b} / {s}" for b, s in results.values())
                    print(f"| {name} | {channels} | {tokens} | {magnitude} | {cells} |")
    print(f"\nTotal mismatches: eager {total['eager']}, compiled {total['compiled']}")


def check_kernels(tokens: int, channels: int) -> None:
    print(f"\n## Exact quantizer kernels, dynamic M, M={tokens} K={channels}\n")
    print("div_rn: `triton.language.div_rn` calls in the kernel. reciprocal: whether the 1/448 multiply appears.")
    print(f"Recompiles: M in {RECOMPILE_TOKENS} after the first call under error_on_recompile.\n")
    print("| quantizer | compile | kernels | heuristics | div_rn | reciprocal | recompiles |")
    print("|---|---|---:|---|---:|---|---|")
    cases = {
        ("1x128", "default"): (quant_1x128_exact, {}),
        ("128x1_tp", "default"): (quant_128x1_tp_exact, {}),
        ("128x1_tp", "Fp8GroupQuantChoices"): (quant_128x1_tp_exact, {"backend": fp8_group_quant_backend()}),
    }
    for (name, label), (fn, compile_kwargs) in cases.items():
        torch._dynamo.reset()
        x = random_bf16(tokens, channels)
        compiled = compile_dynamic(fn, x, **compile_kwargs)
        with V.set_choices_handler(InductorChoices()), torch._inductor.config.patch(force_disable_caches=True):
            with captured_graphs() as graphs:
                compiled(x)
        code = "\n".join(code for _, code in graphs)
        kernels = kernel_summaries(code)
        with torch._dynamo.config.patch(error_on_recompile=True):
            for other_tokens in RECOMPILE_TOKENS[1:]:
                compiled(random_bf16(other_tokens, channels))
        heuristics = " + ".join(k["heuristic"] + k["fixed_config"] for k in kernels)
        reciprocal = "yes" if "0.002232142857142857" in code else "no"
        print(
            f"| {name} | {label} | {len(kernels)} | {heuristics} | {code.count('triton.language.div_rn(')} "
            f"| {reciprocal} | none |"
        )


def check_timing() -> None:
    print("\n## Timing, do_bench median (lower ms is better)\n")
    print(
        "1x128 compiles with default heuristics. 128x1_tp compiles with fp8_group_quant_backend (Fp8GroupQuantChoices)."
    )
    print("ratio = exact / plain. Triton 128x1 includes the F.pad copy when M % 128 != 0.\n")
    print("| quantizer | K | M | plain ms | exact ms | ratio | Triton ms |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    quantizers = {
        "1x128": (quant_1x128, quant_1x128_exact, triton_1x128, {}),
        "128x1_tp": (quant_128x1_tp, quant_128x1_tp_exact, triton_128x1_tp, {"backend": fp8_group_quant_backend()}),
    }
    for channels in TIMING_CHANNELS:
        for name, (plain_fn, exact_fn, triton_fn, compile_kwargs) in quantizers.items():
            torch._dynamo.reset()
            plain = exact = None
            for tokens in TIMING_TOKENS:
                x = random_bf16(tokens, channels)
                if plain is None:
                    plain = compile_dynamic(plain_fn, x, **compile_kwargs)
                    exact = compile_dynamic(exact_fn, x, **compile_kwargs)
                    plain(x)
                    exact(x)
                with torch._dynamo.config.patch(error_on_recompile=True):
                    plain_ms = do_bench(lambda: plain(x), return_mode="median")
                    exact_ms = do_bench(lambda: exact(x), return_mode="median")
                triton_ms = do_bench(lambda: triton_fn(x), return_mode="median")
                print(
                    f"| {name} | {channels} | {tokens} | {plain_ms:.4f} | {exact_ms:.4f} | {exact_ms / plain_ms:.3f} "
                    f"| {triton_ms:.4f} |"
                )


def clamped_silu(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    return ClampedSilu(SWIGLU_LIMIT).apply(gate, up)


def silu_then_quant_plain(gate_up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return quant_1x128(clamped_silu(gate_up))


def silu_then_quant_exact(gate_up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return quant_1x128_exact(clamped_silu(gate_up))


def eager_silu_then_triton(gate_up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return triton_1x128(clamped_silu(gate_up))


def check_producer(tokens_list: tuple[int, ...], intermediate: int) -> None:
    print(f"\n## Compiled ClampedSilu(limit={SWIGLU_LIMIT}) then quant_1x128 vs eager ClampedSilu then Triton quant\n")
    print(f"gate_up ~ N(0, 4^2) bf16 [M, 2I], I={intermediate}. bf16 diff: elements where the compiled bf16 activation")
    print("alone differs from eager. bytes / scales: differing FP8 bytes and scales of the fused compiled kernel.")
    print(
        "Time is do_bench median of the compiled fused kernel (lower is better). Unfused = eager ClampedSilu + Triton.\n"
    )
    print(
        "| M | quantizer | emulate_precision_casts | bf16 diff | bytes / scales differing | kernels | ms | unfused ms |"
    )
    print("|---:|---|---|---:|---|---:|---:|---:|")
    for tokens in tokens_list:
        gate_up = (torch.randn(tokens, 2 * intermediate, device="cuda") * 4).bfloat16()
        reference = eager_silu_then_triton(gate_up)
        eager_activation = clamped_silu(gate_up)
        unfused_ms = do_bench(lambda: eager_silu_then_triton(gate_up), return_mode="median")
        for emulate in (False, True):
            options = {"emulate_precision_casts": emulate}
            torch._dynamo.reset()
            compiled_activation = torch.compile(clamped_silu, fullgraph=True, options=options)(gate_up)
            bf16_diff = (compiled_activation.view(torch.int16) != eager_activation.view(torch.int16)).sum().item()
            for name, fn in (("plain", silu_then_quant_plain), ("exact", silu_then_quant_exact)):
                torch._dynamo.reset()
                compiled = torch.compile(fn, fullgraph=True, options=options)
                with captured_graphs() as graphs:
                    got = compiled(gate_up)
                kernels = sum(len(kernel_summaries(code)) for _, code in graphs)
                byte_diff, scale_diff = mismatches(got, reference)
                fused_ms = do_bench(lambda: compiled(gate_up), return_mode="median")
                print(
                    f"| {tokens} | {name} | {emulate} | {bf16_diff} | {byte_diff} / {scale_diff} | {kernels} "
                    f"| {fused_ms:.4f} | {unfused_ms:.4f} |"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", default=",".join(SECTIONS), help=f"comma-separated subset of {SECTIONS}")
    parser.add_argument("--tokens", type=int, default=31007, help="M for the kernels section")
    parser.add_argument("--channels", type=int, default=4096, help="K for the kernels section")
    parser.add_argument("--producer-tokens", default="31007,190464", help="comma-separated M for the producer section")
    parser.add_argument(
        "--intermediate", type=int, default=2048, help="ClampedSilu output width I for the producer section"
    )
    args = parser.parse_args()
    sections = args.sections.split(",")
    unknown = set(sections) - set(SECTIONS)
    if unknown:
        parser.error(f"unknown sections {sorted(unknown)}")
    print(f"torch {torch.__version__}, device {torch.cuda.get_device_name()}")
    if "bitwise" in sections:
        check_bitwise()
    if "kernels" in sections:
        check_kernels(args.tokens, args.channels)
    if "timing" in sections:
        check_timing()
    if "producer" in sections:
        check_producer(tuple(int(t) for t in args.producer_tokens.split(",")), args.intermediate)


if __name__ == "__main__":
    main()
