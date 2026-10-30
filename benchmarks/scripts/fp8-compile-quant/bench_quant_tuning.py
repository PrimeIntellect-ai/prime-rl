"""Compare ways to compile the 128x1 FP8 quantizer to a single-read persistent kernel, and check they reach backward compiles."""

from __future__ import annotations

import argparse
import re
import sys
import threading
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch._inductor.choices import InductorChoices
from torch._inductor.graph import GraphLowering
from torch._inductor.virtualized import V
from triton.testing import do_bench

from prime_rl.trainer.models.layers.activations import ClampedSilu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_quant import PersistentGroupReductions, random_bf16, triton_1x128, triton_128x1_tp  # noqa: E402
from quant_choices import Fp8GroupQuantChoices, Fp8GroupQuantChoicesBothLayouts, fp8_group_quant_backend  # noqa: E402
from quant_torch import AMAX_FLOOR, BLOCK, FP8_MAX, TMA_ALIGNMENT_FP32, quant_1x128, quant_128x1_tp  # noqa: E402

SECTIONS = ("install", "consumer", "variants", "contiguous", "producers")
CHANNELS = (4096, 2048, 1024)
TOKENS = (32768, 31007)
SWIGLU_LIMIT = 10.0
PRODUCER_INTERMEDIATE = 2048


def quant_128x1_tp_halves(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`quant_128x1_tp` with each group's amax taken over 64 pairs, so default heuristics pick a persistent kernel."""
    tokens, channels = x.shape
    padded_tokens = (tokens + BLOCK - 1) // BLOCK * BLOCK
    token_groups = padded_tokens // BLOCK
    x = F.pad(x, (0, 0, 0, padded_tokens - tokens)).float()
    halves = x.reshape(token_groups, 2, BLOCK // 2, channels)
    low, high = halves[:, 0], halves[:, 1]
    amax = torch.maximum(low.abs(), high.abs()).amax(dim=1).clamp(min=AMAX_FLOOR)
    amax_t = amax.t()

    def cast(half: torch.Tensor) -> torch.Tensor:
        return (
            (half.permute(2, 0, 1) / (amax_t / FP8_MAX).unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        )

    # The casts read amax, not a realized scale buffer, so Inductor writes them into the stack buffer in the amax kernel.
    x_fp8_t = torch.stack((cast(low), cast(high)), dim=2)
    scale = amax / FP8_MAX
    tma_aligned_channels = (channels + TMA_ALIGNMENT_FP32 - 1) // TMA_ALIGNMENT_FP32 * TMA_ALIGNMENT_FP32
    if tma_aligned_channels != channels:
        scale = F.pad(scale, (0, tma_aligned_channels - channels))[:, :channels]
    return x_fp8_t.view(channels, padded_tokens), scale.t()


VARIANTS = {
    "plain, default": (quant_128x1_tp, {"options": {}}, InductorChoices),
    "(1) halves, default": (quant_128x1_tp_halves, {"options": {}}, InductorChoices),
    "(1) halves, coord descent": (
        quant_128x1_tp_halves,
        {"options": {"coordinate_descent_tuning": True}},
        InductorChoices,
    ),
    "(2) multi_kernel": (quant_128x1_tp, {"options": {"triton.multi_kernel": 1}}, InductorChoices),
    "(2) multi_kernel, coord descent": (
        quant_128x1_tp,
        {"options": {"triton.multi_kernel": 1, "coordinate_descent_tuning": True}},
        InductorChoices,
    ),
    "prior: <=128 persistent, coord descent": (
        quant_128x1_tp,
        {"options": {"inductor_choices_class": PersistentGroupReductions, "coordinate_descent_tuning": True}},
        PersistentGroupReductions,
    ),
    "(3) Fp8GroupQuantChoices backend": (quant_128x1_tp, {"backend": fp8_group_quant_backend()}, None),
}


@contextmanager
def captured_graphs():
    """Collect (thread name, output code) of every Inductor graph compiled inside the context, on any thread."""
    graphs: list[tuple[str, str]] = []

    def save_output_code(code: str) -> None:
        graphs.append((threading.current_thread().name, code))

    original = GraphLowering.save_output_code
    GraphLowering.save_output_code = staticmethod(save_output_code)
    try:
        yield graphs
    finally:
        GraphLowering.save_output_code = original


def kernel_summaries(code: str) -> list[dict]:
    summaries = []
    for name, source in re.findall(r"^(triton_\w+) = async_compile\.triton\('\w+', '''(.*?)'''", code, re.S | re.M):
        signature = re.search(r"'signature': \{([^}]*)\}", source).group(1)
        bf16_pointers = re.findall(r"'(\w+)': '\*bf16'", signature)
        loads = [
            index for pointer in bf16_pointers for index in re.findall(rf"tl\.load\({pointer} \+ \(([^)]*)\)", source)
        ]
        fixed = re.search(r"config=(\{[^}]*\})", source)
        summaries.append(
            {
                "name": name,
                "heuristic": re.search(r"@triton_heuristics\.(\w+)", source).group(1),
                "stores_fp8": "*fp8e4nv" in signature,
                "bf16_loads": loads,
                "fixed_config": fixed.group(1) if fixed else "",
            }
        )
    return summaries


def describe_kernels(code: str) -> tuple[int, str, str]:
    """Kernel launch count, whether one kernel loads bf16 x with no repeated load index, and kernel heuristics."""
    kernels = kernel_summaries(code)
    launches = len(re.findall(r"\b((?:triton|multi_kernel)_\w+)\.run\(", code))
    readers = [k for k in kernels if k["bf16_loads"]]
    if "async_compile.multi_kernel(" in code:
        reads_once = "runtime pick"
    elif len(readers) == 1 and len(set(readers[0]["bf16_loads"])) == len(readers[0]["bf16_loads"]):
        reads_once = "yes"
    else:
        reads_once = "no"
    heuristics = " + ".join(k["heuristic"] + k["fixed_config"] for k in kernels)
    return launches, reads_once, heuristics


def check_variants() -> None:
    print("\n## 128x1 quantizer: compile strategies vs Triton (do_bench median, lower ms is better)\n")
    print("compile s: wall time of the first call (Dynamo, Inductor, autotuning) with caches disabled, M=32768.")
    print("reads x once: one kernel loads the bf16 input and no load index repeats (a looped reduction loads twice).")
    print(
        "Triton includes the F.pad copy when M % 128 != 0, as in fp8_blockwise_mm_backward. ratio = variant / Triton."
    )
    print("The M=31007 call reuses the M=32768 graph under error_on_recompile.\n")
    print(
        "| variant | K | launches | reads x once | compile s | M=32768 ms | Triton | ratio | M=31007 ms | Triton | ratio |"
    )
    print("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    torch.compile(lambda a: a.sin() + 1)(torch.randn(16, device="cuda"))
    for channels in CHANNELS:
        inputs = {tokens: random_bf16(tokens, channels) for tokens in TOKENS}
        triton_ms = {tokens: do_bench(lambda: triton_128x1_tp(x), return_mode="median") for tokens, x in inputs.items()}
        reference = triton_128x1_tp(inputs[TOKENS[0]])
        for label, (fn, compile_kwargs, choices_class) in VARIANTS.items():
            torch._dynamo.reset()
            first = inputs[TOKENS[0]]
            torch._dynamo.mark_dynamic(first, 0)
            compiled = torch.compile(fn, fullgraph=True, **compile_kwargs)
            handler = V.set_choices_handler(choices_class()) if choices_class else nullcontext()
            start = time.perf_counter()
            with handler, torch._inductor.config.patch(force_disable_caches=True), captured_graphs() as graphs:
                outputs = compiled(first)
                torch.cuda.synchronize()
            compile_s = time.perf_counter() - start
            launches, reads_once, heuristics = describe_kernels("\n".join(code for _, code in graphs))
            for got, want in zip(outputs, reference):
                assert got.shape == want.shape and got.stride() == want.stride(), (label, got.shape, want.shape)
            byte_equal = (outputs[0].view(torch.uint8) == reference[0].view(torch.uint8)).float().mean().item()
            assert byte_equal > 0.99, (label, byte_equal)
            with torch._dynamo.config.patch(error_on_recompile=True):
                times = {tokens: do_bench(lambda: compiled(x), return_mode="median") for tokens, x in inputs.items()}
            row = " | ".join(
                f"{times[tokens]:.4f} | {triton_ms[tokens]:.4f} | {times[tokens] / triton_ms[tokens]:.2f}"
                for tokens in TOKENS
            )
            print(f"| {label} | {channels} | {launches} | {reads_once} | {compile_s:.1f} | {row} |")
            print(f"|   kernels: {heuristics} | | | | | | | | | | |")


def quant_128x1_tp_3d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`quant_128x1_tp` with FP8 viewed back to contiguous `[channels, token_groups, 128]` for an opaque consumer."""
    x_fp8_t, scale_t = quant_128x1_tp(x)
    return x_fp8_t.view(x_fp8_t.size(0), -1, BLOCK), scale_t


@torch.library.custom_op("prime_rl_bench::fp8_consume", mutates_args=())
def fp8_consume(x_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Stand-in for a pre-quantized GEMM op: opaque, default fixed-stride-order layout, and launches no kernel."""
    return torch.empty((), device=x_fp8.device)


@fp8_consume.register_fake
def _(x_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x_fp8.new_empty((), dtype=torch.float32)


@torch.library.custom_op("prime_rl_bench::fp8_consume_flexible", mutates_args=(), tags=(torch.Tag.flexible_layout,))
def fp8_consume_flexible(x_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """`fp8_consume` tagged `flexible_layout`, so Inductor may pass any strides."""
    return torch.empty((), device=x_fp8.device)


@fp8_consume_flexible.register_fake
def _(x_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x_fp8.new_empty((), dtype=torch.float32)


def quant_1x128_3d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`quant_1x128` with FP8 viewed as `[tokens, channel_groups, 128]` for an opaque consumer."""
    x_fp8, scale = quant_1x128(x)
    return x_fp8.view(x_fp8.size(0), -1, BLOCK), scale


def check_consumer(tokens: int, channels: int) -> None:
    print(f"\n## Quantizers feeding an opaque custom op, dynamic M, M={tokens} K={channels}\n")
    print("Graph outputs (the variants section) let AOTAutograd return the 3D FP8 base. An opaque consumer instead")
    print("makes Inductor realize the FP8 view it receives, with the strides of the fake tensor.")
    print("1x128 rows quantize ClampedSilu(gate_up) and compile with emulate_precision_casts=True.\n")
    print("| quantizer | FP8 tensor handed to the op | op layout | compile | launches | reads input once | kernels |")
    print("|---|---|---|---|---:|---|---|")
    emulate = {"emulate_precision_casts": True}
    cases = [
        ("128x1", "2D [K, Mp] view", "fixed stride order", lambda x: fp8_consume(*quant_128x1_tp(x)), {}),
        ("128x1", "3D [K, G, 128] view", "fixed stride order", lambda x: fp8_consume(*quant_128x1_tp_3d(x)), {}),
        ("128x1", "2D [K, Mp] view", "flexible_layout", lambda x: fp8_consume_flexible(*quant_128x1_tp(x)), {}),
        (
            "1x128",
            "2D [M, K] view",
            "fixed stride order",
            lambda g: fp8_consume(*quant_1x128(clamped_silu(g))),
            emulate,
        ),
        (
            "1x128",
            "3D [M, K/128, 128] view",
            "fixed stride order",
            lambda g: fp8_consume(*quant_1x128_3d(clamped_silu(g))),
            emulate,
        ),
        (
            "1x128",
            "2D [M, K] view",
            "flexible_layout",
            lambda g: fp8_consume_flexible(*quant_1x128(clamped_silu(g))),
            emulate,
        ),
    ]
    for compile_label, backend in (("default", "inductor"), ("Fp8GroupQuantChoices", fp8_group_quant_backend())):
        for quantizer, handed, constraint, fn, options in cases:
            torch._dynamo.reset()
            x = random_bf16(tokens, channels) * (4 if quantizer == "1x128" else 1)
            torch._dynamo.mark_dynamic(x, 0)
            compiled = torch.compile(fn, fullgraph=True, backend=backend, options=options or None)
            with V.set_choices_handler(InductorChoices()), torch._inductor.config.patch(force_disable_caches=True):
                with captured_graphs() as graphs:
                    compiled(x)
            launches, reads_once, heuristics = describe_kernels("\n".join(code for _, code in graphs))
            print(
                f"| {quantizer} | {handed} | {constraint} | {compile_label} | {launches} | {reads_once} | {heuristics} |"
            )


@torch.library.custom_op("prime_rl_bench::fp8_dequant_tp", mutates_args=())
def fp8_dequant_tp(x_fp8_t: torch.Tensor, scale_t: torch.Tensor) -> torch.Tensor:
    """Dequantize a `quant_128x1_tp_3d` result to fp32 `[channels, padded_tokens]`."""
    return x_fp8_t.flatten(1).float() * scale_t.repeat_interleave(BLOCK, dim=1)


@fp8_dequant_tp.register_fake
def _(x_fp8_t: torch.Tensor, scale_t: torch.Tensor) -> torch.Tensor:
    return x_fp8_t.new_empty((x_fp8_t.size(0), x_fp8_t.size(1) * x_fp8_t.size(2)), dtype=torch.float32)


def quant_dequant_128x1(x: torch.Tensor) -> torch.Tensor:
    return fp8_dequant_tp(*quant_128x1_tp_3d(x)).t()[: x.size(0)].to(x.dtype)


class QuantDequantBothWays(torch.autograd.Function):
    """Quantize-dequantize in 128x1 groups in forward, and the gradient the same way in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return quant_dequant_128x1(x)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        return quant_dequant_128x1(grad)


def quant_both_ways_loss(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (QuantDequantBothWays.apply(x) * weight).sum()


def unrelated_loss(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (x.sin() * weight).sum()


def run_forward_backward(fn, tokens: int, channels: int, dynamic: bool) -> list[tuple[str, str]]:
    x = random_bf16(tokens, channels).requires_grad_()
    weight = random_bf16(tokens, channels)
    if dynamic:
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(weight, 0)
    with torch._inductor.config.patch(force_disable_caches=True), captured_graphs() as graphs:
        fn(x, weight).backward()
        torch.cuda.synchronize()
    return graphs


def check_install(tokens: int, channels: int) -> None:
    print("\n## Installing the choices for forward and backward compiles\n")
    print("Before each case, an unrelated fwd+bwd compile initializes Inductor's per-thread choices handler on the")
    print("main thread and the autograd thread. With dynamic M, AOTAutograd compiles the backward eagerly on the main")
    print(
        "thread because symints are saved for it. With static M, it compiles the backward lazily on the autograd thread."
    )
    print(
        "Graphs are listed in compile order (forward first). picked up: every FP8-storing kernel has the fixed config.\n"
    )
    print("| install | M | graph thread | FP8 kernel heuristics | picked up |")
    print("|---|---|---|---|---|")
    cases = {
        "compile(options={inductor_choices_class})": {"options": {"inductor_choices_class": Fp8GroupQuantChoices}},
        "compile(backend=fp8_group_quant_backend())": {"backend": fp8_group_quant_backend()},
    }
    for label, compile_kwargs in cases.items():
        for dynamic in (True, False):
            torch._dynamo.reset()
            run_forward_backward(torch.compile(unrelated_loss, fullgraph=True), tokens, channels, dynamic)
            compiled = torch.compile(quant_both_ways_loss, fullgraph=True, **compile_kwargs)
            for thread_name, code in run_forward_backward(compiled, tokens, channels, dynamic):
                fp8_kernels = [k for k in kernel_summaries(code) if k["stores_fp8"]]
                heuristics = ", ".join(k["heuristic"] + k["fixed_config"] for k in fp8_kernels)
                picked_up = bool(fp8_kernels) and all(k["heuristic"] == "fixed_config" for k in fp8_kernels)
                shape = "dynamic" if dynamic else "static"
                print(f"| {label} | {shape} | {thread_name} | {heuristics} | {'yes' if picked_up else 'NO'} |")


def check_contiguous_config() -> None:
    print("\n## 1x128 quantizer with the optional contiguous fixed config (do_bench median, lower ms is better)\n")
    print("| K | M | default ms | Fp8GroupQuantChoicesBothLayouts ms | Triton ms |")
    print("|---:|---:|---:|---:|---:|")
    for channels in CHANNELS:
        torch._dynamo.reset()
        default = torch.compile(quant_1x128, fullgraph=True)
        fixed = torch.compile(
            quant_1x128, fullgraph=True, backend=fp8_group_quant_backend(Fp8GroupQuantChoicesBothLayouts)
        )
        for tokens in (*TOKENS, 190464):
            x = random_bf16(tokens, channels)
            torch._dynamo.mark_dynamic(x, 0)
            with V.set_choices_handler(InductorChoices()):
                default(x)
            fixed(x)
            default_ms = do_bench(lambda: default(x), return_mode="median")
            fixed_ms = do_bench(lambda: fixed(x), return_mode="median")
            triton_ms = do_bench(lambda: triton_1x128(x), return_mode="median")
            print(f"| {channels} | {tokens} | {default_ms:.4f} | {fixed_ms:.4f} | {triton_ms:.4f} |")


def clamped_silu(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    return ClampedSilu(SWIGLU_LIMIT).apply(gate, up)


PRODUCER_CASES = {
    "ClampedSilu -> 1x128": (
        lambda gate_up: fp8_consume(*quant_1x128(clamped_silu(gate_up))),
        lambda gate_up: triton_1x128(clamped_silu(gate_up)),
        2 * PRODUCER_INTERMEDIATE,
    ),
    "ClampedSilu -> 128x1": (
        lambda gate_up: fp8_consume(*quant_128x1_tp_3d(clamped_silu(gate_up))),
        lambda gate_up: triton_128x1_tp(clamped_silu(gate_up)),
        2 * PRODUCER_INTERMEDIATE,
    ),
    "dY -> 1x128 and 128x1": (
        lambda dy: fp8_consume(*quant_1x128(dy)).add(fp8_consume(*quant_128x1_tp_3d(dy))),
        lambda dy: (triton_1x128(dy), triton_128x1_tp(dy)),
        CHANNELS[0],
    ),
}


def check_producers(tokens: int) -> None:
    print(
        f"\n## Quantizers fused with producers, consumed by an opaque op, dynamic M, M={tokens} (lower ms is better)\n"
    )
    print(f"gate_up ~ N(0, 4^2) bf16 [M, {2 * PRODUCER_INTERMEDIATE}], dY ~ N(0, 1) bf16 [M, {CHANNELS[0]}].")
    print("unfused: eager producer then the Triton quant kernels. Kernels lists every Triton kernel of the graph.\n")
    print("| case | emulate casts | choices | launches | ms | unfused ms | kernels |")
    print("|---|---|---|---:|---:|---:|---|")
    backends = {
        "default": "inductor",
        "Fp8GroupQuantChoices": fp8_group_quant_backend(Fp8GroupQuantChoices),
        "BothLayouts": fp8_group_quant_backend(Fp8GroupQuantChoicesBothLayouts),
    }
    for name, (fused_fn, unfused_fn, channels) in PRODUCER_CASES.items():
        x = (torch.randn(tokens, channels, device="cuda") * (4 if "ClampedSilu" in name else 1)).bfloat16()
        torch._dynamo.mark_dynamic(x, 0)
        unfused_ms = do_bench(lambda: unfused_fn(x), return_mode="median")
        for emulate in (False, True):
            for label, backend in backends.items():
                torch._dynamo.reset()
                options = {"emulate_precision_casts": emulate}
                compiled = torch.compile(fused_fn, fullgraph=True, backend=backend, options=options)
                with V.set_choices_handler(InductorChoices()), torch._inductor.config.patch(force_disable_caches=True):
                    with captured_graphs() as graphs:
                        compiled(x)
                launches, _, heuristics = describe_kernels("\n".join(code for _, code in graphs))
                fused_ms = do_bench(lambda: compiled(x), return_mode="median")
                print(
                    f"| {name} | {emulate} | {label} | {launches} | {fused_ms:.4f} | {unfused_ms:.4f} | {heuristics} |"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", default=",".join(SECTIONS), help=f"comma-separated subset of {SECTIONS}")
    args = parser.parse_args()
    sections = args.sections.split(",")
    unknown = set(sections) - set(SECTIONS)
    if unknown:
        parser.error(f"unknown sections {sorted(unknown)}")
    print(f"torch {torch.__version__}, device {torch.cuda.get_device_name()}")
    if "install" in sections:
        check_install(4096, CHANNELS[0])
    if "consumer" in sections:
        check_consumer(TOKENS[1], CHANNELS[0])
    if "variants" in sections:
        check_variants()
    if "contiguous" in sections:
        check_contiguous_config()
    if "producers" in sections:
        check_producers(TOKENS[1])


if __name__ == "__main__":
    main()
