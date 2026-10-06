"""Benchmark how `max_seqlen` is passed to the FA2/FA3 varlen custom ops.

Compares three ways of feeding `max_seqlen` to a compiled `FlashAttention` block (forward + backward):
`static` (the true longest document, specialized by Dynamo, one graph per value), `dynamic` (the true value
as a symbolic int via `torch.compiler.config.dynamic_sources`, as `apply_compile` sets it), and `upper_bound` (the total packed token count, which is what
FA4 uses when `max_seqlen` is None). It also times the eager op alone with the true value vs the upper bound.
Timing uses `triton.testing.do_bench` (CUDA events, L2 flush between reps). Each arm's forward output is
compared bitwise against the same path with the true value (eager against eager, compiled against compiled
static). Gradients are not compared: FA's GQA backward accumulates dk/dv with atomics, so they vary run to run.

Usage (from the prime-rl repo, on a GPU node):
    uv run python tools/benchmark_flash_attn_max_seqlen.py <output.json> [--versions 2 3] [--reps 200]
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
import triton

from prime_rl.trainer.models.layers.attn import ATTN_IMPL2CLASS, AttentionConfig

TOTAL_TOKENS = 16384
DOC_LAYOUTS = {
    "1x16384": [16384],
    "2 docs": [12820, 3564],
    "3 docs": [9147, 4110, 3127],
    "16x1024": [1024] * 16,
    "64x256": [256] * 64,
}
GLM_4_5_AIR = AttentionConfig(
    hidden_size=4096,
    head_dim=128,
    num_attention_heads=96,
    num_key_value_heads=8,
    is_causal=True,
    attention_bias=True,
    use_qk_norm=False,
    rms_norm_eps=1e-5,
)


def cu_seqlens_for(doc_lens: list[int]) -> torch.Tensor:
    cu_seqlens = torch.tensor([0, *doc_lens], dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32)
    torch._dynamo.mark_dynamic(cu_seqlens, 0)
    return cu_seqlens


def time_ms(fn, grad_to_none: list[torch.Tensor], reps: int) -> dict[str, float]:
    times = triton.testing.do_bench(fn, warmup=25, rep=reps, grad_to_none=grad_to_none, return_mode="all")
    return {"median_ms": statistics.median(times), "min_ms": min(times), "max_ms": max(times)}


def unique_graphs() -> int:
    return torch._dynamo.utils.counters["stats"]["unique_graphs"]


def benchmark_version(version: int, reps: int) -> list[dict]:
    torch.manual_seed(0)
    module = ATTN_IMPL2CLASS[f"flash_attention_{version}"](GLM_4_5_AIR).to(device="cuda", dtype=torch.bfloat16)
    hidden_states = torch.randn(1, TOTAL_TOKENS, GLM_4_5_AIR.hidden_size, device="cuda", dtype=torch.bfloat16)
    dout = torch.randn_like(hidden_states)
    params = list(module.parameters())

    def run(fn, x, cu_seqlens, max_seqlen):
        out, _ = fn(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        out.backward(dout)
        return out

    def forward_output(fn, cu_seqlens, max_seqlen):
        x = hidden_states.detach().clone().requires_grad_()
        return run(fn, x, cu_seqlens, max_seqlen).detach()

    def compile_dynamic_block():
        torch._dynamo.reset()
        block = torch.compile(module, fullgraph=True)
        with torch.compiler.config.patch(dynamic_sources=r".*\['max_seqlen'\]"):
            forward_output(block, cu_seqlens_for(DOC_LAYOUTS["2 docs"]), 13000)
        return block, unique_graphs()

    rows = []
    for layout, doc_lens in DOC_LAYOUTS.items():
        cu_seqlens = cu_seqlens_for(doc_lens)
        true_max = max(doc_lens)
        reference = forward_output(module, cu_seqlens, true_max)
        x = hidden_states.detach().clone().requires_grad_()
        grad_to_none = [x, *params]

        arms = {"eager true max": (module, true_max), "eager upper bound": (module, TOTAL_TOKENS)}
        for name, (fn, max_seqlen) in arms.items():
            result = time_ms(lambda: run(fn, x, cu_seqlens, max_seqlen), grad_to_none, reps)
            bitwise = torch.equal(forward_output(fn, cu_seqlens, max_seqlen), reference)
            rows.append({"version": version, "layout": layout, "arm": name, "bitwise": bitwise, **result})

        compiled_reference = None
        for name, max_seqlen in (("compiled static", true_max), ("compiled upper bound", TOTAL_TOKENS)):
            torch._dynamo.reset()
            static_block = torch.compile(module, fullgraph=True)
            compiled_output = forward_output(static_block, cu_seqlens, max_seqlen)
            compiled_reference = compiled_output if compiled_reference is None else compiled_reference
            result = time_ms(lambda: run(static_block, x, cu_seqlens, max_seqlen), grad_to_none, reps)
            bitwise = torch.equal(compiled_output, compiled_reference)
            rows.append({"version": version, "layout": layout, "arm": name, "bitwise": bitwise, **result})

        dynamic_block, dynamic_graphs = compile_dynamic_block()
        result = time_ms(lambda: run(dynamic_block, x, cu_seqlens, true_max), grad_to_none, reps)
        dynamic_output = forward_output(dynamic_block, cu_seqlens, true_max)
        assert unique_graphs() == dynamic_graphs, "dynamic arm recompiled"
        bitwise = torch.equal(dynamic_output, compiled_reference)
        rows.append({"version": version, "layout": layout, "arm": "compiled dynamic", "bitwise": bitwise, **result})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path, help="JSON file for the result rows")
    parser.add_argument("--versions", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--reps", type=int, default=200, help="do_bench measurement time per arm in ms")
    args = parser.parse_args()

    rows = [row for version in args.versions for row in benchmark_version(version, args.reps)]
    args.output.write_text(json.dumps({"device": torch.cuda.get_device_name(), "rows": rows}, indent=2))
    print(f"{'FA':>2} {'layout':<8} {'arm':<21} {'median ms (lower)':>17} {'min ms':>8} {'max ms':>8} fwd bitwise")
    for row in rows:
        print(
            f"{row['version']:>2} {row['layout']:<8} {row['arm']:<21} {row['median_ms']:>17.3f} "
            f"{row['min_ms']:>8.3f} {row['max_ms']:>8.3f} {row['bitwise']}"
        )


if __name__ == "__main__":
    main()
