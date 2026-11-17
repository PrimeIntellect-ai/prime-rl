"""Summarize an inference_fit.sh sweep as a markdown table.

python benchmarks/recipes/summarize_inference.py <out>
"""

import json
import sys
from pathlib import Path


def fmt(value, spec: str) -> str:
    return spec.format(value) if value is not None else "-"


def main(out: Path) -> None:
    print(
        "| variant | conc | startup s | out tok/s/GPU | engine prefill tok/s | prefix hit | KV mean | KV max "
        "| waiting | preempt | TTFT p50/p99 s | ITL p50/p99 ms | errors |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for run in sorted(p for p in out.iterdir() if p.is_dir()):
        startup = (run / "startup_s").read_text().strip() if (run / "startup_s").exists() else "-"
        results = sorted(run.glob("c*.json"), key=lambda p: int(p.stem[1:]))
        if not results:
            status = (run / "status").read_text().strip() if (run / "status").exists() else "?"
            print(f"| {run.name} | - | {startup} | {status} | | | | | | | | | |")
            continue
        for path in results:
            r = json.loads(path.read_text())
            print(
                f"| {run.name} | {r['concurrency']} | {startup} | {r['output_tok_s_per_gpu']:,.0f} "
                f"| {r['engine_prompt_tok_s']:,.0f} | {fmt(r['prefix_cache_hit_rate'], '{:.2f}')} "
                f"| {fmt(r['kv_usage_mean'], '{:.2f}')} | {fmt(r['kv_usage_max'], '{:.2f}')} "
                f"| {fmt(r['waiting_mean'], '{:.0f}')} | {r['preemptions']:.0f} "
                f"| {fmt(r['ttft_p50_s'], '{:.1f}')}/{fmt(r['ttft_p99_s'], '{:.1f}')} "
                f"| {r['itl_p50_ms']:.0f}/{r['itl_p99_ms']:.0f} | {r['errors']} |"
            )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
