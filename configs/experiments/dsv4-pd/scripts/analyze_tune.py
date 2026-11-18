"""Tuning results per GPU: one row per run, GPUs counted from the engines each window scraped.

    analyze_tune.py stress/<run>.jsonl [...]
"""

import json
import sys
from pathlib import Path


def summary(path: Path) -> dict | None:
    rows = [json.loads(line) for line in path.open()]
    windows = [r for r in rows if r["kind"] == "window" and r.get("phase", "hold") == "hold"]
    if not windows:
        return None
    half = windows[len(windows) // 2 :]
    roles = half[0]["roles"]
    gpus = max(w["roles"]["all"]["engines"] for w in half)

    def mean(role, key):
        vals = [w["roles"][role][key] for w in half if w["roles"].get(role, {}).get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    gen_role = "decode" if "decode" in roles else "all"
    hit_role = "prefill" if "prefill" in roles else "all"
    prompt = mean(hit_role, "prompt_tok_s")
    return {
        "run": path.stem,
        "level": windows[0]["level"],
        "minutes": windows[-1]["level_min"],
        "gpus": gpus,
        "out_per_gpu": mean(gen_role, "gen_tok_s") / gpus,
        "calls": mean("all", "req_done_s") / (2 if "decode" in roles else 1),
        "recompute": mean("all", "prefill_computed_tok_s") / prompt if prompt else None,
        "prefill_waiting": mean("prefill", "waiting") if "prefill" in roles else mean("all", "waiting"),
        "decode_waiting": mean("decode", "waiting") if "decode" in roles else None,
        "ttft": mean("all", "ttft_mean"),
        "itl": mean(gen_role, "itl_mean"),
    }


print("| run | episodes | GPUs | hold min | out tok/s/GPU | calls/s | recompute | prefill wait | decode wait | TTFT s | ITL ms |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for p in map(Path, sys.argv[1:]):
    s = summary(p)
    if s is None:
        print(f"| {p.stem} | no hold data yet |||||||||")
        continue
    dw = "-" if s["decode_waiting"] is None else f"{s['decode_waiting']:.0f}"
    print(f"| {s['run']} | {s['level']} | {s['gpus']} | {s['minutes']:.0f} | {s['out_per_gpu']:,.0f} | {s['calls']:.1f} | "
          f"{s['recompute']:.1%} | {s['prefill_waiting']:.0f} | {dw} | {s['ttft']:.1f} | {s['itl'] * 1000:.0f} |")
