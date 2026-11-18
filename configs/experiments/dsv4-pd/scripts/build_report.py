"""Fill report/template.html with the current results and write report/index.html.

    build_report.py
"""

import json
import time
from pathlib import Path

from analyze_stress import GPUS
from pr_results import ARMS, STRESS, in_progress_levels, level_rows

REPORT = Path("/home/phi/exp/dsv4-pd/report")
KEYS = ("gen", "calls", "recompute", "ttft", "itl", "gpu_hit", "ext_hit", "waiting", "running", "partial")


def burst_series(arm: str) -> list[list[float]]:
    """Per-minute [minute, out tok/s/GPU, recompute share, waiting, prefill-side hit rate] of the 2048 burst."""
    out = []
    for line in (STRESS / f"stress-{arm}-desc.jsonl").open():
        r = json.loads(line)
        if r["kind"] != "window" or r["level"] != 2048 or r.get("phase", "hold") != "hold":
            continue
        roles = r["roles"]
        a, d = roles["all"], roles.get("decode") or roles["all"]
        p = roles.get("prefill") or roles.get("agg") or a
        recompute = a["prefill_computed_tok_s"] / p["prompt_tok_s"] if p["prompt_tok_s"] else None
        out.append([round(r["level_min"], 1), round(d["gen_tok_s"] / GPUS), recompute and round(min(recompute, 1.0), 3),
                    a["waiting"], p["gpu_hit_rate"] and round(p["gpu_hit_rate"], 3)])
    return out


levels = []
for level, arms in sorted(level_rows().items()):
    levels.append({"level": level, "arms": {arm: {k: arms[arm][k] for k in KEYS} for arm in ARMS if arm in arms}})
for entry in levels:
    for row in entry["arms"].values():
        row["gen"] = round(row["gen"] / GPUS)
in_progress = sorted(in_progress_levels())
data = {
    "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
    "in_progress": in_progress,
    "levels": levels,
    "burst": {arm: burst_series(arm) for arm in ARMS},
}
html = (REPORT / "template.html").read_text().replace("/*__DATA__*/{}", json.dumps(data, separators=(",", ":")))
(REPORT / "index.html").write_text(html)
print(REPORT / "index.html", len(html), "bytes;", [e["level"] for e in levels])
