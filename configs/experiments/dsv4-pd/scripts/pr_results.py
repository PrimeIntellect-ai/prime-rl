"""Rebuild the stress-test tables in the PR body from every level's driver data, between the
<!-- stress-results:start/end --> markers, and print the new body path.

    pr_results.py <pr_body.md>
"""

import sys
import time
from pathlib import Path

from analyze_stress import GPUS, levels

STRESS = Path("/home/phi/exp/dsv4-pd/stress")
ARMS = ("agg", "pd", "pdmc")
START, END = "<!-- stress-results:start -->", "<!-- stress-results:end -->"


def in_progress_levels() -> set[int]:
    """Levels whose driver is still running: no end marker, and its log written in the last 10 minutes."""
    return {
        int(log.stem.rsplit("-L", 1)[1])
        for log in Path("/home/phi/exp/dsv4-pd/logs").glob("driver_stress-*-L*.log")
        if "=== schedule done" not in log.read_text() and time.time() - log.stat().st_mtime < 600
    }


def level_rows() -> dict[int, dict[str, dict]]:
    """level -> arm -> row, from the descending 2048 run, the per-level bursts and the ascending level 32."""
    out: dict[int, dict[str, dict]] = {}
    for arm in ARMS:
        files = [STRESS / f"stress-{arm}.jsonl", STRESS / f"stress-{arm}-desc.jsonl", *STRESS.glob(f"stress-{arm}-L*.jsonl")]
        for path in files:
            if not path.exists():
                continue
            for row in levels(path):
                level = row["level"] if isinstance(row["level"], int) else int(str(row["level"]).split()[0])
                if path.name == f"stress-{arm}.jsonl" and level != 32:
                    continue  # the ascending run only completed level 32
                row["partial"] = not isinstance(row["level"], int)
                out.setdefault(level, {})[arm] = row
    return out


RUNNING = in_progress_levels()


def cell(row: dict | None, text: str, level: int) -> str:
    if row is None:
        return "–"
    if level in RUNNING:
        return text + " (running)"
    return text + (" (partial)" if row["partial"] else "")


def tables() -> str:
    rows = level_rows()
    lines = ["**Output tok/s/GPU** (model calls/s in parentheses):", "", "| Concurrency | agg | pd | pdmc |", "|---|---|---|---|"]
    for level in sorted(rows, reverse=True):
        cells = []
        for arm in ARMS:
            r = rows[level].get(arm)
            text = "" if r is None else f"{r['gen'] / GPUS:,.0f} ({r['calls']:.1f})"
            if r is not None and r["recompute"] is not None and r["recompute"] >= 0.25:
                text = f"**{text}, thrashed**"
            cells.append(cell(r, text, level))
        lines.append(f"| {level} | " + " | ".join(cells) + " |")
    lines += ["", "**Recomputed prompt share / time to first token / inter-token latency:**", "",
              "| Concurrency | agg | pd | pdmc |", "|---|---|---|---|"]
    for level in sorted(rows, reverse=True):
        cells = []
        for arm in ARMS:
            r = rows[level].get(arm)
            if r is None:
                cells.append("–")
                continue
            itl = f"{r['itl'] * 1000:.0f} ms" if r["itl"] and r["running"] >= 100 else "–"
            cells.append(cell(r, f"{r['recompute']:.1%} / {r['ttft']:.1f} s / {itl}", level))
        lines.append(f"| {level} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    body_path = Path(sys.argv[1])
    body = body_path.read_text()
    head, rest = body.split(START)
    _, tail = rest.split(END)
    body_path.write_text(f"{head}{START}\n{tables()}\n{END}{tail}")
    print(body_path)
