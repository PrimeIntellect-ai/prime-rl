"""Collect every A/B run under an output root into parity.md and parity.json."""

import argparse
import itertools
import json
import re
import statistics
import time
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

from graph_breaks import recompile_events, resolve_run, table

HARNESS_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = Path("/home/garrett/tmp/ngram-module/ab")
METRICS = {
    "loss/mean": "loss",
    "optim/grad_norm": "grad_norm",
    "time/step": "step_time",
    "perf/peak_memory": "peak_mem",
}
BASELINE_ARM = "main"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
LOGURU = re.compile(r"(\d\d:\d\d:\d\d)\s+(?:INFO|SUCCESS|WARNING|DEBUG|ERROR)\s")
STARTUP_MARKERS = {
    "launcher_ready": ("launcher_log", "Starting SFT run"),
    "trainer_ready": ("trainer_log", "Starting SFT trainer in World"),
    "loop_start": ("trainer_log", "Starting training loop"),
}


def load_steps(metrics_path: Path) -> dict[int, dict]:
    steps: dict[int, dict] = defaultdict(dict)
    if not metrics_path.exists():
        return {}
    for line in metrics_path.open():
        record = json.loads(line)
        for key, name in METRICS.items():
            if key in record:
                steps[record["step"]][name] = record[key]
        if "time/step" in record:
            steps[record["step"]]["end"] = record["time"]
    return dict(sorted(steps.items()))


def marker_epoch(path: Path, text: str, start: float) -> float | None:
    if not path.exists():
        return None
    day = time.strftime("%Y-%m-%d", time.localtime(start))
    for line in path.open(errors="replace"):
        line = ANSI.sub("", line)
        if text in line and (match := LOGURU.search(line)):
            epoch = time.mktime(time.strptime(f"{day} {match.group(1)}", "%Y-%m-%d %H:%M:%S"))
            return epoch + 86400 if epoch < start - 1 else epoch
    return None


def startup(meta: dict, first_step_end: float | None) -> dict:
    start = meta["start_time"]
    marks = {name: marker_epoch(Path(meta[log]), text, start) for name, (log, text) in STARTUP_MARKERS.items()}
    points = [start, marks["launcher_ready"], marks["trainer_ready"], marks["loop_start"], first_step_end]
    names = ["launcher_s", "trainer_import_s", "setup_s", "first_step_s"]
    return {
        name: (b - a if a is not None and b is not None else None) for name, a, b in zip(names, points, points[1:])
    } | {"wall_s": meta["end_time"] - start if meta.get("end_time") else None}


def spread(values: list[float]) -> dict | None:
    if not values:
        return None
    return {"n": len(values), "median": statistics.median(values), "min": min(values), "max": max(values)}


def collect_run(out_dir: Path) -> dict:
    run = resolve_run(out_dir)
    meta = run.meta
    steps = load_steps(Path(meta["metrics"]))
    ends = [(step, values["end"]) for step, values in steps.items() if "end" in values]

    def step_of(epoch: float) -> int:
        index = bisect_left([end for _, end in ends], epoch)
        return ends[index][0] if index < len(ends) else (ends[-1][0] + 1 if ends else 1)

    events = recompile_events(run)
    recompiles_per_rank: dict[int, int] = defaultdict(int)
    recompile_steps = []
    for event in events:
        recompiles_per_rank[event["rank"]] += 1
        recompile_steps.append(step_of(event["epoch"]))
    last_recompile = max(recompile_steps) if recompile_steps else None
    post_start = max(1, last_recompile or 1) + 1
    times = {step: values["step_time"] for step, values in steps.items() if "step_time" in values}
    peak = [values["peak_mem"] for values in steps.values() if "peak_mem" in values]
    errors = [error for error in (meta.get("rank_errors") or {}).values() if error]
    return {
        "arm": meta["arm"],
        "cell": meta["cell_dir"],
        "repeat": meta["repeat"],
        "head": meta["head"],
        "out_dir": str(out_dir),
        "num_gpus": meta["num_gpus"],
        "requested_steps": meta["steps"],
        "venv": meta.get("venv"),
        "exit_code": meta["exit_code"],
        "error": errors[0] if errors else None,
        "startup": startup(meta, ends[0][1] if ends else None),
        "steps": {step: {k: v for k, v in values.items() if k != "end"} for step, values in steps.items()},
        "step_time_all": spread(list(times.values())),
        "post_compile_from_step": post_start,
        "step_time_post_compile": spread([t for step, t in times.items() if step >= post_start]),
        "peak_mem_gib": max(peak) if peak else None,
        "recompiles_per_rank": dict(sorted(recompiles_per_rank.items())),
        "recompile_steps": sorted(set(recompile_steps)),
        "last_recompile_step": last_recompile,
    }


def pair_gap(x: dict, y: dict) -> dict | None:
    common = sorted(
        step
        for step in set(x["steps"]) & set(y["steps"])
        if {"loss", "grad_norm"} <= x["steps"][step].keys() & y["steps"][step].keys()
    )
    if not common:
        return None
    half = len(common) // 2
    windows = {"all": common, "first": common[:half] or common, "second": common[half:]}

    def loss_gap(steps):
        return max(abs(x["steps"][s]["loss"] - y["steps"][s]["loss"]) for s in steps)

    def grad_gap(steps):
        return max(
            abs(x["steps"][s]["grad_norm"] - y["steps"][s]["grad_norm"]) / abs(y["steps"][s]["grad_norm"])
            for s in steps
        )

    return {
        "steps": len(common),
        **{f"loss_{name}": loss_gap(steps) for name, steps in windows.items()},
        **{f"grad_{name}": grad_gap(steps) for name, steps in windows.items()},
    }


def worst(gaps: list[dict]) -> dict | None:
    gaps = [g for g in gaps if g]
    if not gaps:
        return None
    return {"pairs": len(gaps), "steps": min(g["steps"] for g in gaps)} | {
        key: max(g[key] for g in gaps) for key in gaps[0] if key != "steps"
    }


def cell_parity(runs: list[dict]) -> list[dict]:
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        by_arm[run["arm"]].append(run)
    arms = sorted(by_arm, key=lambda arm: (arm != BASELINE_ARM, arm))
    rows = []
    for base, other in itertools.combinations(arms, 2):
        gaps = [pair_gap(x, y) for x in by_arm[other] for y in by_arm[base]]
        rows.append({"comparison": f"{other} vs {base}", **(worst(gaps) or {})})
    for arm in arms:
        gaps = [pair_gap(x, y) for y, x in itertools.combinations(by_arm[arm], 2)]
        rows.append({"comparison": f"{arm} vs {arm} (floor)", **(worst(gaps) or {})})
    return rows


def fmt(value, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def run_label(run: dict) -> str:
    return f"{run['arm']} r{run['repeat']}"


def status(run: dict) -> str:
    if run["exit_code"] is None:
        return "running or killed"
    if run["exit_code"] == 0:
        return "ok"
    return f"crashed, exit {run['exit_code']}"


def render(runs: list[dict], cells: dict[str, list[dict]], root: Path) -> str:
    out = [
        "# n-gram module A/B: parity and step time",
        "",
        f"Runs under `{root}`, collected {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}. Per-step values come from "
        "each run's `monitors/file/metrics.jsonl` (rank 0). Recompiles are `Recompiling function` records in every rank's "
        "`TORCH_LOGS=recompiles` output, placed in the step whose end time follows them.",
        "",
        "## Runs",
        "",
        "Peak memory: max over steps of `perf/peak_memory` (rank 0, reserved, GiB, lower is better). Recompiles: total over "
        "ranks, fewer is better. Last: the latest step with a recompile on any rank.",
        "",
    ]
    rows = [
        [
            r["cell"],
            r["arm"],
            r["repeat"],
            r["head"][:9],
            len(r["steps"]),
            fmt(r["peak_mem_gib"], ".1f"),
            sum(r["recompiles_per_rank"].values()),
            fmt(r["last_recompile_step"], "d"),
            status(r),
        ]
        for r in runs
    ]
    out.append(
        table(
            ["cell", "arm", "rep", "head", "steps", "peak GiB", "recompiles", "last", "status"],
            rows,
            right={2, 4, 5, 6, 7},
        )
    )
    crashed = [r for r in runs if r["exit_code"] not in (0, None)]
    if crashed:
        out += ["", "Crashes, with the last exception line on any rank:", ""]
        out += [f"- {r['cell']} {run_label(r)}: `{r['error'] or '?'}`" for r in crashed]
    out += [
        "",
        "## Step time",
        "",
        "Seconds per step (`time/step`), lower is better. All: every logged step, including the compile step 1. "
        "Post-compile: steps from the first step after both step 1 and the last recompile on any rank.",
        "",
    ]
    rows = []
    for r in runs:
        a, p = r["step_time_all"] or {}, r["step_time_post_compile"] or {}
        rows.append(
            [
                r["cell"],
                run_label(r),
                fmt(a.get("median"), ".3f"),
                fmt(a.get("min"), ".3f"),
                fmt(a.get("max"), ".3f"),
                r["post_compile_from_step"],
                p.get("n", 0),
                fmt(p.get("median"), ".3f"),
                fmt(p.get("min"), ".3f"),
                fmt(p.get("max"), ".3f"),
            ]
        )
    out.append(
        table(
            ["cell", "run", "all med", "all min", "all max", "post from", "post n", "post med", "post min", "post max"],
            rows,
            right=set(range(2, 10)),
        )
    )
    out += [
        "",
        "## Startup",
        "",
        "Seconds, lower is better, from loguru timestamps (1 s resolution). Launcher: `run_ab.py` start to the launcher's "
        "first log line (launcher imports). Trainer import: to the trainer's first log line (pre-download, torchrun, rank "
        "imports). Setup: to the training loop (model build, weight load and conversion, FSDP). Step 1: compile and the "
        "first step. Wall: the whole run including shutdown.",
        "",
    ]
    rows = []
    for r in runs:
        s = r["startup"]
        rows.append(
            [
                r["cell"],
                run_label(r),
                "yes" if r["venv"] else "no",
                *(fmt(s[k], ".0f") for k in ("launcher_s", "trainer_import_s", "setup_s", "first_step_s", "wall_s")),
            ]
        )
    out.append(
        table(
            ["cell", "run", "local venv", "launcher", "trainer import", "setup", "step 1", "wall"],
            rows,
            right=set(range(3, 8)),
        )
    )
    out += [
        "",
        "## Parity",
        "",
        "Loss gap: max over steps and run pairs of abs(loss x - loss y), in nats. Grad gap: max over steps and run pairs of "
        "abs(grad norm x - grad norm y) / grad norm y, y the second-named arm. Both: lower is closer. Halves split the "
        "common steps in two. A real bug shows as an arm gap above the floor that grows from the first half to the second. "
        "Comparisons without a pair of finished runs are left out.",
        "",
    ]
    rows = []
    for cell, comparisons in cells.items():
        for c in comparisons:
            if not c.get("pairs"):
                continue
            rows.append(
                [
                    cell,
                    c["comparison"],
                    c.get("pairs", 0),
                    c.get("steps", 0),
                    *(fmt(c.get(f"loss_{w}"), ".1e") for w in ("all", "first", "second")),
                    *(fmt(c.get(f"grad_{w}"), ".1e") for w in ("all", "first", "second")),
                ]
            )
    out.append(
        table(
            [
                "cell",
                "comparison",
                "pairs",
                "steps",
                "loss all",
                "loss 1st",
                "loss 2nd",
                "grad all",
                "grad 1st",
                "grad 2nd",
            ],
            rows,
            right=set(range(2, 10)),
        )
    )
    out += ["", "## Per-step loss and grad norm", ""]
    by_cell: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_cell[r["cell"]].append(r)
    for cell, cell_runs in by_cell.items():
        steps = sorted(set().union(*(r["steps"] for r in cell_runs)))
        if not steps:
            continue
        for metric, title, spec in (("loss", "loss (nats)", ".6f"), ("grad_norm", "grad norm", ".6f")):
            out += [f"### {cell}: {title}", ""]
            rows = [[step, *(fmt(r["steps"].get(step, {}).get(metric), spec) for r in cell_runs)] for step in steps]
            out += [table(["step", *(run_label(r) for r in cell_runs)], rows), ""]
    return "\n".join(out).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=DEFAULT_ROOT, help="output root of run_ab.py")
    parser.add_argument("--out-dir", type=Path, default=HARNESS_DIR / "results", help="where parity.md/json go")
    args = parser.parse_args()

    run_dirs = sorted(path.parent for path in args.root.glob("*/*/r*/run.json"))
    runs = sorted(
        (collect_run(d) for d in run_dirs), key=lambda r: (r["cell"], r["arm"] != BASELINE_ARM, r["arm"], r["repeat"])
    )
    by_cell: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        by_cell[run["cell"]].append(run)
    cells = {cell: cell_parity(cell_runs) for cell, cell_runs in by_cell.items()}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "parity.md").write_text(render(runs, cells, args.root.resolve()))
    (args.out_dir / "parity.json").write_text(
        json.dumps({"root": str(args.root.resolve()), "runs": runs, "parity": cells}, indent=1) + "\n"
    )
    print(f"{len(runs)} runs in {len(cells)} cells -> {args.out_dir / 'parity.md'}")


if __name__ == "__main__":
    main()
