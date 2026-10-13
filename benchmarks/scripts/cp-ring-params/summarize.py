"""Summarize CP attention-params runs: recompiles, CP guard failures, step times, loss and grad norm.

usage: uv run python summarize.py <runs-dir> [--steps-from N] [--json out.json]
"""

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

CP_FIELDS = re.compile(r"max_seqlen|cu_seqlens|local_k|_PARAMS")
RECOMPILE = re.compile(r"^\[(?P<thread>[^\]]+)\].*Recompiling function (?P<frame>\S+) in (?P<loc>\S+)")
GUARD = re.compile(r"^\s+- \d+/\d+: (?P<guard>.*)")


def rank0_logs(run: Path) -> list[Path]:
    rl_logs = sorted(run.glob("run/**/trainer/torchrun/*/attempt_0/0/stderr.log"))
    return rl_logs or sorted(run.glob("trainer.log")) or sorted(run.glob("sft.log"))


def scan_log(path: Path) -> dict:
    recompiles = Counter()
    frames = Counter()
    cp_guards = []
    limit_hits = 0
    graph_breaks = 0
    in_guards = False
    for line in path.read_text(errors="replace").splitlines():
        match = RECOMPILE.match(line)
        if match:
            recompiles[match["thread"]] += 1
            frames[f"{match['thread']} {match['frame']} {match['loc'].rsplit('/', 1)[-1]}"] += 1
            in_guards = True
            continue
        if in_guards:
            guard = GUARD.match(line)
            if guard:
                if CP_FIELDS.search(guard["guard"]):
                    cp_guards.append(guard["guard"].split("#")[0].strip())
                continue
            if not line.startswith(" "):
                in_guards = False
        if "recompile_limit" in line and "hit" in line:
            limit_hits += 1
        if "Graph break" in line:
            graph_breaks += 1
    return {
        "recompiles_by_thread": dict(recompiles),
        "max_recompiles_per_frame": max(frames.values(), default=0),
        "frames": dict(frames),
        "cp_guard_failures": len(cp_guards),
        "cp_guard_examples": sorted(set(cp_guards))[:5],
        "recompile_limit_hits": limit_hits,
        "graph_breaks": graph_breaks,
    }


def scan_metrics(run: Path, steps_from: int) -> dict:
    files = sorted(run.glob("run/**/monitors/file/metrics.jsonl"))
    if not files:
        return {}
    rows = [json.loads(line) for line in files[0].read_text().splitlines() if line.strip()]
    # Rows are split by producer. The orchestrator also logs time/step, so take it only from trainer rows,
    # which are the ones carrying time/forward_backward.
    by_step: dict[int, dict] = {}
    for row in rows:
        if row.get("step") is None:
            continue
        merged = by_step.setdefault(row["step"], {"step": row["step"]})
        trainer_row = "time/forward_backward" in row
        for key in ("loss/mean", "optim/grad_norm", "perf/peak_memory", "time/forward_backward"):
            if key in row:
                merged[key] = row[key]
        if trainer_row and "time/step" in row:
            merged["time/step"] = row["time/step"]
    steps = sorted((r for r in by_step.values() if "loss/mean" in r), key=lambda r: r["step"])
    steady = [r for r in steps if r["step"] >= steps_from]

    def stats(key: str) -> dict | None:
        values = [r[key] for r in steady if r.get(key) is not None]
        if not values:
            return None
        return {"median": statistics.median(values), "min": min(values), "max": max(values)}

    return {
        "num_steps": len(steps),
        "time_step_s": stats("time/step"),
        "time_fwd_bwd_s": stats("time/forward_backward"),
        "peak_memory": stats("perf/peak_memory"),
        "per_step": [
            {k: r.get(k) for k in ("step", "time/step", "time/forward_backward", "loss/mean", "optim/grad_norm")}
            for r in steps
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("--steps-from", type=int, default=5)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    summary = {}
    for run in sorted(p for p in args.runs_dir.iterdir() if p.is_dir()):
        logs = rank0_logs(run)
        launcher_logs = [run / "rl.log", run / "trainer.log", run / "sft.log"]
        tail = [
            line for log in launcher_logs if log.exists() for line in log.read_text().splitlines() if line.startswith("exit=")
        ]
        summary[run.name] = {
            "sha": (run / "sha").read_text().strip() if (run / "sha").exists() else None,
            "exit": tail[-1] if tail else "running or missing",
            "log": scan_log(logs[0]) if logs else None,
            "metrics": scan_metrics(run, args.steps_from),
        }

    if args.json:
        args.json.write_text(json.dumps(summary, indent=2))
    for name, s in summary.items():
        log = s["log"] or {}
        m = s["metrics"] or {}
        ts = m.get("time_step_s") or {}
        fb = m.get("time_fwd_bwd_s") or {}
        print(
            f"{name:34s} {s['exit']:22s} steps={m.get('num_steps', 0):3d} "
            f"recomp={log.get('recompiles_by_thread', {})} maxframe={log.get('max_recompiles_per_frame', 0)} "
            f"cpguards={log.get('cp_guard_failures', 0)} limit={log.get('recompile_limit_hits', 0)} "
            f"step_med={ts.get('median', float('nan')):.2f}s fwdbwd_med={fb.get('median', float('nan')):.2f}s"
        )


if __name__ == "__main__":
    main()
