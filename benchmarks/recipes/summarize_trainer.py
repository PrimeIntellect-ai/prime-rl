"""Summarize a trainer_fit.sh sweep as a markdown table.

python benchmarks/recipes/summarize_trainer.py <out>
"""

import json
import re
import sys
from pathlib import Path
from statistics import mean

GPU_MEM_GIB = 139.8  # H200 total memory as torch reports it

OOM = re.compile(r"OutOfMemoryError|CUDA out of memory|cudaErrorMemoryAllocation|failed with OOM")


def status(run: Path, steps: list[dict]) -> str:
    exit_file = run / "exit"
    rc = int(exit_file.read_text().split()[0]) if exit_file.exists() else None
    log = (run / "stdout.log").read_text(errors="replace") if (run / "stdout.log").exists() else ""
    if OOM.search(log):
        return "OOM"
    if rc == 124:
        return "timeout"
    if rc not in (0, None):
        return f"fail rc={rc}"
    return "ok" if steps else "no steps"


def main(out: Path) -> None:
    rows = []
    for run in sorted(p for p in out.iterdir() if p.is_dir()):
        metrics = run / "monitors" / "file" / "metrics.jsonl"
        by_step: dict[int, dict] = {}
        for line in metrics.read_text().splitlines() if metrics.exists() else []:
            record = json.loads(line)
            by_step.setdefault(record.get("step", 0), {}).update(record)
        steps = list(by_step.values())
        # Step 1 includes compile and allocator warmup.
        steady = [s for s in steps if s.get("step", 0) > 1 and "time/step" in s]

        def avg(key: str) -> float | None:
            values = [s[key] for s in steady if key in s]
            return mean(values) if values else None

        peak = max((s["perf/peak_memory"] for s in steps if "perf/peak_memory" in s), default=None)
        wall = (run / "exit").read_text().split()[1] if (run / "exit").exists() else "-"
        rows.append(
            (
                run.name,
                status(run, steps),
                len(steady),
                avg("time/step"),
                avg("perf/throughput"),
                avg("perf/mfu"),
                peak,
                wall,
            )
        )

    print("| variant | status | steps | step s | tok/s | MFU % | peak GiB | headroom GiB | wall s |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, st, n, step, tps, mfu, peak, wall in rows:
        fmt = lambda v, f: f.format(v) if v is not None else "-"  # noqa: E731
        head = GPU_MEM_GIB - peak if peak is not None else None
        print(
            f"| {name} | {st} | {n} | {fmt(step, '{:.1f}')} | {fmt(tps, '{:,.0f}')} | {fmt(mfu, '{:.1f}')} "
            f"| {fmt(peak, '{:.1f}')} | {fmt(head, '{:.1f}')} | {wall} |"
        )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
