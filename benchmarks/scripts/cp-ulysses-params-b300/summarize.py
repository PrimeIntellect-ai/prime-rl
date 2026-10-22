"""Summarize SFT A/B runs: steady-state step time, loss, graph breaks and recompiles per rank.

usage: uv run python summarize.py <runs-dir> <run> [<run> ...]
"""

import json
import statistics
import sys
from pathlib import Path


def load_metrics(run: Path) -> dict[int, dict]:
    by_step: dict[int, dict] = {}
    for line in (run / "monitors/file/metrics.jsonl").read_text().splitlines():
        record = json.loads(line)
        by_step.setdefault(record["step"], {}).update(record)
    return by_step


def rank_logs(run: Path) -> list[str]:
    return [log.read_text() for log in sorted(run.glob("logs/attempt_1/trainer/torchrun/*/attempt_0/*/stderr.log"))]


def main() -> None:
    runs_dir = Path(sys.argv[1])
    print("| run | step med (min-max) s | peak GiB | loss 1 | loss 20 | breaks | recompiles per rank | ULYSSES_PARAMS guard failures |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name in sys.argv[2:]:
        run = runs_dir / name
        by_step = load_metrics(run)
        times = [by_step[step]["time/step"] for step in sorted(by_step) if 5 <= step <= 20]
        peak = max(record["perf/peak_memory"] for record in by_step.values())
        logs = rank_logs(run)
        breaks = max(log.count("[__graph_breaks] Graph break") for log in logs)
        recompiles = sorted({log.count("[__recompiles] Recompiling") for log in logs})
        cp_guards = max(log.count("ULYSSES_PARAMS") for log in logs)
        print(
            f"| {name} | {statistics.median(times):.3f} ({min(times):.3f}-{max(times):.3f}) | {peak:.1f} | "
            f"{by_step[1]['loss/mean']:.4f} | {by_step[max(by_step)]['loss/mean']:.4f} | {breaks} | "
            f"{'-'.join(map(str, recompiles))} | {cp_guards} |"
        )


if __name__ == "__main__":
    main()
