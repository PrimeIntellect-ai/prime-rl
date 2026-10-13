"""Compare the DSv4 dsa_backend A/B runs that `launch.sh` wrote: speed, parity, and compile counts per rank and step.

usage (any compute node; reads metrics and logs only):
  uv run --no-sync python benchmarks/scripts/dsv4_cudnn_flashmla_pr/compare.py LABEL=RUN_DIR [...] \
    [--pairs A:B,B:B2,...] [--warmup N]

Speed: median, min and max of `time/step`, `perf/throughput_per_gpu` and `perf/peak_memory` over the steps after
the first `--warmup`. Parity: per-step absolute differences of `loss/mean` and `optim/grad_norm` per pair.

Compiles are read from every rank's torchrun stdout and stderr. stdout and stderr are merged by timestamp,
and each compile line is attributed to the next `Step N` line of the same rank:
- cuDNN: `Compiled the cuDNN DSA backward ... in X.XXs`, with its seconds;
- TileLang: `begins to compile` / `completes to compile` pairs, with seconds from their 1 s timestamps;
- dynamo: `Recompiling function` lines (TORCH_LOGS=recompiles), counted only, since they carry no duration.
The early-step table reports each step's `time/step` next to it minus the max over ranks of that step's cuDNN
plus TileLang compile seconds.
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

SPEED = [
    ("time/step", "step s", "lower"),
    ("perf/throughput_per_gpu", "tok/s/GPU", "higher"),
    ("perf/peak_memory", "peak GiB", "lower"),
]
PARITY = ["loss/mean", "optim/grad_norm"]
ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(r"\bStep (\d+) \|")
CUDNN = re.compile(r"Compiled the cuDNN DSA backward (\w+) kernel.* in ([0-9.]+)s")
TILELANG_BEGIN = re.compile(r"^(\S+ \S+)\s+\[TileLang.*begins to compile kernel `(\w+)`")
TILELANG_END = re.compile(r"^(\S+ \S+)\s+\[TileLang.*completes to compile kernel `(\w+)`")
DYNAMO = re.compile(r"Recompiling function")
TIME_OF_DAY = re.compile(r"\b(\d{2}):(\d{2}):(\d{2})\b")


def load_metrics(run_dir: Path) -> dict[int, dict[str, float]]:
    by_step = defaultdict(dict)
    for line in (run_dir / "monitors/file/metrics.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record.get("step") is not None:
            by_step[record["step"]].update(record)
    return dict(by_step)


def rank_dirs(run_dir: Path) -> dict[int, Path]:
    attempts = sorted((run_dir / "logs/trainer/torchrun").glob("*/attempt_0"))
    assert attempts, f"no torchrun logs under {run_dir}"
    return {int(path.name): path for path in attempts[-1].iterdir() if path.name.isdigit()}


def timed_lines(path: Path) -> list[tuple[int, int, str]]:
    """`(seconds of day, 0 for a compile line or 1 otherwise, line)`; an untimed line takes the last time seen."""
    lines = []
    seconds = 0
    for raw in path.read_text(errors="replace").splitlines():
        line = ANSI.sub("", raw)
        if match := TIME_OF_DAY.search(line):
            hours, minutes, secs = (int(part) for part in match.groups())
            seconds = hours * 3600 + minutes * 60 + secs
        lines.append((seconds, 1 if STEP.search(line) else 0, line))
    return lines


def rank_compiles(rank_dir: Path) -> dict[int, dict[str, float]]:
    """Per step: cuDNN and TileLang compile counts and seconds, and dynamo recompile counts.

    stdout and stderr are merged by their 1 s timestamps, a compile line sorting before a step line of the same
    second. Compiles after the last step line are filed under step -1.
    """
    merged = []
    for name in ("stdout.log", "stderr.log"):
        if (rank_dir / name).exists():
            merged.extend(timed_lines(rank_dir / name))
    merged.sort(key=lambda timed: timed[:2])

    per_step = defaultdict(lambda: defaultdict(float))
    pending = defaultdict(float)
    tilelang_begin = {}
    for _, _, line in merged:
        if match := STEP.search(line):
            step = int(match.group(1))
            for key, value in pending.items():
                per_step[step][key] += value
            pending.clear()
        elif match := CUDNN.search(line):
            pending["cudnn"] += 1
            pending["cudnn_s"] += float(match.group(2))
        elif match := TILELANG_BEGIN.search(line):
            tilelang_begin[match.group(2)] = datetime.fromisoformat(match.group(1))
        elif match := TILELANG_END.search(line):
            pending["tilelang"] += 1
            begin = tilelang_begin.pop(match.group(2), None)
            if begin is not None:
                pending["tilelang_s"] += (datetime.fromisoformat(match.group(1)) - begin).total_seconds()
        elif DYNAMO.search(line):
            pending["dynamo"] += 1
    if pending:
        per_step[-1] = pending
    return per_step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="LABEL=RUN_DIR")
    parser.add_argument("--pairs", default="", help="comma-separated LEFT:RIGHT labels for the parity table")
    parser.add_argument("--warmup", type=int, default=5, help="leading steps excluded from the speed medians")
    parser.add_argument("--early-steps", type=int, default=20)
    args = parser.parse_args()

    runs = {}
    run_dirs = {}
    for spec in args.runs:
        label, run_dir = spec.split("=", 1)
        run_dirs[label] = Path(run_dir)
        runs[label] = load_metrics(Path(run_dir))

    print(f"Steady state = steps after the first {args.warmup}; median (min-max).\n")
    print("| arm | steps | " + " | ".join(f"{label} ({better} is better)" for _, label, better in SPEED) + " |")
    print("|---|---|" + "---|" * len(SPEED))
    for label, metrics in runs.items():
        steady = [step for step in sorted(metrics) if step > args.warmup]
        cells = []
        for key, _, _ in SPEED:
            values = [metrics[step][key] for step in steady if key in metrics[step]]
            cells.append(f"{statistics.median(values):.4g} ({min(values):.4g}-{max(values):.4g})" if values else "-")
        print(f"| {label} | {len(steady)} | " + " | ".join(cells) + " |")

    pairs = [pair.split(":") for pair in args.pairs.split(",") if pair]
    if pairs:
        print("\nPer-step parity: max |difference| over common steps, and the step it occurs at.\n")
        print("| pair | steps | " + " | ".join(f"max abs d {key}" for key in PARITY) + " |")
        print("|---|---|" + "---|" * len(PARITY))
    for left, right in pairs:
        steps = sorted(set(runs[left]) & set(runs[right]))
        cells = []
        for key in PARITY:
            diffs = [
                (abs(runs[left][step][key] - runs[right][step][key]), step)
                for step in steps
                if key in runs[left][step] and key in runs[right][step]
            ]
            worst, at = max(diffs) if diffs else (float("nan"), "-")
            cells.append(f"{worst:.3g} (step {at})")
        print(f"| {left} vs {right} | {len(steps)} | " + " | ".join(cells) + " |")

    for label, run_dir in run_dirs.items():
        compiles = {rank: rank_compiles(path) for rank, path in sorted(rank_dirs(run_dir).items())}
        print(f"\n### {label}: compiles per rank (cuDNN / TileLang / dynamo recompiles), and the steps they land in\n")
        print("| rank | cuDNN | TileLang | dynamo | steps with a compile |")
        print("|---|---|---|---|---|")
        for rank, per_step in compiles.items():
            totals = {key: int(sum(step[key] for step in per_step.values())) for key in ("cudnn", "tilelang", "dynamo")}
            steps = ",".join(str(step) for step in sorted(per_step) if any(per_step[step].values()))
            print(f"| {rank} | {totals['cudnn']} | {totals['tilelang']} | {totals['dynamo']} | {steps} |")

        metrics = runs[label]
        steady = [metrics[step]["time/step"] for step in sorted(metrics) if step > args.warmup]
        steady_median = statistics.median(steady) if steady else float("nan")
        print(f"\n{label}: per-step time, steady median {steady_median:.2f} s; flagged steps are >20% over it.\n")
        print("| step | time/step s | max-rank compile s | time minus compile s | cumulative s | flag |")
        print("|---|---|---|---|---|---|")
        cumulative = 0.0
        for step in sorted(metrics)[: args.early_steps]:
            step_time = metrics[step]["time/step"]
            cumulative += step_time
            compile_s = max(
                (per_step.get(step, {}).get("cudnn_s", 0.0) + per_step.get(step, {}).get("tilelang_s", 0.0))
                for per_step in compiles.values()
            )
            flag = "slow" if step_time > 1.2 * steady_median else ""
            print(
                f"| {step} | {step_time:.2f} | {compile_s:.1f} | {step_time - compile_s:.2f} | {cumulative:.1f} | {flag} |"
            )


if __name__ == "__main__":
    main()
