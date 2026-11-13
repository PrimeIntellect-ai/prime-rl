"""Summarize the trainer A/B runs: step-time stats, throughput, MFU, memory, teardown timing and loss parity."""

import argparse
import json
import re
import statistics
from pathlib import Path

STEP_LINE = re.compile(r"Step (\d+) \|")


def read_metrics(run_dir: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for line in (run_dir / "monitors" / "file" / "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("producer") == "trainer" and "step" in row:
            rows.setdefault(row["step"], {}).update(row)
    return {step: row for step, row in rows.items() if "time/step" in row}


def last_step_epoch(rank_log: Path, last_step: int) -> float | None:
    for line in rank_log.read_text(errors="replace").splitlines():
        match = STEP_LINE.search(line)
        if match and int(match.group(1)) == last_step:
            return float(line.split(" ", 1)[0])
    return None


def teardown(run_dir: Path, last_step: int) -> tuple[list[float], list[str], list[str]]:
    ranks_dir = run_dir / "bench" / "ranks"
    reference = last_step_epoch(ranks_dir / "rank_0.log", last_step)
    exit_seconds, problems = [], []
    for exit_file in sorted(ranks_dir.glob("exit_rank_*.txt"), key=lambda p: int(p.stem.split("_")[-1])):
        rank = int(exit_file.stem.split("_")[-1])
        fields = dict(re.findall(r"(\w+) (\S+)", exit_file.read_text()))
        if "exit" not in fields:
            problems.append(f"rank {rank}: no exit record")
            continue
        if fields["rc"] != "0":
            problems.append(f"rank {rank}: rc {fields['rc']}")
        rank_step = last_step_epoch(ranks_dir / f"rank_{rank}.log", last_step)
        anchor = rank_step if rank_step is not None else reference
        if anchor is not None:
            exit_seconds.append(float(fields["exit"]) - anchor)
    verdicts = []
    for node_info in sorted((run_dir / "bench").glob("node_*/node_info.txt")):
        match = re.search(r"verdict (\w+)", node_info.read_text())
        verdicts.append(match.group(1) if match else "UNKNOWN")
    return exit_seconds, problems, verdicts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_root", type=Path)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    metrics = {}
    print("| run | arm | commit | median step s (min, max) | tok/s/GPU | MFU % | peak mem GiB | exit s (min, max) | exit |")
    print("|---|---|---|---|---|---|---|---|---|")
    for run in args.runs:
        run_dir = args.runs_root / run
        info = (run_dir / "bench" / "run_info.txt").read_text()
        arm = re.search(r"arm (\S+)", info).group(1)
        commit = re.search(r"commit (\S+)", info).group(1)
        rows = read_metrics(run_dir)
        metrics[run] = rows
        last_step = max(rows)
        steady = [rows[s] for s in sorted(rows) if s > args.warmup]
        step_times = [r["time/step"] for r in steady]
        exit_seconds, problems, verdicts = teardown(run_dir, last_step)
        clean = "clean" if not problems and all(v == "CLEAN" for v in verdicts) else f"{verdicts} {problems}"
        exit_cell = f"{statistics.median(exit_seconds):.1f} ({min(exit_seconds):.1f}, {max(exit_seconds):.1f})" if exit_seconds else "n/a"
        print(
            f"| {run} | {arm} | {commit} | {statistics.median(step_times):.3f} ({min(step_times):.3f}, {max(step_times):.3f}) "
            f"| {statistics.median(r['perf/throughput_per_gpu'] for r in steady):.0f} "
            f"| {statistics.median(r['perf/mfu'] for r in steady):.2f} "
            f"| {max(r['perf/peak_memory'] for r in rows.values()):.1f} | {exit_cell} | {clean} |"
        )
    print(f"\nsteady state = steps > {args.warmup}; exit s = median (min, max) over ranks of rank exit epoch minus that rank's step-{last_step} log epoch\n")

    names = list(metrics)
    steps = sorted(set.intersection(*(set(m) for m in metrics.values())))
    print("| step | " + " | ".join(f"{n} loss | {n} grad norm | {n} step s" for n in names) + " |")
    print("|---|" + "---|" * (3 * len(names)))
    for step in steps:
        cells = []
        for n in names:
            row = metrics[n][step]
            cells += [f"{row['loss/mean']:.6f}", f"{row.get('optim/grad_norm', float('nan')):.6f}", f"{row['time/step']:.3f}"]
        print(f"| {step} | " + " | ".join(cells) + " |")
    for key in ("loss/mean", "optim/grad_norm"):
        worst = max(
            (abs(metrics[a][s][key] - metrics[b][s][key]), s, a, b)
            for s in steps
            for i, a in enumerate(names)
            for b in names[i + 1 :]
        )
        print(f"\nmax |diff| {key}: {worst[0]:.3g} at step {worst[1]} ({worst[2]} vs {worst[3]})")


if __name__ == "__main__":
    main()
