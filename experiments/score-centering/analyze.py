"""Export matched-step metrics and curves from a pair of experiment directories."""

import argparse
import csv
import json
import math
import re
from bisect import bisect_right
from itertools import accumulate
from pathlib import Path


def metric_records(run):
    for path in sorted((run / "monitors/file").glob("metrics*.jsonl")):
        with path.open() as stream:
            for line in stream:
                if line.endswith("\n"):
                    yield json.loads(line)


def read_metrics(run, through_step=None):
    rows = {}
    generation = []
    cutoff = math.inf
    if through_step is not None:
        boundary = [
            row for row in metric_records(run) if row.get("step") == through_step and row.get("producer") == "trainer"
        ]
        if not any("optim/grad_norm" in row for row in boundary):
            raise ValueError(f"{run} has no completed optimizer step {through_step}")
        cutoff = max(row["time"] for row in boundary)
    for row in metric_records(run):
        if row["time"] > cutoff:
            continue
        generated = [v for k, v in row.items() if k.endswith("/generation_tokens_total")]
        if generated:
            generation.append((row["time"], sum(generated)))
        if "optim/grad_norm" in row:
            row["trainer/update_time"] = row["time"]
        producer = "trainer" if "time/forward_backward" in row else "orchestrator"
        if "time/forward_backward" in row or "progress/output_tokens" in row:
            row.update({f"{producer}/{k}": v for k, v in row.items() if k.startswith("time/")})
        step = row.get("step")
        if step is None or (through_step is not None and step > through_step):
            continue
        rows.setdefault(step, {}).update(row)
    attach_budgets(run, rows, generation)
    return rows


def attach_budgets(run, rows, generation):
    path = run / "monitors/file/traces/stream.index.jsonl"
    if not path.exists():
        return
    with path.open() as stream:
        episodes = [json.loads(line) for line in stream if line.endswith("\n")]
    episodes = [ep for ep in episodes if ep.get("kind") == "train" and ep.get("arrival") is not None]
    if not episodes:
        return
    arrivals, output_tokens = zip(*sorted((ep["arrival"], ep.get("output_tokens", 0)) for ep in episodes))
    cumulative = [0, *accumulate(output_tokens)]
    start = min(ep["dispatch"] for ep in episodes if ep.get("dispatch") is not None)
    generation.sort()
    generation_times = [time for time, _ in generation]
    for row in rows.values():
        if (time := row.get("trainer/update_time")) is None:
            continue
        row["budget/elapsed_seconds"] = time - start
        row["budget/received_train_output_tokens"] = cumulative[bisect_right(arrivals, time)]
        index = bisect_right(generation_times, time) - 1
        if index >= 0:
            row["budget/server_output_tokens_including_eval"] = generation[index][1]


def read_lineage(segments):
    combined = {}
    last_step = -1
    offsets = {}
    for index, segment in enumerate(segments):
        through_step = segment.get("through_step")
        if index < len(segments) - 1 and through_step is None:
            raise ValueError("Every completed lineage segment needs through_step")
        rows = read_metrics(Path(segment["run"]), through_step)
        for step, row in rows.items():
            if step <= last_step:
                continue
            for key in row.keys() & offsets.keys():
                row[key] += offsets[key]
            combined[step] = row
        if through_step is not None:
            if through_step <= last_step:
                raise ValueError("Lineage boundaries must increase")
            last_step = through_step
            offsets = {key: value for key, value in combined[last_step].items() if key.startswith("budget/")}
    return combined


def numerical_warnings(run):
    pattern = re.compile(r"\b(?:non-finite|nonfinite|nan_count|nan)\b|gradient.*\b(?:inf|nan)\b", re.IGNORECASE)
    warnings = []
    for path in sorted((run / "logs").glob("attempt_*/**/*.log")):
        with path.open(errors="replace") as stream:
            for number, line in enumerate(stream, 1):
                if pattern.search(line):
                    warnings.append({"path": str(path), "line": number, "message": line.strip()})
    return warnings


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("runs", type=Path, nargs="*")
    parser.add_argument("--lineage", type=Path, help="Named arms with checkpoint-linked run segments")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-step", type=int, help="Limit both arms to this optimizer step")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.lineage is not None:
        if args.runs:
            parser.error("Use run directories or --lineage, not both")
        lineage = json.loads(args.lineage.read_text())
        runs = {name: read_lineage(segments) for name, segments in lineage.items()}
        sources = {name: [Path(segment["run"]) for segment in segments] for name, segments in lineage.items()}
    else:
        if not args.runs:
            parser.error("Provide run directories or --lineage")
        runs = {run.name: read_metrics(run) for run in args.runs}
        sources = {run.name: [run] for run in args.runs}
    if args.max_step is not None:
        runs = {name: {step: row for step, row in rows.items() if step <= args.max_step} for name, rows in runs.items()}
    pattern = re.compile(
        r"^budget/|^trainer/time/|^orchestrator/time/|reward/mean$|avg@1$|entropy/all/mean$|mismatch_kl/all/(?:mean|std|max)$|"
        r"optim/grad_norm$|is_masked/mean$|score_centering/.*/mean$|"
        r"num_output_tokens/mean$|truncat.*mean$|off_policy.*mean$|has_error/mean$|loss/(?:.*/)?mean$"
    )
    keys = sorted({key for rows in runs.values() for row in rows.values() for key in row if pattern.search(key)})
    summary = {}
    for name, rows in runs.items():
        ordered = sorted(rows)
        latest = {key: next((rows[s][key] for s in reversed(ordered) if key in rows[s]), None) for key in keys}
        nonfinite = [
            (s, k) for s, row in rows.items() for k, v in row.items() if isinstance(v, float) and not math.isfinite(v)
        ]
        summary[name] = {
            "last_step": max(rows, default=None),
            "latest": latest,
            "nonfinite": nonfinite,
            "numerical_warnings": [warning for run in sources[name] for warning in numerical_warnings(run)],
            "sources": [str(run) for run in sources[name]],
        }
    with (args.output / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run", "step", *keys])
        writer.writeheader()
        for name, rows in runs.items():
            for step, row in sorted(rows.items()):
                writer.writerow({"run": name, "step": step, **{k: row[k] for k in keys if k in row}})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not keys:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(keys), 1, figsize=(11, 3 * len(keys)), squeeze=False)
    for key, ax in zip(keys, axes[:, 0]):
        for name, rows in runs.items():
            points = [(s, row[key]) for s, row in sorted(rows.items()) if isinstance(row.get(key), (int, float))]
            if points:
                x, y = zip(*points)
                ax.plot(x, y, label=name, linewidth=1)
        ax.set_title(key)
        ax.set_xlabel("Optimizer step")
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.output / "curves.png", dpi=150)
    fig.savefig(args.output / "curves.pdf")
    plt.close(fig)

    overview = {
        "train/agg/all/agent/reward/mean": "Training success (all arrivals)",
        "eval/terminal-bench-2/all/agent/avg@1": "Terminal Bench 2 success (inspect failures separately)",
        "optim/grad_norm": "Gradient norm",
        "mismatch_kl/all/mean": "Trainer–sampler mismatch KL (mean)",
        "mismatch_kl/all/max": "Trainer–sampler mismatch KL (maximum token)",
        "entropy/all/mean": "Token entropy",
    }
    fig, axes = plt.subplots(len(overview), 2, figsize=(14, 3 * len(overview)), squeeze=False)
    for (key, title), pair in zip(overview.items(), axes):
        for column, ax in enumerate(pair):
            for name, rows in runs.items():
                points = []
                for step, row in sorted(rows.items()):
                    x = step if column == 0 else row.get("budget/received_train_output_tokens")
                    y = row.get(key)
                    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                        points.append((x if column == 0 else x / 1e6, y))
                if points:
                    x, y = zip(*points)
                    ax.plot(x, y, label=name, linewidth=1.5)
            ax.set_title(title)
            ax.set_xlabel("Optimizer step" if column == 0 else "Received training output tokens (millions)")
            if ax.lines:
                ax.legend()
            ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.output / "overview.png", dpi=150)
    fig.savefig(args.output / "overview.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
