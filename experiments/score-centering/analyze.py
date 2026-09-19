"""Export matched-step metrics and curves from a pair of experiment directories."""

import argparse
import csv
import json
import math
import re
from pathlib import Path


def read_metrics(run):
    rows = {}
    for path in sorted((run / "monitors/file").glob("metrics*.jsonl")):
        with path.open() as stream:
            for line in stream:
                if not line.endswith("\n"):
                    continue
                row = json.loads(line)
                step = row.get("step")
                if step is None:
                    continue
                rows.setdefault(step, {}).update(row)
    return rows


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
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = {run.name: read_metrics(run) for run in args.runs}
    pattern = re.compile(
        r"reward/mean$|avg@1$|entropy/all/mean$|mismatch_kl/all/mean$|"
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
            "numerical_warnings": numerical_warnings(next(run for run in args.runs if run.name == name)),
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


if __name__ == "__main__":
    main()
