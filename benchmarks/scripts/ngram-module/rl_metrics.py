"""Tabulate per-step trainer metrics of several RL runs side by side as a markdown table."""

import argparse
import json
from pathlib import Path

METRICS = {
    "mismatch_kl/all/mean": "mismatch KL",
    "loss/mean": "loss",
}


def trainer_rows(run_dir: Path) -> dict[int, dict]:
    rows = {}
    for line in (run_dir / "monitors" / "file" / "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("producer") == "trainer" and row.get("step") is not None:
            rows[row["step"]] = rows.get(row["step"], {}) | row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="LABEL=RUN_DIR pairs")
    args = parser.parse_args()

    runs = {label: trainer_rows(Path(path)) for label, path in (run.split("=", 1) for run in args.runs)}
    header = ["step"] + [f"{label} {name}" for name in METRICS.values() for label in runs]
    widths = [max(len(column), 6) for column in header]
    lines = [
        "| " + " | ".join(column.ljust(width) for column, width in zip(header, widths)) + " |",
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    for step in sorted(set().union(*runs.values())):
        cells = [str(step)]
        for metric in METRICS:
            for rows in runs.values():
                value = rows.get(step, {}).get(metric)
                cells.append("" if value is None else f"{value:.4g}")
        lines.append("| " + " | ".join(cell.rjust(width) for cell, width in zip(cells, widths)) + " |")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
