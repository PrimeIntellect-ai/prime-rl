"""Count weight reloads and vLLM layerwise-reload warnings per RL run from its inference log, as a markdown table."""

import argparse
import re
from collections import Counter
from pathlib import Path

RELOAD_RE = re.compile(r"\((Worker_DP\d+) pid=\d+\).*Reloading checkpoint-format weights")
FAILED_RE = re.compile(r"\((Worker_DP\d+) pid=\d+\).*\] (\w+): Failed to load weights")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="LABEL=RUN_DIR pairs")
    args = parser.parse_args()

    reloads, warnings = {}, {}
    for label, path in (run.split("=", 1) for run in args.runs):
        log = (Path(path) / "logs" / "attempt_1" / "inference.log").read_text(errors="replace")
        per_worker = Counter(match.group(1) for match in RELOAD_RE.finditer(log))
        reloads[label] = per_worker
        failed = Counter(match.group(2) for match in FAILED_RE.finditer(log))
        warnings[label] = {name: count / sum(per_worker.values()) for name, count in failed.items()}

    labels = list(reloads)
    first = "Failed-load warnings per worker and reload"
    widths = [max(len(first), *(len(name) for counts in warnings.values() for name in counts))] + [
        max(len(label), 10) for label in labels
    ]

    def row(cells: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(width) for cell, width in zip(cells, widths)) + " |"

    lines = [row([first, *labels]), row(["-" * width for width in widths])]
    lines.append(
        row(
            ["Reloads per worker (workers)"]
            + [f"{min(r.values())}-{max(r.values())} ({len(r)})" for r in reloads.values()]
        )
    )
    for name in sorted({name for counts in warnings.values() for name in counts}):
        lines.append(row([name] + [f"{warnings[label].get(name, 0):g}" for label in labels]))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
