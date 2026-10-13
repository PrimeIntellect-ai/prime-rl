"""Compare per-step loss, grad norm and entropy between pairs of arms.

Without pairs it compares the fake-data RL trainer arms of `matrix.sh`; with pairs, the named runs.
usage: uv run python compare_numerics.py <runs-dir> [<run-a>:<run-b> ...]
"""

import json
import sys
from pathlib import Path

KEYS = ("loss/mean", "optim/grad_norm", "entropy/all/mean", "mismatch_kl/all/mean")


def load(run: Path) -> dict[int, dict]:
    paths = sorted(run.glob("run/**/monitors/file/metrics.jsonl"))
    if not paths:
        return {}
    path = paths[0]
    by_step: dict[int, dict] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("producer") != "trainer" or row.get("step") is None:
            continue
        merged = by_step.setdefault(row["step"], {})
        merged.update({k: row[k] for k in KEYS if k in row})
    return by_step


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def compare(runs: Path, a: str, b: str) -> None:
    left, right = load(runs / a), load(runs / b)
    steps = sorted(set(left) & set(right))
    if not steps:
        print(f"{a:24s} vs {b:24s}  missing")
        return
    first = steps[0]
    step1 = "  ".join(f"{k.split('/')[0]}={rel(left[first][k], right[first][k]):.1e}" for k in KEYS if k in left[first])
    worst = {k: max(rel(left[s][k], right[s][k]) for s in steps if k in left[s] and k in right[s]) for k in KEYS[:2]}
    print(
        f"{a:24s} vs {b:24s}  steps={len(steps):2d}  step{first} rel: {step1}  "
        f"max rel over steps: loss={worst['loss/mean']:.1e} grad_norm={worst['optim/grad_norm']:.1e}"
    )


def main() -> None:
    runs = Path(sys.argv[1])
    if len(sys.argv) > 2:
        for pair in sys.argv[2:]:
            compare(runs, *pair.split(":"))
        return
    print("--- branch vs main (same config)")
    for cfg in ("off", "fg"):
        compare(runs, f"trainer-br-{cfg}", f"trainer-main-{cfg}")
    print("--- same-code repeats (noise floor)")
    compare(runs, "trainer-main-off", "trainer-main-off-r2")
    compare(runs, "trainer-br-off", "trainer-br-off-r2")
    print("--- fullgraph off vs on, same code")
    compare(runs, "trainer-main-off", "trainer-main-fg")
    compare(runs, "trainer-br-off", "trainer-br-fg")


if __name__ == "__main__":
    main()
