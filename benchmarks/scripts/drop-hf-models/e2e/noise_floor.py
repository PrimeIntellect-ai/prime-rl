"""Per metric, compare each non-main run's gap to its nearest main run against the largest gap among the main runs."""

import argparse
import itertools
from pathlib import Path

from compare import diff_series, load_metrics


def max_gap(left: dict, right: dict, key: str) -> float | None:
    diff = diff_series(left, right, key)
    return diff["max_abs"] if diff else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="E2E_OUT, holding one directory per arm or repeat")
    parser.add_argument("cell")
    parser.add_argument("keys", nargs="+", help="metric keys such as trainer:loss/mean")
    args = parser.parse_args()

    runs = {
        path.name: load_metrics(path / args.cell)
        for path in sorted(args.root.iterdir())
        if (path / args.cell / "monitors").is_dir()
    }
    mains = [name for name in runs if name.startswith("main")]
    others = [name for name in runs if not name.startswith("main")]
    print(f"{args.cell}: main runs {mains}, others {others}")
    for key in args.keys:
        floor = max(gap for a, b in itertools.combinations(mains, 2) if (gap := max_gap(runs[a], runs[b], key)) is not None)
        nearest = {arm: min(max_gap(runs[main], runs[arm], key) for main in mains) for arm in others}
        pairs = {f"{a}-{b}": max_gap(runs[a], runs[b], key) for a, b in itertools.combinations(others, 2)}
        print(f"  {key:<45} main-main max {floor:.2e}")
        print("    nearest main: " + ", ".join(f"{arm} {gap:.2e} ({gap / floor:.2f}x)" if floor else f"{arm} {gap:.2e} (main is bitwise)" for arm, gap in nearest.items()))
        print("    other pairs: " + ", ".join(f"{pair} {gap:.2e}" for pair, gap in pairs.items()))


if __name__ == "__main__":
    main()
