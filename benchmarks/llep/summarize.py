"""Markdown tables from bench_llep.py JSON lines: one table per (world, tokens per rank).

python benchmarks/llep/summarize.py results.jsonl --baseline mega --target llep_mega
"""

import argparse
import json
from collections import defaultdict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--baseline", default="mega")
    parser.add_argument("--target", default="llep_mega")
    parser.add_argument("--memory", action="store_true", help="also print peak allocator + static buffer GiB")
    args = parser.parse_args()
    rows = [json.loads(line) for line in open(args.path)]
    # Keep the latest measurement of each (world, tokens, scenario, method)
    latest = {}
    for row in rows:
        latest[(row["world"], row["tokens_per_rank"], row["scenario"], row["method"])] = row
    groups = defaultdict(dict)
    for (world, tokens, scenario, method), row in latest.items():
        groups[(world, tokens)].setdefault(scenario, {})[method] = row
    for (world, tokens), scenarios in sorted(groups.items()):
        methods = sorted({m for per in scenarios.values() for m in per}, key=lambda m: (m.startswith("llep"), m))
        print(f"\n### EP={world}, {tokens} tokens/rank (ms)\n")
        header = ["scenario", "max_vio", "raw", "rank imb.", *methods, f"{args.target} vs {args.baseline}"]
        print("| " + " | ".join(header) + " |")
        print("|" + "---|" * len(header))
        for scenario, per in scenarios.items():
            any_row = next(iter(per.values()))
            cells = [
                scenario,
                f"{any_row['max_vio']:.2f}",
                f"{any_row['max_vio_raw']:.2f}",
                f"{any_row['rank_imbalance']:.2f}",
            ]
            for method in methods:
                row = per.get(method)
                if row is None:
                    cells.append("")
                elif row.get("error"):
                    cells.append(row["error"])
                else:
                    cell = f"{row['ms']:.2f}"
                    if args.memory:
                        cell += f" ({row['peak_gib']:.1f}+{row.get('static_gib', 0):.1f})"
                    cells.append(cell)
            base, target = per.get(args.baseline), per.get(args.target)
            ratio = ""
            if base and target and not base.get("error") and not target.get("error"):
                ratio = f"{base['ms'] / target['ms']:.2f}x"
            cells.append(ratio)
            print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
