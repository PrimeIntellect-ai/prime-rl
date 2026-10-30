"""Summarize the tensors saved for backward per compiled graph, as recorded by saved_activations/sitecustomize.py.

Usage: saved_activations.py [--rank R] [--base NAME] NAME=DIR [NAME=DIR ...]

Each DIR holds rank<R>.jsonl. Graphs are numbered in compile order, which is layer order for the per-layer
compiled DSv4 proxy. Module paths are made relative to the checkpointed block. Prints, per arm, each graph's
saved activation bytes (op outputs, excluding graph inputs such as parameters and the layer input), then the saved
op outputs grouped by (module path, op) with their size per graph. With --base, also prints per graph the saved op
outputs that differ from the base arm's, so a selective AC arm shows exactly what it keeps beyond full AC.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

GIB = 1024**3
MODULE_SUBSCRIPT = re.compile(r"(?:\.__dict__\['_modules'\])?\['([^']+)'\]")
BLOCK_PREFIX = "_checkpoint_wrapped_module."


def block_path(source_name: str) -> str:
    path = MODULE_SUBSCRIPT.sub(r".\1", source_name)
    _, found, relative = path.rpartition(BLOCK_PREFIX)
    return relative if found else "(block)"


def load(run_dir: Path, rank: int) -> list[list[dict]]:
    graphs: dict[str, list[dict]] = {}
    for line in (run_dir / f"rank{rank}.jsonl").read_text().splitlines():
        record = json.loads(line)
        record["path"] = block_path(record["module"])
        graphs.setdefault(record["compile_id"], []).append(record)
    return list(graphs.values())


def outputs(graph: list[dict]) -> list[dict]:
    return [record for record in graph if record["op"] == "call_function"]


def key(record: dict) -> tuple:
    return (record["path"], record["target"], tuple(record["shape"] or ()), record["dtype"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="NAME=DIR")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--base", help="arm NAME to diff the other arms against")
    args = parser.parse_args()

    arms = {}
    for spec in args.runs:
        name, _, run_dir = spec.partition("=")
        arms[name] = load(Path(run_dir), args.rank)

    for name, graphs in arms.items():
        print(f"## {name} (rank {args.rank})\n")
        print("| graph | saved op outputs (GiB) | saved graph inputs (GiB) | saved op outputs (count) |")
        print("|---:|---:|---:|---:|")
        for index, graph in enumerate(graphs):
            out_bytes = sum(record["bytes"] for record in outputs(graph))
            in_bytes = sum(record["bytes"] for record in graph if record["op"] == "placeholder")
            print(f"| {index} | {out_bytes / GIB:.3f} | {in_bytes / GIB:.3f} | {len(outputs(graph))} |")
        print()
        grouped: dict[tuple, list[int]] = defaultdict(lambda: [0] * len(graphs))
        for index, graph in enumerate(graphs):
            for record in outputs(graph):
                grouped[(record["path"], record["target"])][index] += record["bytes"]
        print("| module path | op | MiB per graph |")
        print("|---|---|---|")
        for (path, target), sizes in sorted(grouped.items(), key=lambda item: -sum(item[1])):
            print(f"| {path} | {target} | {', '.join(f'{size / 2**20:.0f}' for size in sizes)} |")
        print()

    if args.base is None:
        return
    base = arms[args.base]
    for name, graphs in arms.items():
        if name == args.base:
            continue
        print(f"## {name} minus {args.base}: saved op outputs that differ (negative: saved only by {args.base})\n")
        print("| graph | module path | op | shape | dtype | MiB |")
        print("|---:|---|---|---|---|---:|")
        for index, graph in enumerate(graphs):
            extra = Counter(key(record) for record in outputs(graph))
            extra.subtract(Counter(key(record) for record in outputs(base[index])))
            sizes = {key(record): record["bytes"] for record in outputs(graph) + outputs(base[index])}
            total = 0
            for record_key, count in sorted(extra.items()):
                if count == 0:
                    continue
                path, target, shape, dtype = record_key
                total += count * sizes[record_key]
                shape_text = "x".join(shape)
                print(f"| {index} | {path} | {target} | {shape_text} | {dtype} | {count * sizes[record_key] / 2**20:.0f} |")
            print(f"| {index} | **total** | | | | {total / 2**20:.0f} |")
        print()


if __name__ == "__main__":
    main()
