#!/usr/bin/env python3
"""A tiny training dataset for the in-run eval smoke run (configs/advanced/nemotron-3-super/tb4-inrun-eval-smoke.toml),
cut from the v2 SFT build: whole rows (no truncation) short enough for the smoke's seq_len, same schema and subset.

    python scripts/tb4_inrun_eval/make_smoke_dataset.py --src <nemotron-distill-sft-tb4-1m-v2> \\
        --out <dir>/nemotron-distill-sft-tb4-smoke [--seq-len 65536 --train-rows 24 --val-rows 4]

Upload ``<out>`` to ``datasets/nemotron-distill-sft-tb4-smoke`` on the run's volume (after approval). The rows are
the build's own (private task content): the dataset goes to the private volume only, never into this repository.
Rows are picked deterministically (by sha256 of their id, one per task first), only rows with ``num_tokens <=
seq_len``, so packing never truncates a row and every packed row holds whole samples.
"""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def pick(table, rows: int, seq_len: int) -> list[int]:
    ids = table.column("id").to_pylist()
    tasks = table.column("task").to_pylist()
    tokens = table.column("num_tokens").to_pylist()
    fitting = sorted(
        (i for i in range(len(ids)) if tokens[i] <= seq_len), key=lambda i: hashlib.sha256(ids[i].encode()).hexdigest()
    )
    chosen, seen = [], set()
    for i in fitting:  # one row per task first, then fill up
        if tasks[i] not in seen:
            chosen.append(i)
            seen.add(tasks[i])
        if len(chosen) == rows:
            return sorted(chosen)
    for i in fitting:
        if i not in chosen:
            chosen.append(i)
        if len(chosen) == rows:
            break
    if len(chosen) < rows:
        raise SystemExit(f"only {len(chosen)} rows fit in {seq_len} tokens, {rows} wanted")
    return sorted(chosen)


def main() -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--subset", default="tb4")
    p.add_argument("--seq-len", type=int, default=65536)
    p.add_argument("--train-rows", type=int, default=24)
    p.add_argument("--val-rows", type=int, default=4)
    args = p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty")
    (args.out / args.subset).mkdir(parents=True, exist_ok=True)

    manifest = {"source": str(args.src), "subset": args.subset, "seq_len": args.seq_len, "splits": {}}
    for split, rows in (("train", args.train_rows), ("validation", args.val_rows)):
        files = sorted((args.src / args.subset).glob(f"{split}-*.parquet"))
        if not files:
            raise SystemExit(f"no {split}-*.parquet under {args.src / args.subset}")
        table = pa.concat_tables([pq.read_table(f) for f in files])
        chosen = pick(table, rows, args.seq_len)
        small = table.take(pa.array(chosen))
        target = args.out / args.subset / f"{split}-00000-of-00001.parquet"
        pq.write_table(small, target)
        manifest["splits"][split] = {
            "rows": small.num_rows,
            "tasks": len(set(small.column("task").to_pylist())),
            "tokens": sum(small.column("num_tokens").to_pylist()),
            "loss_tokens": sum(small.column("num_loss_tokens").to_pylist()),
            "max_row_tokens": max(small.column("num_tokens").to_pylist()),
            "source_files": {f.name: sha256(f) for f in files},
            "ids": small.column("id").to_pylist(),
        }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out / "README.md").write_text(
        "---\nconfigs:\n- config_name: "
        f"{args.subset}\n  data_files:\n  - split: train\n    path: {args.subset}/train-*.parquet\n"
        f"  - split: validation\n    path: {args.subset}/validation-*.parquet\n---\n\n"
        f"# Smoke subset of {args.src.name}\n\nWhole rows of at most {args.seq_len} tokens (manifest.json), for the "
        "in-run eval smoke run only.\n"
    )
    print(json.dumps({s: {k: v for k, v in m.items() if k != "ids"} for s, m in manifest["splits"].items()}, indent=1))


if __name__ == "__main__":
    main()
