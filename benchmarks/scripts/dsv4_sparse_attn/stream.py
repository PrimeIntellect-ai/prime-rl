"""Replay the corpus's dynamic stream once per backend, in fresh processes with fresh compile caches, cold then warm.

usage (repo root, on an otherwise idle GPU):
  uv run --no-sync python benchmarks/scripts/dsv4_sparse_attn/stream.py --out RESULTS.json [--label NAME]
      [--corpus DIR] [--backends NAME ...] [--cache-root DIR]
  uv run --no-sync python benchmarks/scripts/dsv4_sparse_attn/stream.py --compare A.json [B.json ...]

Each backend gets a new cache directory holding the TileLang, Triton, Inductor, CuTe DSL and CUDA caches and
TMPDIR. A first process replays the stream against the empty caches (cold), a second process replays it
against what the first left behind (warm). Every item runs once, forward+backward (forward alone for a
forward-only arm), and its first-call time is measured with the GPU idle before and after. Compiles are counted
exactly by the backend's compile-entry-point wrapper, per item; new cache-directory entries are a cross-check.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
from common import (
    DEFAULT_CORPUS_DIR,
    SM_SCALE,
    backend_names,
    forward_backward,
    load_backend,
    load_indices,
    load_manifest,
    make_inputs,
    provenance,
    stream_items,
)

CACHE_ENV = {
    "TILELANG_CACHE_DIR": "tilelang",
    "TRITON_CACHE_DIR": "triton",
    "TORCHINDUCTOR_CACHE_DIR": "inductor",
    "CUTE_DSL_CACHE_DIR": "cute_dsl",
    "CUDA_CACHE_PATH": "cuda",
    "TMPDIR": "tmp",
}
DEFAULT_CACHE_ROOT = Path("~/tmp/dsv4-sparse-attn-bench/stream-caches").expanduser()
PHASES = ("cold", "warm")


def cache_entries(cache_root: Path) -> dict[str, dict[str, int]]:
    """Per cache, the number of files and of leaf directories under it."""
    entries = {}
    for subdir in CACHE_ENV.values():
        files = leaf_dirs = 0
        for _, dirnames, filenames in os.walk(cache_root / subdir):
            files += len(filenames)
            leaf_dirs += not dirnames
        entries[subdir] = {"files": files, "leaf_dirs": leaf_dirs}
    return entries


def worker(name: str, corpus_dir: Path, result_path: Path) -> None:
    backend = load_backend(name)
    read_counts = backend.install_compile_counter()
    manifest = load_manifest(corpus_dir)
    records = []
    for position, item in enumerate(stream_items(manifest)):
        indices = load_indices(corpus_dir, item)
        inputs = make_inputs(item, requires_grad=not backend.FORWARD_ONLY)
        args = (inputs["q"], inputs["kv"], indices, inputs["sinks"], SM_SCALE)
        torch.cuda.synchronize()
        before = read_counts()
        start = time.perf_counter()
        if backend.FORWARD_ONLY:
            with torch.no_grad():
                backend.fwd(*args)
        else:
            forward_backward(backend, *args, inputs["grad_out"])
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1e3
        after = read_counts()
        records.append(
            {
                "position": position,
                "id": item.id,
                "total_len": item.total_len,
                "n_queries": item.n_queries,
                "n_positions": item.n_positions,
                "n_slots": item.n_slots,
                "first_call_ms": elapsed_ms,
                "counts": {key: after[key] - before.get(key, 0) for key in after},
            }
        )
        print(f"{position:2d} {item.id}: {elapsed_ms:.1f} ms, {records[-1]['counts']}", flush=True)
        del inputs, indices, args
    result = {
        "stream_total_s": sum(record["first_call_ms"] for record in records) / 1e3,
        "counts": read_counts(),
        "items": records,
    }
    result_path.write_text(json.dumps(result, indent=1))


def replay(name: str, corpus_dir: Path, cache_root: Path) -> dict:
    cache_dir = Path(tempfile.mkdtemp(prefix=f"{name}-", dir=cache_root))
    env = dict(os.environ)
    for variable, subdir in CACHE_ENV.items():
        (cache_dir / subdir).mkdir()
        env[variable] = str(cache_dir / subdir)
    phases = {}
    for phase in PHASES:
        result_path = cache_dir / f"{phase}.json"
        before = cache_entries(cache_dir)
        start = time.perf_counter()
        command = [
            sys.executable,
            __file__,
            "--worker",
            name,
            "--corpus",
            str(corpus_dir),
            "--result",
            str(result_path),
        ]
        subprocess.run(command, env=env, check=True)
        process_wall_s = time.perf_counter() - start
        after = cache_entries(cache_dir)
        phases[phase] = json.loads(result_path.read_text()) | {
            "process_wall_s": process_wall_s,
            "new_cache_entries": {
                subdir: {key: after[subdir][key] - before[subdir][key] for key in after[subdir]} for subdir in after
            },
        }
    return {"cache_dir": str(cache_dir), "phases": phases}


def print_tables(runs: list[dict]) -> None:
    hashes = {run["provenance"]["corpus_hash"] for run in runs}
    if len(hashes) > 1:
        raise SystemExit(f"refusing to compare results built on different corpora: {sorted(hashes)}")
    multiple = len(runs) > 1
    arms = [
        (f"{name}@{run['label']}" if multiple else name, run, replayed)
        for run in runs
        for name, replayed in run["backends"].items()
    ]
    print(f"Dynamic stream of {len(runs[0]['stream'])} items, each called once; times in seconds, lower is better.")
    print("`compiles` and `loads` come from the compile-entry-point wrapper; `new files` from the cache dirs.\n")
    print("| backend | phase | stream total s | process wall s | compiles | loads | new cache files |")
    print("|---|---|---|---|---|---|---|")
    for label, _run, replayed in arms:
        for phase in PHASES:
            result = replayed["phases"][phase]
            new_files = sum(entry["files"] for entry in result["new_cache_entries"].values())
            print(
                f"| {label} | {phase} | {result['stream_total_s']:.2f} | {result['process_wall_s']:.1f} | "
                f"{result['counts'].get('compiles', 0)} | {result['counts'].get('disk_loads', '-')} | {new_files} |"
            )

    print("\nPer-item first-call time in ms (lower is better) and compiles triggered by that item.\n")
    print("| # | item | backend | cold ms | cold compiles | warm ms | warm compiles |")
    print("|---|---|---|---|---|---|---|")
    for position, item_id in enumerate(runs[0]["stream"]):
        for label, _run, replayed in arms:
            cold, warm = (replayed["phases"][phase]["items"][position] for phase in PHASES)
            print(
                f"| {position} | {item_id} | {label} | {cold['first_call_ms']:.1f} | "
                f"{cold['counts'].get('compiles', 0)} | {warm['first_call_ms']:.1f} | {warm['counts'].get('compiles', 0)} |"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compare", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--label", default=None)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--backends", nargs="+", default=None)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        worker(args.worker, args.corpus, args.result)
        return
    if args.compare:
        print_tables([json.loads(path.read_text()) for path in args.compare])
        return
    if args.out is None:
        parser.error("--out is required unless --compare is given")

    manifest = load_manifest(args.corpus)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    backends, skipped = {}, {}
    for name in args.backends or backend_names():
        reason = load_backend(name).unavailable_reason()
        if reason is None:
            backends[name] = replay(name, args.corpus, args.cache_root)
        else:
            skipped[name] = reason
            print(f"skipping {name}: {reason}")
    prov = provenance(manifest["corpus_hash"])
    result = {
        "label": args.label or prov["git"]["sha"][:9],
        "provenance": prov,
        "stream": manifest["stream"],
        "backends": backends,
        "skipped_backends": skipped,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print_tables([result])


if __name__ == "__main__":
    main()
