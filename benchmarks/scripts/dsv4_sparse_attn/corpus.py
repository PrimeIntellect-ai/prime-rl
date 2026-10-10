"""Generate the synthetic DSv4 sparse attention corpus, or check that a rerun reproduces it.

usage (repo root, on a GPU node):
  uv run --no-sync python benchmarks/scripts/dsv4_sparse_attn/corpus.py [--out DIR] [--check]

Each packed row (a composition at a total length) is run through one real DeepSeek V4 Flash attention layer
per layer type, with random weights, and the gather indices the layer hands the kernel are recorded. Only
structure is stored: the int32 indices, their KV position count, the document lengths, and a seed from which
the benchmark regenerates `q`, `kv`, `sinks` and `dO`. Under context parallelism a rank's indices are the
matching rows of the whole row's, so cp = 8 items are row slices for the first, middle and last rank.
`--check` regenerates every item and compares its indices hash against the existing manifest.
"""

import argparse
import hashlib
import json
import math
import random
import socket
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch
from common import DEFAULT_CORPUS_DIR, git_state, indices_sha256, slot_coverage

from tests.unit.train.models.test_deepseek_v4_kernels import (
    V4FLASH_CONFIG,
    V4FLASH_CSA_LAYER,
    V4FLASH_HCA_LAYER,
    V4FLASH_SLIDING_LAYER,
    _packed_context,
    _record_attention,
    v4flash_attention,
)

GRID_LENGTHS = [2048, 4096, 16384, 49208, 65536]
COMPOSITIONS = ["single", "short", "heavy", "tiny"]
LAYERS = {"csa": V4FLASH_CSA_LAYER, "hca": V4FLASH_HCA_LAYER, "sliding": V4FLASH_SLIDING_LAYER}
CP = 8
CP_RANKS = [0, CP // 2, CP - 1]
STREAM_LENGTH = 32
STREAM_MIN_LEN, STREAM_MAX_LEN = 2048, 65536
STREAM_CP_FRACTION = 0.25
SEED = 0
SHORT_DOC_MEDIAN = 1024


def stable_seed(*parts: object) -> int:
    return zlib.crc32("/".join(str(part) for part in parts).encode())


def doc_lens(composition: str, total_len: int, seed: int) -> list[int]:
    """Document lengths summing to `total_len`, the last one truncated to fit."""
    if composition == "single":
        return [total_len]
    rng = random.Random(seed)
    draws = {
        "short": lambda: rng.lognormvariate(mu=math.log(SHORT_DOC_MEDIAN), sigma=0.75),
        "heavy": lambda: 128 * rng.paretovariate(1.1),
        "tiny": lambda: rng.randint(1, 127),
    }[composition]
    lengths = []
    while sum(lengths) < total_len:
        lengths.append(max(1, int(draws())))
    lengths[-1] -= sum(lengths) - total_len
    return [length for length in lengths if length > 0]


def record_indices(module: torch.nn.Module, lens: list[int], seed: int) -> tuple[torch.Tensor, int]:
    """The `(n_tokens, n_slots)` indices one attention layer builds for this row, and its KV position count."""
    total_len = sum(lens)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    hidden = torch.randn(
        1, total_len, V4FLASH_CONFIG.hidden_size, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    with pytest.MonkeyPatch.context() as monkeypatch, torch.no_grad():
        recorded = _record_attention(monkeypatch)
        module(hidden, packed=_packed_context(tuple(lens), torch.bfloat16, V4FLASH_CONFIG))
    return recorded["indices"][0, :, 0, :].contiguous(), recorded["kv_buf"].shape[1]


def row_specs(seed: int) -> tuple[list[dict], list[str]]:
    """The packed rows to generate, each with its `(cp, cp_rank)` shards per layer, and the stream's item ids."""
    grid_shards = [(1, 0)] + [(CP, rank) for rank in CP_RANKS]
    rows = [
        {
            "row": f"{composition}-{total_len}",
            "composition": composition,
            "total_len": total_len,
            "shards": {layer: grid_shards for layer in LAYERS},
        }
        for total_len in GRID_LENGTHS
        for composition in COMPOSITIONS
    ]
    rng = random.Random(stable_seed(seed, "stream"))
    stream = []
    for position in range(STREAM_LENGTH):
        total_len = rng.randrange(STREAM_MIN_LEN, STREAM_MAX_LEN + 1, CP)
        composition = rng.choice(COMPOSITIONS)
        layer = rng.choice(sorted(LAYERS))
        cp, cp_rank = (CP, rng.randrange(CP)) if rng.random() < STREAM_CP_FRACTION else (1, 0)
        row = f"stream{position:02d}-{composition}-{total_len}"
        rows.append(
            {"row": row, "composition": composition, "total_len": total_len, "shards": {layer: [(cp, cp_rank)]}}
        )
        stream.append(item_id(row, layer, cp, cp_rank))
    return rows, stream


def item_id(row: str, layer: str, cp: int, cp_rank: int) -> str:
    return f"{row}-{layer}-cp1" if cp == 1 else f"{row}-{layer}-cp{cp}r{cp_rank}"


def generate(out_dir: Path | None, seed: int) -> dict:
    """Build every item; write its indices under `out_dir` unless it is `None`. Returns the manifest."""
    rows, stream = row_specs(seed)
    modules = {}
    for layer, layer_idx in LAYERS.items():
        torch.manual_seed(stable_seed(seed, "weights", layer))
        modules[layer] = v4flash_attention(layer_idx, dtype=torch.bfloat16)

    items = []
    for spec in rows:
        lens = doc_lens(spec["composition"], spec["total_len"], stable_seed(seed, "docs", spec["row"]))
        for layer, shards in spec["shards"].items():
            indices, n_positions = record_indices(modules[layer], lens, stable_seed(seed, "hidden", spec["row"]))
            for cp, cp_rank in shards:
                n_queries = spec["total_len"] // cp
                shard = indices[cp_rank * n_queries : (cp_rank + 1) * n_queries].contiguous()
                identifier = item_id(spec["row"], layer, cp, cp_rank)
                coverage = slot_coverage(shard[None, :, None, :])
                n_valid = coverage["n_valid"].float()
                entry = {
                    "id": identifier,
                    "row": spec["row"],
                    "composition": spec["composition"],
                    "total_len": spec["total_len"],
                    "doc_lens": lens,
                    "max_doc_len": max(lens),
                    "n_docs": len(lens),
                    "layer_type": layer,
                    "cp": cp,
                    "cp_rank": cp_rank,
                    "n_queries": n_queries,
                    "n_positions": n_positions,
                    "n_slots": shard.shape[-1],
                    "seed": stable_seed(seed, "inputs", identifier),
                    "file": f"indices/{identifier}.pt",
                    "indices_sha256": indices_sha256(shard),
                    "sum_valid": int(coverage["n_valid"].sum()),
                    "n_valid_min": int(n_valid.min()),
                    "n_valid_median": float(n_valid.median()),
                    "n_valid_max": int(n_valid.max()),
                }
                if out_dir is not None:
                    torch.save(shard.cpu(), out_dir / entry["file"])
                items.append(entry)
                print(f"{identifier}: n_slots {entry['n_slots']}, n_positions {n_positions}", flush=True)

    structure = [(entry["id"], entry["indices_sha256"], entry["n_positions"], entry["seed"]) for entry in items]
    corpus_hash = hashlib.sha256(json.dumps({"items": structure, "stream": stream}).encode()).hexdigest()
    return {
        "corpus_hash": corpus_hash,
        "generator": git_state(),
        "created": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(),
        "seed": seed,
        "items": items,
        "stream": stream,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--check", action="store_true", help="regenerate and compare hashes with the manifest")
    args = parser.parse_args()

    if args.check:
        existing = json.loads((args.out / "manifest.json").read_text())
        regenerated = generate(None, existing["seed"])
        expected = {entry["id"]: entry["indices_sha256"] for entry in existing["items"]}
        actual = {entry["id"]: entry["indices_sha256"] for entry in regenerated["items"]}
        mismatched = sorted(item for item in expected.keys() | actual.keys() if expected.get(item) != actual.get(item))
        print(f"corpus hash: manifest {existing['corpus_hash']}, regenerated {regenerated['corpus_hash']}")
        if mismatched:
            print(f"{len(mismatched)} items differ: {mismatched}")
            sys.exit(1)
        print(f"all {len(actual)} items reproduce")
        return

    (args.out / "indices").mkdir(parents=True, exist_ok=True)
    manifest = generate(args.out, args.seed)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(manifest['items'])} items, corpus hash {manifest['corpus_hash']}")


if __name__ == "__main__":
    main()
