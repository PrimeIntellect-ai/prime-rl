"""Prepare the vision-graft stage-0 SFT alignment blend from Nemotron-Image-Training-v3.

Samples a ~155k blend from raw JSONL annotations in datasets/raw/, strips
<think> reasoning spans, normalizes messages to uniform content parts
({type, text, image}), rewrites media paths to repo-root-relative targets,
and writes per-subset parquet files plus a media fetch manifest.

Run from the repo root:
    uv run python scripts/prepare_vlm_sft_data.py
"""

import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).parent.parent
RAW_DIR = REPO_ROOT / "datasets" / "raw"
OUT_DIR = REPO_ROOT / "datasets" / "nemotron_vl_sft"
MEDIA_PREFIX = "datasets/nemotron_vl_sft/media"

SEED = 20260719
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# held-out validation rows, split proportionally across subsets
VAL_TOTAL = 1_000
BLEND_TOTAL = 155_000


@dataclass
class SubsetSpec:
    """One output subset sampled from one or more raw source files."""

    name: str
    sources: list[str]  # raw subset dirs under datasets/raw/
    target: int  # total samples across sources
    # per-source share of target; None = split proportionally to pool size
    weights: list[float] | None = None
    min_answer_words: int = 1
    max_answer_words: int | None = None
    max_images: int | None = None
    min_images: int = 1
    # exclusive bands let two subsets draw disjoint slices from the same source
    answer_band: tuple[int | None, int | None] = (None, None)


SPECS = [
    SubsetSpec(
        name="openimages_captions",
        sources=["openimages_1", "openimages_2"],
        target=55_000,
        max_answer_words=250,
    ),
    SubsetSpec(
        name="cc3m_captions_ocr",
        sources=["cc3m"],
        target=25_000,
        max_answer_words=250,
    ),
    SubsetSpec(
        name="openimages_grounding",
        sources=["openimages_5"],
        target=20_000,
    ),
    SubsetSpec(
        name="scenetext_qa",
        sources=["textvqa_commercial", "textcaps_commercial"],
        target=20_000,
        weights=[0.5, 0.5],
        max_answer_words=100,
    ),
    SubsetSpec(
        name="longdoc_multiimage",
        sources=[
            "long_document_ccpdf_02",
            "long_document_ccpdf_04",
            "long_document_ccpdf_06",
        ],
        target=15_000,
        max_images=12,  # spec asked <=8 pages, but the pool minimum is >8; 12 is the tightest feasible cap
        max_answer_words=300,
    ),
    SubsetSpec(
        name="descriptive_long",
        sources=["openimages_1", "cc3m"],
        target=10_000,
        weights=[0.5, 0.5],
        answer_band=(251, None),  # disjoint from the capped slices above
    ),
    SubsetSpec(
        name="chart_doc_qa",
        sources=["chartqa_1", "gqa_1", "docvqa"],
        target=10_000,
        weights=[0.4, 0.3, 0.3],
        max_answer_words=100,
    ),
]

SOURCE_MEDIA_KIND = {
    "openimages_1": "openimages",
    "openimages_2": "openimages",
    "openimages_5": "openimages",
    "textvqa_commercial": "openimages",
    "textcaps_commercial": "openimages",
    "cc3m": "cc3m",
    "gqa_1": "gqa",
    "chartqa_1": "chartqa",
    "docvqa": "docvqa",
    "long_document_ccpdf_02": "ccpdf",
    "long_document_ccpdf_04": "ccpdf",
    "long_document_ccpdf_06": "ccpdf",
}

# per-source raw-path prefixes stripped before building the target media path
STRIP_PREFIXES = ("train/data/", "train_images/", "train/png/")


def media_target(source: str, raw_path: str) -> str:
    """Map a raw media reference to its repo-relative target path."""
    kind = SOURCE_MEDIA_KIND[source]
    rel = raw_path
    for prefix in STRIP_PREFIXES:
        rel = rel.removeprefix(prefix)
    if kind == "ccpdf":
        # page trees from different ccpdf batches can collide; namespace by source
        return f"{MEDIA_PREFIX}/ccpdf/{source.removeprefix('long_document_ccpdf_')}/{rel}"
    return f"{MEDIA_PREFIX}/{kind}/{rel}"


def iter_parts(content) -> list:
    return content if isinstance(content, list) else [content]


def strip_think(text: str) -> str:
    return THINK_RE.sub("", text).strip()


def row_stats(row: dict) -> tuple[int, int]:
    """(num_images, max stripped answer words across assistant turns)."""
    n_img = 0
    max_words = 0
    for msg in row["messages"]:
        parts = iter_parts(msg["content"])
        if msg["role"] == "assistant":
            text = " ".join(p for p in parts if isinstance(p, str))
            words = len(strip_think(text).split())
            max_words = max(max_words, words)
        else:
            n_img += sum(1 for p in parts if isinstance(p, dict) and p.get("type") == "image")
    return n_img, max_words


def eligible(row: dict, spec: SubsetSpec) -> bool:
    n_img, ans_words = row_stats(row)
    if n_img < spec.min_images:
        return False
    if spec.max_images is not None and n_img > spec.max_images:
        return False
    lo, hi = spec.answer_band
    if lo is not None and ans_words < lo:
        return False
    if hi is not None and ans_words > hi:
        return False
    if ans_words < spec.min_answer_words:
        return False
    if spec.max_answer_words is not None and ans_words > spec.max_answer_words:
        return False
    return True


def convert_row(row: dict, source: str, manifest: dict) -> dict | None:
    """Normalize one raw row to uniform content parts with rewritten media paths."""
    out_messages = []
    for msg in row["messages"]:
        parts = []
        for p in iter_parts(msg["content"]):
            if isinstance(p, str):
                text = strip_think(p) if msg["role"] == "assistant" else p
                if text:
                    parts.append({"type": "text", "text": text, "image": None})
            elif isinstance(p, dict) and p.get("type") == "image":
                target = media_target(source, p["image"])
                manifest[target] = {"source": source, "raw_path": p["image"]}
                parts.append({"type": "image", "text": None, "image": target})
            else:
                raise ValueError(f"Unexpected content part in {source}: {p!r}")
        if msg["role"] == "assistant" and not any(p["type"] == "text" for p in parts):
            return None  # answer was pure <think> with nothing left
        out_messages.append({"role": msg["role"], "content": parts})
    return {"id": row["id"], "source": source, "messages": out_messages}


def sample_source(path: Path, spec: SubsetSpec, n: int, rng: random.Random) -> list[dict]:
    """Two-pass sampling: collect eligible byte offsets, then load a random subset."""
    offsets = []
    with open(path, "rb") as f:
        pos = 0
        for line in f:
            row = json.loads(line)
            if eligible(row, spec):
                offsets.append(pos)
            pos += len(line)
    if len(offsets) < n:
        print(f"  WARNING: {path.name} pool {len(offsets)} < target {n}, taking all")
        picked = offsets
    else:
        picked = rng.sample(offsets, n)
    picked.sort()
    rows = []
    with open(path, "rb") as f:
        for off in picked:
            f.seek(off)
            rows.append(json.loads(f.readline()))
    return len(offsets), rows


MESSAGES_TYPE = pa.list_(
    pa.struct(
        [
            ("role", pa.string()),
            (
                "content",
                pa.list_(pa.struct([("type", pa.string()), ("text", pa.string()), ("image", pa.string())])),
            ),
        ]
    )
)
SCHEMA = pa.schema([("id", pa.string()), ("source", pa.string()), ("messages", MESSAGES_TYPE)])


def main():
    rng = random.Random(SEED)
    manifest: dict[str, dict] = {}
    summary = []

    for spec in SPECS:
        print(f"== {spec.name} (target {spec.target}) ==")
        # pool sizes first, to split the target proportionally across sources
        pools, rows_per_source = {}, {}
        for source in spec.sources:
            path = RAW_DIR / source / f"{source}.jsonl"
            pool, rows = sample_source(path, spec, spec.target, rng)  # oversample; trim below
            pools[source] = pool
            rows_per_source[source] = rows
        total_pool = sum(pools.values())
        converted, dropped = [], 0
        for i, source in enumerate(spec.sources):
            frac = spec.weights[i] if spec.weights else pools[source] / total_pool
            share = round(spec.target * frac)
            take = rows_per_source[source][:share]
            for row in take:
                out = convert_row(row, source, manifest)
                if out is None:
                    dropped += 1
                else:
                    converted.append(out)
            print(f"  {source}: pool={pools[source]} took={len(take)}")
        rng.shuffle(converted)
        val_n = round(len(converted) * VAL_TOTAL / BLEND_TOTAL)
        splits = {"validation": converted[:val_n], "train": converted[val_n:]}
        for split, rows in splits.items():
            out_path = OUT_DIR / "data" / spec.name / f"{split}.parquet"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), out_path)
        print(f"  wrote {len(splits['train'])} train + {val_n} validation rows ({dropped} dropped empty-after-strip)")
        summary.append((spec.name, len(splits["train"]), val_n))

    manifest_path = OUT_DIR / "media_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    print(f"\nmedia manifest: {len(manifest)} unique files -> {manifest_path}")
    by_kind = Counter(t.split("/")[3] for t in manifest)
    for kind, cnt in by_kind.most_common():
        print(f"  {kind}: {cnt}")
    print("\nblend summary (train/validation):")
    for name, n_train, n_val in summary:
        print(f"  {name}: {n_train}/{n_val}")
    print(f"  TOTAL: {sum(n for _, n, _ in summary)}/{sum(v for _, _, v in summary)}")


if __name__ == "__main__":
    main()
