"""Prepare the phase-2 SFT blend (vision slices) from Nemotron-Image-Training-v3.

Phase 2 trains the LM + projector (vision encoder frozen) on browser-use (converted
separately by scripts/prepare_browser_traces.py) mixed with vision slices.
Slice token budgets follow Hubert's spec scaled x0.652 so browser traces (~0.71B tokens)
is >=30% of the total mix:

  longdoc_p2    ~520M  ccpdf_01..11 (multi-page, "find the answer across pages")
  grounding_p2  ~124M  flickr30k + openimages_5
  ocr_docs_p2   ~292M  docvqa + cc3m + textcaps/textvqa_commercial
  charts_p2     ~127M  chartqa_1 + plotqa_1 (figureqa/ecd/mapqa deferred: media hosts)
  reasoning_p2  ~260M  mulberry_1/2 + aokvqa_1 + clevr_1 — <think> traces KEPT
  natural_qa_p2 ~260M  openimages_1..4

Sampling is TOKEN-budgeted: per-row cost = text_chars/3.6 + n_images * per-source
image-token estimate (Omni dynamic-res tiling) + 25/message. <think> spans are
stripped everywhere EXCEPT reasoning_p2. Rows with more than MAX_IMAGES_65K
images (won't fit 65k context) are skipped.

Writes parquet per slice (train + validation) under datasets/nemotron_vl_sft_phase2/
plus media_manifest_phase2.json for scripts/fetch_vlm_sft_media_phase2.py.

Run from the prime-rl repo root:
    uv run python scripts/prepare_vlm_sft_phase2.py
"""

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "datasets" / "raw"
OUT_DIR = REPO_ROOT / "datasets" / "nemotron_vl_sft_phase2"
MEDIA_PREFIX = "datasets/nemotron_vl_sft_phase2/media"
STAGE0_MEDIA_PREFIX = "datasets/nemotron_vl_sft/media"

SEED = 20260720
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
CHARS_PER_TOKEN = 3.6
MSG_OVERHEAD = 25
MAX_IMAGES_65K = 48  # ~48 * 1.16k image tokens + text must fit under 65536


@dataclass
class SliceSpec:
    name: str
    sources: list[str]
    token_budget: int
    keep_think: bool = False
    max_images: int | None = None
    img_tokens: dict[str, int] | None = None  # per-source estimate override


IMG_TOKENS_DEFAULT = {
    # Omni dynamic-res estimates by source family (measured on stage-0 / typical dims)
    "ccpdf": 1160,
    "openimages": 1090,
    "flickr30k": 1090,
    "docvqa": 1150,
    "cc3m": 1080,
    "chartqa": 700,
    "plotqa": 700,
    "mulberry": 950,
    "coco": 1090,
    "clevr": 258,
}

SOURCE_MEDIA_KIND = {
    **{f"long_document_ccpdf_{i:02d}": "ccpdf" for i in range(1, 12)},
    "openimages_1": "openimages",
    "openimages_2": "openimages",
    "openimages_3": "openimages",
    "openimages_4": "openimages",
    "openimages_5": "openimages",
    "textvqa_commercial": "openimages",
    "textcaps_commercial": "openimages",
    "flickr30k": "flickr30k",
    "docvqa": "docvqa",
    "cc3m": "cc3m",
    "chartqa_1": "chartqa",
    "plotqa_1": "plotqa",
    "mulberry_1": "mulberry",
    "mulberry_2": "mulberry",
    "aokvqa_1": "coco",
    "clevr_1": "clevr",
}

# Media kinds stage-0 already fetched into datasets/nemotron_vl_sft/media/<kind>/ —
# reuse those files instead of re-downloading (same flat naming per kind).
STAGE0_KINDS = {"openimages", "cc3m", "chartqa", "docvqa"}

SLICES = [
    SliceSpec(
        name="longdoc_p2",
        sources=[f"long_document_ccpdf_{i:02d}" for i in range(1, 12)],
        token_budget=520_000_000,
        max_images=MAX_IMAGES_65K,
    ),
    SliceSpec(name="grounding_p2", sources=["flickr30k", "openimages_5"], token_budget=124_000_000),
    SliceSpec(
        name="ocr_docs_p2",
        sources=["docvqa", "cc3m", "textcaps_commercial", "textvqa_commercial"],
        token_budget=292_000_000,
    ),
    SliceSpec(name="charts_p2", sources=["chartqa_1", "plotqa_1"], token_budget=127_000_000),
    SliceSpec(
        name="reasoning_p2",
        sources=["mulberry_1", "mulberry_2", "aokvqa_1", "clevr_1"],
        token_budget=260_000_000,
        keep_think=True,
    ),
    SliceSpec(
        name="natural_qa_p2",
        sources=["openimages_1", "openimages_2", "openimages_3", "openimages_4"],
        token_budget=260_000_000,
    ),
]

STRIP_PREFIXES = ("train/data/", "train_images/", "train/png/")


def media_target(source: str, raw_path: str) -> str:
    kind = SOURCE_MEDIA_KIND[source]
    rel = raw_path
    for prefix in STRIP_PREFIXES:
        rel = rel.removeprefix(prefix)
    prefix = STAGE0_MEDIA_PREFIX if kind in STAGE0_KINDS else MEDIA_PREFIX
    if kind == "ccpdf":
        return f"{prefix}/ccpdf/{source.removeprefix('long_document_ccpdf_').lstrip('0') or '0'}/{rel}"
    if kind == "mulberry":
        return f"{MEDIA_PREFIX}/mulberry/{rel}"  # keeps original nested tree
    return f"{prefix}/{kind}/{rel}"


def iter_parts(content):
    return content if isinstance(content, list) else [content]


def row_cost_and_images(row: dict, source: str, keep_think: bool) -> tuple[int, int]:
    kind = SOURCE_MEDIA_KIND[source]
    per_img = IMG_TOKENS_DEFAULT[kind]
    chars = 0
    n_img = 0
    for msg in row["messages"]:
        for p in iter_parts(msg["content"]):
            if isinstance(p, str):
                text = p if (keep_think or msg["role"] != "assistant") else THINK_RE.sub("", p)
                chars += len(text)
            elif isinstance(p, dict) and p.get("type") == "image":
                n_img += 1
    tokens = int(chars / CHARS_PER_TOKEN) + n_img * per_img + MSG_OVERHEAD * len(row["messages"])
    return tokens, n_img


def convert_row(row: dict, source: str, keep_think: bool, manifest: dict) -> dict | None:
    out_messages = []
    for msg in row["messages"]:
        parts = []
        for p in iter_parts(msg["content"]):
            if isinstance(p, str):
                text = p if (keep_think or msg["role"] != "assistant") else THINK_RE.sub("", p).strip()
                if text:
                    parts.append({"type": "text", "text": text, "image": None})
            elif isinstance(p, dict) and p.get("type") == "image":
                target = media_target(source, p["image"])
                manifest[target] = {"source": source, "raw_path": p["image"]}
                parts.append({"type": "image", "text": None, "image": target})
            else:
                raise ValueError(f"Unexpected content part in {source}: {p!r}")
        if msg["role"] == "assistant" and not any(pt["type"] == "text" for pt in parts):
            return None
        out_messages.append({"role": msg["role"], "content": parts})
    return {"id": row["id"], "source": source, "messages": out_messages}


def sample_slice(spec: SliceSpec, rng: random.Random, manifest: dict) -> list[dict]:
    """Round-robin the slice's sources, each with budget proportional to pool size,
    accumulating rows in random order until the token budget is met."""
    per_source_budget = spec.token_budget // len(spec.sources)
    rows: list[dict] = []
    carry = 0  # unspent budget rolls into subsequent sources
    for source in spec.sources:
        path = RAW_DIR / source / f"{source}.jsonl"
        if not path.exists():
            print(f"  !! missing raw file for {source}, skipping")
            continue
        offsets = []
        with open(path, "rb") as f:
            pos = 0
            for line in f:
                offsets.append(pos)
                pos += len(line)
        rng.shuffle(offsets)
        budget = per_source_budget + carry
        got = 0
        with open(path, "rb") as f:
            for off in offsets:
                if got >= budget:
                    break
                f.seek(off)
                row = json.loads(f.readline())
                cost, n_img = row_cost_and_images(row, source, spec.keep_think)
                if n_img < 1 or (spec.max_images is not None and n_img > spec.max_images):
                    continue
                if cost > 60_000:  # single row must fit the 65k context with margin
                    continue
                converted = convert_row(row, source, spec.keep_think, manifest)
                if converted is None:
                    continue
                rows.append(converted)
                got += cost
        carry = budget - got
        print(f"  {source}: {got / 1e6:.0f}M tokens ({len(rows)} rows cumulative)")
    if carry > spec.token_budget * 0.1:
        print(f"  !! {spec.name}: {carry / 1e6:.0f}M budget unfilled (pool exhausted)")
    return rows


MESSAGE_TYPE = pa.struct(
    [
        ("role", pa.string()),
        ("content", pa.list_(pa.struct([("type", pa.string()), ("text", pa.string()), ("image", pa.string())]))),
    ]
)
SCHEMA = pa.schema([("id", pa.string()), ("source", pa.string()), ("messages", pa.list_(MESSAGE_TYPE))])


def main() -> None:
    rng = random.Random(SEED)
    manifest: dict = {}
    (OUT_DIR / "data").mkdir(parents=True, exist_ok=True)

    for spec in SLICES:
        print(
            f"== {spec.name} (budget {spec.token_budget / 1e6:.0f}M, think={'KEEP' if spec.keep_think else 'strip'}) =="
        )
        rows = sample_slice(spec, rng, manifest)
        rng.shuffle(rows)
        n_val = max(20, len(rows) // 200)
        splits = {"validation": rows[:n_val], "train": rows[n_val:]}
        out = OUT_DIR / "data" / spec.name
        out.mkdir(parents=True, exist_ok=True)
        for split, split_rows in splits.items():
            pq.write_table(pa.Table.from_pylist(split_rows, schema=SCHEMA), out / f"{split}.parquet")
        print(f"  -> {len(splits['train'])} train / {n_val} validation rows")

    manifest_path = OUT_DIR / "media_manifest_phase2.json"
    manifest_path.write_text(json.dumps(manifest))
    by_kind: dict[str, int] = {}
    for target in manifest:
        kind = target.split("/media/")[1].split("/")[0]
        by_kind[kind] = by_kind.get(kind, 0) + 1
    print(f"manifest: {len(manifest)} media files -> {by_kind}")


if __name__ == "__main__":
    main()
