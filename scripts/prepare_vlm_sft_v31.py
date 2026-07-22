"""Prepare the Blend v3.1 SFT mixture (single 65k stage, FFT incl. vision encoder).

Eight subsets (= per-source loss curves), token budgets per the blend-v3.1 doc:

  browser_use_sft_dataset  passthrough of the existing corpus (native 65k; the run
                           config oversamples it ~2x via interleave probabilities —
                           the doc's "browser-epoch reserve lever")
  gui_v31          1.6B   AgentNet + ScaleCUA + GroundCUA + ScreenQA + VisualWebInstruct
                          (~55% grounding / ~30% short action sequences / ~15% screen QA)
  docs_ocr_v31     1.8B   ITv3 docvqa/textcaps*/textvqa/hiertext/pubtables/fintabnet/
                          sa_finance_1-5/cc3m + long_document_ccpdf/sec/arxiv (uncapped
                          multi-image) + Docmatix sample (~400M)
  charts_v31       850M   ITv3 chartqa_1/figureqa/plotqa/mapqa/ecd + FineVision chart
                          slices (top-rated) as the CharXiv substitute
  reasoning_v31    1.8B   ITv3 mulberry/aokvqa/clevr_1/geometry cluster (<think> KEPT,
                          style-filtered) + Vision-R1-cold + LLaVA-CoT-100k + MathV360K
                          + MMK12 (boxed final answers)
  perception_v31   1.1B   PixMo-Cap + PixMo-Points (point/box format) + Localized
                          Narratives (sampled) + ITv3 flickr30k/openimages_5/toloka +
                          openimages_1-4 small (~250M ballast)
  general_v31      600M   FineVision top-rated general slices + Honey-Data-1M sample
  text_replay_v31  2.0B   Nemotron-Post-Training-v3 proportional, tool/agentic 2x
                          (Ultra fine-tune SFT data slots in here when located)

All rows use the unified schema (id, source, tools JSON-string, messages with
content parts {type,text,image} and tool_calls with arguments-as-JSON-string).
InternVL coordinate convention: ints normalized to 0-1000 per axis,
<box>[[x1,y1,x2,y2]]</box>, <point>[[x,y],...]</point>, <ref>text</ref>.
GUI trajectories are rewritten into the computer_* tool vocabulary below.

Run per-slice on the data host (raw sources under $RAW_ROOT):
    uv run python scripts/prepare_vlm_sft_v31.py --slices gui_v31 charts_v31 ...
Media manifest entries are written for scripts/fetch_v31_media.py; sources with
embedded images (Docmatix/Honey/FineVision) write media files directly.
"""

import argparse
import hashlib
import json
import os
import random
import re
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = Path(os.environ.get("RAW_ROOT", "/shared/hubert/raw"))
OUT_DIR = Path(os.environ.get("OUT_ROOT", "/shared/hubert/datasets/nemotron_vl_sft_v31"))
MEDIA_ROOT = Path(os.environ.get("MEDIA_ROOT", "/shared/hubert/datasets/media_v31"))
MEDIA_PREFIX = "datasets/media_v31"  # row-visible path (repo has datasets -> /shared/hubert/datasets)
BROWSER_SRC = Path(
    os.environ.get("BROWSER_SRC", "/shared/hubert/datasets/nemotron_vl_sft_phase2/data/browser_use_sft_dataset")
)

SEED = 20260722
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
CHARS_PER_TOKEN = 3.6
MSG_OVERHEAD = 25
MAX_IMAGES_65K = 48
MAX_ROW_TOKENS = 60_000

# Foreign-model tells / refusal boilerplate: rows containing these are dropped
# from reasoning/general slices (style filter).
STYLE_FILTER_RE = re.compile(
    r"as an ai\b|i'?m sorry, (?:but )?i can|language model|openai|chatgpt|gpt-4|gemini|claude|"
    r"i cannot assist|<summary>|</conclusion>",
    re.IGNORECASE,
)

COMPUTER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "computer_click",
            "description": "Click at a screen position. Coordinates are integers normalized to 0-1000 on each axis.",
            "parameters": {
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "button": {"type": "string", "enum": ["left", "right", "middle"], "default": "left"},
                    "double": {"type": "boolean", "default": False},
                },
                "required": ["x", "y"],
                "type": "object",
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "computer_type",
            "description": "Type text with the keyboard.",
            "parameters": {"properties": {"text": {"type": "string"}}, "required": ["text"], "type": "object"},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "computer_key",
            "description": "Press a key or chord, e.g. 'enter', 'ctrl+s'.",
            "parameters": {"properties": {"keys": {"type": "string"}}, "required": ["keys"], "type": "object"},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "computer_scroll",
            "description": "Scroll at a position. dy>0 scrolls down.",
            "parameters": {
                "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}, "dy": {"type": "integer"}},
                "required": ["dy"],
                "type": "object",
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "computer_drag",
            "description": "Drag from one position to another.",
            "parameters": {
                "properties": {
                    "x1": {"type": "integer"},
                    "y1": {"type": "integer"},
                    "x2": {"type": "integer"},
                    "y2": {"type": "integer"},
                },
                "required": ["x1", "y1", "x2", "y2"],
                "type": "object",
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "computer_wait",
            "description": "Wait for the screen to settle.",
            "parameters": {"properties": {}, "type": "object"},
        },
    },
]
COMPUTER_TOOLS_JSON = json.dumps(COMPUTER_TOOLS)


def norm_coord(v: float, size: float) -> int:
    return max(0, min(1000, round(v / size * 1000)))


def fmt_box(x1, y1, x2, y2, w, h) -> str:
    return f"<box>[[{norm_coord(x1, w)},{norm_coord(y1, h)},{norm_coord(x2, w)},{norm_coord(y2, h)}]]</box>"


def fmt_points(points: list[tuple[float, float]], w: float, h: float) -> str:
    inner = ",".join(f"[{norm_coord(x, w)},{norm_coord(y, h)}]" for x, y in points)
    return f"<point>[{inner}]</point>"


def png_jpeg_dims(data: bytes) -> tuple[int, int] | None:
    """Image dimensions from header bytes without a full decode."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        w, h = struct.unpack(">II", data[16:24])
        return w, h
    if data[:2] == b"\xff\xd8":  # JPEG: scan for SOFn
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                h, w = struct.unpack(">HH", data[i + 5 : i + 9])
                return w, h
            i += 2 + struct.unpack(">H", data[i + 2 : i + 4])[0]
    return None


def est_image_tokens(w: int | None = None, h: int | None = None, default: int = 1000) -> int:
    """Omni dynamic-res token estimate: patch grid clamped to [1024, 13312], /4 shuffle."""
    if not w or not h:
        return default
    patches = max(1024, min(13312, (w // 16) * (h // 16)))
    return patches // 4


def text_part(text: str) -> dict:
    return {"type": "text", "text": text, "image": None}


def image_part(path: str) -> dict:
    return {"type": "image", "text": None, "image": path}


def make_row(rid: str, source: str, messages: list[dict], tools: str = "[]") -> dict:
    for m in messages:
        m.setdefault("tool_calls", None)
    return {"id": rid, "source": source, "tools": tools, "messages": messages}


def row_text_chars(messages: list[dict]) -> int:
    chars = 0
    for m in messages:
        for p in m["content"]:
            if p.get("text"):
                chars += len(p["text"])
        for tc in m.get("tool_calls") or []:
            chars += len(tc["function"]["name"]) + len(tc["function"]["arguments"]) + 20
    return chars


def row_cost(messages: list[dict], image_tokens: list[int]) -> int:
    return int(row_text_chars(messages) / CHARS_PER_TOKEN) + sum(image_tokens) + MSG_OVERHEAD * len(messages)


class MediaStore:
    """Writes embedded images to MEDIA_ROOT/<kind>/<sha1>.<ext>, dedup by content hash."""

    def __init__(self, kind: str):
        self.kind = kind
        self.dir = MEDIA_ROOT / kind
        self.dir.mkdir(parents=True, exist_ok=True)

    def put(self, data: bytes, ext: str = "jpg") -> tuple[str, tuple[int, int] | None]:
        sha = hashlib.sha1(data).hexdigest()
        fname = f"{sha}.{ext}"
        fpath = self.dir / fname
        if not fpath.exists():
            fpath.write_bytes(data)
        return f"{MEDIA_PREFIX}/{self.kind}/{fname}", png_jpeg_dims(data)


# ---------------------------------------------------------------------------
# Converters. Each yields (row, cost_tokens). `manifest` collects media that a
# separate fetch step must materialize: target_path -> {kind, ref}.
# ---------------------------------------------------------------------------


def _first_existing(*paths: Path) -> Path | None:
    return next((p for p in paths if p.exists()), None)


# Subset -> media kind. Kinds with dedicated fetch handlers in
# scripts/fetch_vlm_sft_media.py (openimages/cc3m/chartqa/docvqa/ccpdf/coco/
# mulberry/clevr/plotqa/flickr30k) reuse them; every other kind is resolved by
# the generic `archives` lane from the downloaded per-subset media archives.
ITV3_MEDIA_KIND = {
    **{f"long_document_ccpdf_{i:02d}": "ccpdf" for i in range(1, 12)},
    **{f"openimages_{i}": "openimages" for i in range(1, 6)},
    **{f"sa_finance_{i}": "sa_finance" for i in range(1, 6)},
    "textvqa_commercial": "openimages",
    "textcaps_commercial": "openimages",
    "textcaps": "coco",
    "flickr30k": "flickr30k",
    "docvqa": "docvqa",
    "cc3m": "cc3m",
    "chartqa_1": "chartqa",
    "plotqa_1": "plotqa",
    "plotqa_2": "plotqa",
    "mulberry_1": "mulberry",
    "mulberry_2": "mulberry",
    "aokvqa_1": "coco",
    "aokvqa_2": "coco",
    "clevr_1": "clevr",
}
ITV3_STRIP_PREFIXES = ("train/data/", "train_images/", "train/png/")


def itv3_media_target(source: str, raw_path: str) -> tuple[str, str]:
    kind = ITV3_MEDIA_KIND.get(source, source)
    rel = raw_path
    for prefix in ITV3_STRIP_PREFIXES:
        rel = rel.removeprefix(prefix)
    if kind == "ccpdf":
        batch = source.removeprefix("long_document_ccpdf_").lstrip("0") or "0"
        return kind, f"{MEDIA_PREFIX}/media/{kind}/{batch}/{rel}"
    return kind, f"{MEDIA_PREFIX}/media/{kind}/{rel}"


def iter_itv3(source: str, keep_think: bool, manifest: dict, rng: random.Random) -> Iterator[tuple[dict, int]]:
    """nvidia house format: {id, messages} with content = [str | {type:image,image:relpath}]."""
    base = _first_existing(
        RAW_ROOT / "nemotron_image_v3" / source,
        RAW_ROOT / "Nemotron-Image-Training-v3" / source,
        RAW_ROOT / "itv3" / source,
        REPO_ROOT / "datasets" / "raw" / source,
    )
    if base is None:
        print(f"  !! itv3/{source}: raw not found, skipping")
        return
    files = sorted(base.rglob("*.jsonl")) or sorted(base.rglob("*.parquet"))
    rng.shuffle(files)
    per_img = (
        700
        if any(k in source for k in ("chartqa", "plotqa", "figureqa", "ecd", "mapqa"))
        else (258 if "clevr" in source else 1100)
    )

    def rows_from(f: Path) -> Iterator[dict]:
        if f.suffix == ".jsonl":
            idx = Path(str(f) + ".idx")
            if idx.exists():
                # uint64 LE byte offsets, one per line + end sentinel: sample rows
                # in shuffled order so budget-capped slices draw uniformly.
                import numpy as np

                offsets = np.fromfile(idx, dtype="<u8")[:-1]
                order = list(range(len(offsets)))
                rng.shuffle(order)
                with open(f, "rb") as fh:
                    for i in order:
                        fh.seek(int(offsets[i]))
                        yield json.loads(fh.readline())
            else:
                with open(f) as fh:
                    for line in fh:
                        yield json.loads(line)
        else:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=256):
                yield from batch.to_pylist()

    for f in files:
        for raw in rows_from(f):
            messages, img_tokens, ok = [], [], True
            for msg in raw["messages"]:
                parts = []
                content = msg["content"] if isinstance(msg["content"], list) else [msg["content"]]
                for p in content:
                    if isinstance(p, str):
                        text = p if (keep_think or msg["role"] != "assistant") else THINK_RE.sub("", p).strip()
                        if text:
                            parts.append(text_part(text))
                    elif isinstance(p, dict) and p.get("type") == "image":
                        kind, target = itv3_media_target(source, p["image"])
                        manifest[target] = {"kind": kind, "ref": p["image"], "source": source, "raw_path": p["image"]}
                        parts.append(image_part(target))
                        img_tokens.append(per_img)
                    else:
                        ok = False
                if msg["role"] == "assistant" and not any(pt["type"] == "text" for pt in parts):
                    ok = False
                messages.append({"role": msg["role"], "content": parts})
            if not ok or not img_tokens or len(img_tokens) > MAX_IMAGES_65K:
                continue
            yield make_row(raw["id"], source, messages), row_cost(messages, img_tokens)


def iter_sharegpt(
    name: str,
    base: Path,
    image_root_kind: str,
    manifest: dict,
    rng: random.Random,
    transform: Callable[[str], str | None] | None = None,
) -> Iterator[tuple[dict, int]]:
    """ShareGPT format: {id?, image, conversations:[{from: human|gpt, value}]}."""
    files = sorted(base.rglob("*.json")) + sorted(base.rglob("*.jsonl")) + sorted(base.rglob("*.parquet"))
    files = [f for f in files if "meta" not in f.name.lower()]
    rng.shuffle(files)

    def rows_from(f: Path) -> Iterator[dict]:
        if f.suffix == ".json":
            yield from json.loads(f.read_text())
        elif f.suffix == ".jsonl":
            with open(f) as fh:
                for line in fh:
                    yield json.loads(line)
        else:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=256):
                yield from batch.to_pylist()

    for f in files:
        for i, raw in enumerate(rows_from(f)):
            convs = raw.get("conversations") or []
            images = raw.get("image")
            images = [images] if isinstance(images, str) else (images or [])
            if not convs or not images or len(images) > MAX_IMAGES_65K:
                continue
            messages, img_tokens, placed, ok = [], [], 0, True
            for turn in convs:
                role = {"human": "user", "gpt": "assistant"}.get(turn["from"])
                if role is None:
                    ok = False
                    break
                value = turn["value"]
                if role == "assistant" and transform is not None:
                    value = transform(value)
                    if value is None:
                        ok = False
                        break
                parts = []
                n_ph = value.count("<image>")
                value_clean = value.replace("<image>", "").strip()
                if role == "user":
                    take = images[placed : placed + n_ph] if n_ph else ([] if placed else images)
                    for img in take:
                        target = f"{MEDIA_PREFIX}/{image_root_kind}/{img}"
                        manifest[target] = {"kind": image_root_kind, "ref": img, "source": name}
                        parts.append(image_part(target))
                        img_tokens.append(1000)
                        placed += 1
                if value_clean:
                    parts.append(text_part(value_clean))
                if parts:
                    messages.append({"role": role, "content": parts})
            if not ok or not img_tokens or not messages or messages[-1]["role"] != "assistant":
                continue
            rid = str(raw.get("id") or f"{name}-{f.stem}-{i}")
            yield make_row(rid, name, messages), row_cost(messages, img_tokens)


def llava_cot_transform(value: str) -> str | None:
    """<SUMMARY>/<CAPTION>/<REASONING> -> <think> block; <CONCLUSION> -> answer."""
    m = re.search(r"<CONCLUSION>\s*(.*?)\s*</CONCLUSION>", value, re.DOTALL)
    if not m:
        return None
    conclusion = m.group(1).strip()
    inner = re.sub(r"</?(?:SUMMARY|CAPTION|REASONING)>", "", value[: m.start()]).strip()
    return f"<think>\n{inner}\n</think>\n{conclusion}"


def boxed_transform(value: str) -> str | None:
    m = re.search(r"[Tt]he answer is[:\s]*(.+?)\s*$", value.strip(), re.DOTALL)
    if m and "\\boxed" not in value:
        ans = m.group(1).strip().rstrip(".")
        if len(ans) <= 80:
            return value[: m.start()].rstrip() + f"\nThe answer is \\boxed{{{ans}}}"
    return value


def iter_embedded_qa(
    name: str,
    base: Path,
    kind: str,
    rng: random.Random,
    quality_min: int | None = None,
    max_files: int | None = None,
    strip_think: bool = True,
) -> Iterator[tuple[dict, int]]:
    """Parquet with embedded images + texts/conversations (Docmatix, FineVision, Honey)."""
    store = MediaStore(kind)
    files = sorted(base.rglob("*.parquet"))
    rng.shuffle(files)
    if max_files:
        files = files[:max_files]
    for f in files:
        try:
            pf = pq.ParquetFile(f)
        except Exception as e:
            print(f"  !! {name}: unreadable {f.name}: {e}")
            continue
        for batch in pf.iter_batches(batch_size=32):
            for i, raw in enumerate(batch.to_pylist()):
                if quality_min is not None:
                    mins = [
                        raw.get(k)
                        for k in (
                            "relevance_min",
                            "image_correspondence_min",
                            "visual_dependency_min",
                            "formatting_min",
                        )
                    ]
                    if any(m is not None and m < quality_min for m in mins):
                        continue
                imgs = raw.get("images") or []
                img_paths, img_tokens = [], []
                for img in imgs[:MAX_IMAGES_65K]:
                    data = img.get("bytes") if isinstance(img, dict) else None
                    if data is None:
                        continue
                    path, dims = store.put(data, "jpg")
                    img_paths.append(path)
                    img_tokens.append(est_image_tokens(*(dims or (None, None))))
                if not img_paths:
                    continue
                messages = []
                if raw.get("texts"):  # [{user, assistant, ...}]
                    for j, qa in enumerate(raw["texts"]):
                        q, a = qa.get("user"), qa.get("assistant")
                        if not q or not a:
                            continue
                        uparts = [image_part(p) for p in img_paths] if j == 0 else []
                        uparts.append(text_part(q))
                        messages.append({"role": "user", "content": uparts})
                        messages.append({"role": "assistant", "content": [text_part(a)]})
                elif raw.get("conversations"):
                    placed = False
                    for turn in raw["conversations"]:
                        role = {"human": "user", "gpt": "assistant"}.get(turn["from"])
                        if role is None:
                            messages = []
                            break
                        value = turn["value"].replace("<image>", "").strip()
                        if role == "assistant" and strip_think:
                            value = THINK_RE.sub("", value).strip()
                        parts = []
                        if role == "user" and not placed:
                            parts.extend(image_part(p) for p in img_paths)
                            placed = True
                        if value:
                            parts.append(text_part(value))
                        if parts:
                            messages.append({"role": role, "content": parts})
                if not messages or messages[-1]["role"] != "assistant":
                    continue
                full_text = " ".join(p["text"] or "" for m in messages for p in m["content"])
                if STYLE_FILTER_RE.search(full_text):
                    continue
                rid = str(raw.get("id") or f"{name}-{f.stem}-{i}")
                yield make_row(rid, name, messages), row_cost(messages, img_tokens)


def iter_groundcua(manifest: dict, rng: random.Random) -> Iterator[tuple[dict, int]]:
    """Per-element rows grouped by screenshot -> multi-QA grounding conversations."""
    base = _first_existing(RAW_ROOT / "GroundCUA", RAW_ROOT / "groundcua")
    if base is None:
        print("  !! groundcua raw not found")
        return
    by_image: dict[str, list[dict]] = defaultdict(list)
    files = sorted(base.rglob("*.parquet")) + sorted(base.rglob("*.json"))
    for f in files:
        rows = []
        if f.suffix == ".parquet":
            rows = pq.read_table(f).to_pylist()
        else:
            try:
                data = json.loads(f.read_text())
                rows = data if isinstance(data, list) else data.get("data", [])
            except Exception:
                continue
        for r in rows:
            if r.get("image_path") and r.get("bbox"):
                by_image[r["image_path"]].append(r)
    print(f"  groundcua: {len(by_image)} screenshots")
    images = list(by_image)
    rng.shuffle(images)
    preferred = ("Button", "Menu", "Input")
    for image_path in images:
        elems = by_image[image_path]
        img_file = base / "images" / image_path if (base / "images").exists() else base / image_path
        dims = None
        if img_file.exists():
            with open(img_file, "rb") as fh:
                dims = png_jpeg_dims(fh.read(64 * 1024))
        if dims is None:
            continue
        w, h = dims
        good = []
        for e in elems:
            x1, y1, x2, y2 = e["bbox"]
            if not e.get("text") or not e["text"].strip() or (x2 - x1) * (y2 - y1) < w * h * 1e-4:
                continue
            good.append(e)
        if not good:
            continue
        good.sort(key=lambda e: (e.get("category") not in preferred, rng.random()))
        picked = good[:8]
        target = f"{MEDIA_PREFIX}/groundcua/{image_path}"
        manifest[target] = {"kind": "groundcua", "ref": image_path, "source": "groundcua"}
        messages = []
        for j, e in enumerate(picked):
            x1, y1, x2, y2 = e["bbox"]
            box = fmt_box(x1, y1, x2, y2, w, h)
            uparts = [image_part(target)] if j == 0 else []
            if rng.random() < 0.7:
                uparts.append(text_part(f"Locate the element <ref>{e['text'].strip()}</ref> in the screenshot."))
                messages.append({"role": "user", "content": uparts})
                messages.append({"role": "assistant", "content": [text_part(box)]})
            else:
                uparts.append(text_part(f"What is the text of the UI element at {box}?"))
                messages.append({"role": "user", "content": uparts})
                messages.append({"role": "assistant", "content": [text_part(e["text"].strip())]})
        yield (
            make_row(f"groundcua-{hashlib.sha1(image_path.encode()).hexdigest()[:16]}", "groundcua", messages),
            row_cost(messages, [est_image_tokens(w, h)]),
        )


ACTION_RE = re.compile(r"(\w+)\((.*?)\)", re.DOTALL)


def scalecua_action_to_tool_calls(value: str, idx: int) -> list[dict] | None:
    m = re.search(r"<action>\s*(.*?)\s*</action>", value, re.DOTALL)
    if not m:
        return None
    calls = []
    for name, args_s in ACTION_RE.findall(m.group(1)):
        kwargs = {}
        for kv in re.findall(r"(\w+)\s*=\s*([^,()]+)", args_s):
            k, v = kv
            v = v.strip().strip("'\"")
            try:
                v = float(v) if "." in v else int(v)
            except ValueError:
                pass
            kwargs[k] = v

        def scale(k):
            return int(round(float(kwargs[k]) * 1000)) if k in kwargs else None

        if name in ("click", "doubleClick", "rightClick", "left_double", "hover", "moveTo"):
            args = {"x": scale("x"), "y": scale("y")}
            if name in ("doubleClick", "left_double"):
                args["double"] = True
            if name == "rightClick":
                args["button"] = "right"
            calls.append(("computer_click", args))
        elif name in ("write", "type", "typewrite"):
            calls.append(("computer_type", {"text": str(kwargs.get("message", kwargs.get("text", "")))}))
        elif name in ("press", "hotkey", "keyDown"):
            keys = kwargs.get("keys") or kwargs.get("key") or "+".join(str(v) for v in kwargs.values())
            calls.append(("computer_key", {"keys": str(keys)}))
        elif name == "scroll":
            calls.append(
                (
                    "computer_scroll",
                    {
                        "x": scale("x") or 500,
                        "y": scale("y") or 500,
                        "dy": int(kwargs.get("clicks", kwargs.get("dy", -3)) * -1),
                    },
                )
            )
        elif name in ("dragTo", "drag"):
            calls.append(
                (
                    "computer_drag",
                    {
                        "x1": scale("from_x") or scale("x1") or 500,
                        "y1": scale("from_y") or scale("y1") or 500,
                        "x2": scale("x") or scale("x2") or 500,
                        "y2": scale("y") or scale("y2") or 500,
                    },
                )
            )
        elif name in ("wait", "sleep"):
            calls.append(("computer_wait", {}))
        else:
            return None  # unknown action: drop the row rather than mistranslate
    if not calls:
        return None
    return [
        {"id": f"call_{idx}_{j}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
        for j, (n, a) in enumerate(calls)
    ]


def iter_scalecua(manifest: dict, rng: random.Random) -> Iterator[tuple[dict, int]]:
    base = _first_existing(RAW_ROOT / "ScaleCUA-Data", RAW_ROOT / "scalecua")
    if base is None:
        print("  !! scalecua raw not found")
        return
    files = sorted(base.rglob("*.parquet")) + sorted(base.rglob("*.jsonl"))
    rng.shuffle(files)
    for f in files:
        rows: Iterator[dict]
        if f.suffix == ".parquet":
            rows = (r for batch in pq.ParquetFile(f).iter_batches(batch_size=64) for r in batch.to_pylist())
        else:
            rows = (json.loads(line) for line in open(f))
        for i, raw in enumerate(rows):
            convs, img = raw.get("conversations"), raw.get("image")
            if not convs or not img:
                continue
            target = f"{MEDIA_PREFIX}/scalecua/{img}"
            messages, ok = [], True
            for turn in convs:
                role = {"human": "user", "gpt": "assistant"}.get(turn["from"])
                value = turn["value"].replace("<image>", "").strip()
                if role == "user":
                    parts = ([image_part(target)] if not messages else []) + ([text_part(value)] if value else [])
                    messages.append({"role": "user", "content": parts})
                elif role == "assistant":
                    tcs = scalecua_action_to_tool_calls(turn["value"], i)
                    if tcs is None and "<action>" in turn["value"]:
                        ok = False
                        break
                    thought = re.sub(r"<action>.*?</action>", "", value, flags=re.DOTALL).strip()
                    msg = {"role": "assistant", "content": [text_part(thought)] if thought else []}
                    if tcs:
                        msg["tool_calls"] = tcs
                    messages.append(msg)
                else:
                    ok = False
                    break
            if not ok or not messages or messages[-1]["role"] != "assistant":
                continue
            manifest[target] = {"kind": "scalecua", "ref": img, "source": "scalecua"}
            w, h = raw.get("width"), raw.get("height")
            yield (
                make_row(f"scalecua-{f.stem}-{i}", "scalecua", messages, tools=COMPUTER_TOOLS_JSON),
                row_cost(messages, [est_image_tokens(w, h)]),
            )


def iter_pixmo_points(rng: random.Random) -> Iterator[tuple[dict, int]]:
    base = _first_existing(RAW_ROOT / "pixmo-points", RAW_ROOT / "pixmo_points")
    img_root = RAW_ROOT / "pixmo_images"
    if base is None:
        print("  !! pixmo-points raw not found")
        return
    for f in sorted(base.rglob("*.parquet")):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=256):
            for i, raw in enumerate(batch.to_pylist()):
                url, label, points = raw.get("image_url"), raw.get("label"), raw.get("points") or []
                if not url or not label or not points or len(points) > 40:
                    continue
                sha = hashlib.sha1(url.encode()).hexdigest()
                img_file = next(iter(img_root.glob(f"{sha}.*")), None)
                if img_file is None:
                    continue
                target = f"{MEDIA_PREFIX}/pixmo/{img_file.name}"
                dst = MEDIA_ROOT / "pixmo" / img_file.name
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    os.link(img_file, dst)
                # pixmo points are percentages of the image dimensions
                pts = fmt_points([(p["x"] * 10, p["y"] * 10) for p in points], 1000, 1000)
                if rng.random() < 0.5:
                    q = f"Point to every instance of: {label}."
                    a = pts
                else:
                    q = f"How many {label} are in the image? Point to each one."
                    a = f"There are {len(points)}. {pts}"
                messages = [
                    {"role": "user", "content": [image_part(target), text_part(q)]},
                    {"role": "assistant", "content": [text_part(a)]},
                ]
                yield make_row(f"pixmopts-{sha[:16]}-{i}", "pixmo_points", messages), row_cost(messages, [1000])


def iter_pixmo_cap(rng: random.Random) -> Iterator[tuple[dict, int]]:
    base = _first_existing(RAW_ROOT / "pixmo-cap", RAW_ROOT / "pixmo_cap")
    img_root = RAW_ROOT / "pixmo_images"
    if base is None:
        print("  !! pixmo-cap raw not found")
        return
    prompts = [
        "Describe this image in detail.",
        "Write a dense, thorough caption for this image.",
        "What is shown in this image? Describe everything you can see.",
    ]
    for f in sorted(base.rglob("*.parquet")):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=256):
            for i, raw in enumerate(batch.to_pylist()):
                url, caption = raw.get("image_url"), raw.get("caption")
                if not url or not caption:
                    continue
                sha = hashlib.sha1(url.encode()).hexdigest()
                img_file = next(iter(img_root.glob(f"{sha}.*")), None)
                if img_file is None:
                    continue
                target = f"{MEDIA_PREFIX}/pixmo/{img_file.name}"
                dst = MEDIA_ROOT / "pixmo" / img_file.name
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    os.link(img_file, dst)
                messages = [
                    {"role": "user", "content": [image_part(target), text_part(rng.choice(prompts))]},
                    {"role": "assistant", "content": [text_part(caption)]},
                ]
                yield make_row(f"pixmocap-{sha[:16]}", "pixmo_cap", messages), row_cost(messages, [1000])


def iter_vwi(manifest: dict, rng: random.Random) -> Iterator[tuple[dict, int]]:
    base = _first_existing(RAW_ROOT / "VisualWebInstruct", RAW_ROOT / "visualwebinstruct")
    if base is None:
        print("  !! visualwebinstruct raw not found")
        return
    for f in sorted(base.rglob("*.parquet")):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=256):
            for i, raw in enumerate(batch.to_pylist()):
                images, q, a = raw.get("image") or [], raw.get("question"), raw.get("answer")
                if not images or not q or not a or len(images) > 8:
                    continue
                q = re.sub(r"\s*Please conclude your answer as Answer: xxx at the end if possible\.?\s*", "", q)
                parts = []
                for img in images:
                    target = f"{MEDIA_PREFIX}/vwi/{img}"
                    manifest[target] = {"kind": "vwi", "ref": img, "source": "visualwebinstruct"}
                    parts.append(image_part(target))
                parts.append(text_part(q))
                messages = [{"role": "user", "content": parts}, {"role": "assistant", "content": [text_part(a)]}]
                yield (
                    make_row(f"vwi-{raw.get('idx', i)}", "visualwebinstruct", messages),
                    row_cost(messages, [900] * len(images)),
                )


def iter_screenqa(rng: random.Random) -> Iterator[tuple[dict, int]]:
    """google-research ScreenQA short answers over RICO screenshots: grouped
    into multi-QA conversations per screen (screen-state QA share of the GUI slice)."""
    base = _first_existing(RAW_ROOT / "screen_qa")
    rico = _first_existing(RAW_ROOT / "rico" / "combined")
    if base is None or rico is None:
        print("  !! screenqa/rico raw not found")
        return
    by_image: dict[int, list[dict]] = defaultdict(list)
    for split in ("train.json", "validation.json"):
        f = base / "short_answers" / split
        if f.exists():
            for r in json.loads(f.read_text()):
                if r.get("ground_truth"):
                    by_image[r["image_id"]].append(r)
    images = list(by_image)
    rng.shuffle(images)
    store_dir = MEDIA_ROOT / "screenqa"
    store_dir.mkdir(parents=True, exist_ok=True)
    for image_id in images:
        src = rico / f"{image_id}.jpg"
        if not src.exists():
            continue
        dst = store_dir / f"{image_id}.jpg"
        if not dst.exists():
            os.link(src, dst)
        target = f"{MEDIA_PREFIX}/screenqa/{image_id}.jpg"
        qas = by_image[image_id][:6]
        messages = []
        for j, qa in enumerate(qas):
            uparts = [image_part(target)] if j == 0 else []
            uparts.append(text_part(qa["question"]))
            messages.append({"role": "user", "content": uparts})
            messages.append({"role": "assistant", "content": [text_part(qa["ground_truth"][0])]})
        yield make_row(f"screenqa-{image_id}", "screenqa", messages), row_cost(messages, [1300])


def iter_text_replay(rng: random.Random) -> Iterator[tuple[dict, int]]:
    """Nemotron-Post-Training-v3: text SFT rows; tool/agentic subsets oversampled 2x
    by yielding them twice (interleave shuffles)."""
    base = _first_existing(RAW_ROOT / "text_replay", RAW_ROOT / "Nemotron-Post-Training-Dataset-v3", RAW_ROOT / "nptv3")
    if base is None:
        print("  !! nptv3 raw not found")
        return
    files = sorted(base.rglob("*.parquet")) + sorted(base.rglob("*.jsonl"))
    rng.shuffle(files)
    for f in files:
        rel = str(f.relative_to(base)).lower()
        weight = 2 if ("tool" in rel or "agent" in rel) else 1
        rows: Iterator[dict]
        if f.suffix == ".parquet":
            rows = (r for batch in pq.ParquetFile(f).iter_batches(batch_size=128) for r in batch.to_pylist())
        else:
            rows = (json.loads(line) for line in open(f))
        for i, raw in enumerate(rows):
            msgs_raw = raw.get("messages")
            if msgs_raw is None and raw.get("input") is not None and raw.get("output") is not None:
                inp = raw["input"]
                msgs_raw = (inp if isinstance(inp, list) else [{"role": "user", "content": str(inp)}]) + [
                    {"role": "assistant", "content": raw["output"]}
                ]
            if not msgs_raw:
                continue
            messages = []
            ok = True
            for m in msgs_raw:
                role, content = m.get("role"), m.get("content")
                if role not in ("system", "user", "assistant", "tool") or not isinstance(content, str):
                    ok = False
                    break
                messages.append({"role": role, "content": [text_part(content)]})
            if not ok or not messages or messages[-1]["role"] != "assistant":
                continue
            row = make_row(f"nptv3-{f.stem}-{i}", f"nptv3_{f.parts[-2] if len(f.parts) > 1 else 'root'}", messages)
            cost = row_cost(messages, [])
            if cost > MAX_ROW_TOKENS:
                continue
            for _ in range(weight):
                yield row, cost


# ---------------------------------------------------------------------------
# Slice specs
# ---------------------------------------------------------------------------


@dataclass
class SourceSpec:
    name: str
    it: Callable[..., Iterator[tuple[dict, int]]]
    budget: int
    kwargs: dict = field(default_factory=dict)


@dataclass
class SliceSpec:
    name: str
    sources: list[SourceSpec]


def build_slices(manifest: dict, rng: random.Random) -> dict[str, SliceSpec]:
    B = 1_000_000
    itv3 = lambda src, keep=False: (lambda m=manifest, r=rng, s=src, k=keep: iter_itv3(s, k, m, r))
    slices = {
        "gui_v31": SliceSpec(
            "gui_v31",
            [
                SourceSpec("groundcua", lambda: iter_groundcua(manifest, rng), 500 * B),
                SourceSpec("scalecua", lambda: iter_scalecua(manifest, rng), 480 * B),
                SourceSpec("visualwebinstruct", lambda: iter_vwi(manifest, rng), 240 * B),
                SourceSpec("screenqa", lambda: iter_screenqa(rng), 240 * B),
                # agentnet appended once its extracted trajectory layout is confirmed
            ],
        ),
        "docs_ocr_v31": SliceSpec(
            "docs_ocr_v31",
            [
                *[
                    SourceSpec(s, itv3(s), b * B)
                    for s, b in [
                        ("docvqa", 120),
                        ("textcaps", 60),
                        ("textcaps_commercial", 60),
                        ("textvqa_commercial", 60),
                        ("hiertext", 80),
                        ("pubtables_1m", 120),
                        ("fintabnet", 100),
                        ("sa_finance_1", 40),
                        ("sa_finance_2", 40),
                        ("sa_finance_3", 40),
                        ("sa_finance_4", 40),
                        ("sa_finance_5", 40),
                        ("cc3m", 100),
                    ]
                ],
                *[
                    SourceSpec(f"long_document_ccpdf_{i:02d}", itv3(f"long_document_ccpdf_{i:02d}"), 46 * B)
                    for i in range(1, 12)
                ],
                *[SourceSpec(f"long_document_sec_{i}", itv3(f"long_document_sec_{i}"), 20 * B) for i in range(1, 5)],
                # long_document_arxiv_1-3 dropped: media needs arXiv bulk S3 (requester
                # pays) + local PDF rendering; budget folded into ccpdf.
                SourceSpec(
                    "docmatix", lambda: iter_embedded_qa("docmatix", RAW_ROOT / "Docmatix", "docmatix", rng), 400 * B
                ),
            ],
        ),
        "charts_v31": SliceSpec(
            "charts_v31",
            [
                *[
                    SourceSpec(s, itv3(s), b * B)
                    for s, b in [
                        ("chartqa_1", 150),
                        ("figureqa", 120),
                        ("plotqa_1", 120),
                        ("plotqa_2", 60),
                        ("mapqa", 100),
                        ("ecd", 100),
                    ]
                ],
                SourceSpec(
                    "finevision_charts",
                    lambda: iter_finevision_group(
                        ["chartqa", "chart2text", "Unichart", "SynthChartNet", "CoSyn_400k_chart"], rng
                    ),
                    200 * B,
                ),
            ],
        ),
        "reasoning_v31": SliceSpec(
            "reasoning_v31",
            [
                *[
                    SourceSpec(s, itv3(s, keep=True), b * B)
                    for s, b in [
                        ("mulberry_1", 200),
                        ("mulberry_2", 100),
                        ("aokvqa_1", 120),
                        ("aokvqa_2", 60),
                        ("clevr_1", 100),
                        ("geometry3k", 80),
                        ("geomverse", 80),
                        ("unigeo", 60),
                        ("raven", 60),
                    ]
                ],
                SourceSpec(
                    "llava_cot",
                    lambda: iter_sharegpt(
                        "llava_cot",
                        _first_existing(RAW_ROOT / "LLaVA-CoT-100k", RAW_ROOT / "llava_cot") or RAW_ROOT / "llava_cot",
                        "llavacot",
                        manifest,
                        rng,
                        transform=llava_cot_transform,
                    ),
                    340 * B,
                ),
                SourceSpec(
                    "mathv360k",
                    lambda: iter_sharegpt(
                        "mathv360k",
                        _first_existing(RAW_ROOT / "MathV360K", RAW_ROOT / "mathv360k") or RAW_ROOT / "mathv360k",
                        "mathv",
                        manifest,
                        rng,
                        transform=boxed_transform,
                    ),
                    300 * B,
                ),
                # vision-r1-cold + mmk12 appended by --extend after repo resolution
            ],
        ),
        "perception_v31": SliceSpec(
            "perception_v31",
            [
                SourceSpec("pixmo_cap", lambda: iter_pixmo_cap(rng), 350 * B),
                SourceSpec("pixmo_points", lambda: iter_pixmo_points(rng), 150 * B),
                *[
                    SourceSpec(s, itv3(s), b * B)
                    for s, b in [
                        ("flickr30k", 80),
                        ("openimages_5", 100),
                        ("toloka", 70),
                        ("openimages_1", 65),
                        ("openimages_2", 65),
                        ("openimages_3", 60),
                        ("openimages_4", 60),
                    ]
                ],
                # localized narratives appended by --extend
            ],
        ),
        "general_v31": SliceSpec(
            "general_v31",
            [
                SourceSpec(
                    "finevision_general",
                    lambda: iter_finevision_group(
                        ["ai2d_merged", "a_okvqa", "cocoqa", "LLaVA_Instruct_150K", "cambrian(filtered)_processed"],
                        rng,
                        quality_min=4,
                    ),
                    300 * B,
                ),
                SourceSpec(
                    "honey",
                    lambda: iter_embedded_qa("honey", RAW_ROOT / "Honey-Data-1M", "honey", rng, strip_think=True),
                    300 * B,
                ),
            ],
        ),
        "text_replay_v31": SliceSpec(
            "text_replay_v31",
            [
                SourceSpec("nptv3", lambda: iter_text_replay(rng), 2000 * B),
            ],
        ),
    }
    return slices


def iter_finevision_group(configs: list[str], rng: random.Random, quality_min: int = 4):
    base = RAW_ROOT / "FineVision"
    for cfg in configs:
        cfg_dir = next(iter(base.glob(f"**/{cfg}")), None)
        if cfg_dir is None:
            print(f"  !! finevision/{cfg} not downloaded, skipping")
            continue
        yield from iter_embedded_qa(f"finevision_{cfg}", cfg_dir, "finevision", rng, quality_min=quality_min)


MESSAGE_TYPE = pa.struct(
    [
        ("role", pa.string()),
        ("content", pa.list_(pa.struct([("type", pa.string()), ("text", pa.string()), ("image", pa.string())]))),
        (
            "tool_calls",
            pa.list_(
                pa.struct(
                    [
                        ("id", pa.string()),
                        ("type", pa.string()),
                        ("function", pa.struct([("name", pa.string()), ("arguments", pa.string())])),
                    ]
                )
            ),
        ),
    ]
)
SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("source", pa.string()),
        ("tools", pa.string()),
        ("messages", pa.list_(MESSAGE_TYPE)),
    ]
)


def build_slice(spec: SliceSpec, manifest: dict) -> None:
    print(f"== {spec.name} ==")
    rows: list[dict] = []
    total = 0
    for src in spec.sources:
        got, n0 = 0, len(rows)
        try:
            for row, cost in src.it():
                if cost > MAX_ROW_TOKENS:
                    continue
                rows.append(row)
                got += cost
                if got >= src.budget:
                    break
        except Exception as e:
            print(f"  !! {src.name} failed: {type(e).__name__}: {e}")
        total += got
        print(f"  {src.name}: {got / 1e6:.0f}M/{src.budget / 1e6:.0f}M tokens, {len(rows) - n0} rows")
    rng = random.Random(SEED + hash(spec.name) % 1000)
    rng.shuffle(rows)
    n_val = min(600, max(50, len(rows) // 200)) if rows else 0
    out = OUT_DIR / "data" / spec.name
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        pq.write_table(pa.Table.from_pylist(rows[:n_val], schema=SCHEMA), out / "validation.parquet")
        pq.write_table(pa.Table.from_pylist(rows[n_val:], schema=SCHEMA), out / "train.parquet")
    print(f"  -> {len(rows) - n_val} train / {n_val} val rows, {total / 1e9:.2f}B tokens")


def finalize() -> None:
    """Copy the browser subset in, write the README yaml configs, and compute
    interleave probabilities: p_i ∝ epochs_i * train_rows_i (browser epochs=2,
    everything else 1) so all subsets finish their pass together."""
    import shutil

    browser_out = OUT_DIR / "data" / "browser_use_sft_dataset"
    if not browser_out.exists() and BROWSER_SRC.exists():
        shutil.copytree(BROWSER_SRC, browser_out)
        print(f"browser subset copied from {BROWSER_SRC}")

    subsets = [
        "browser_use_sft_dataset",
        "gui_v31",
        "docs_ocr_v31",
        "charts_v31",
        "reasoning_v31",
        "perception_v31",
        "general_v31",
        "text_replay_v31",
    ]
    counts, missing = {}, []
    for s in subsets:
        f = OUT_DIR / "data" / s / "train.parquet"
        if not f.exists():
            missing.append(s)
            continue
        counts[s] = pq.ParquetFile(f).metadata.num_rows
    if missing:
        print(f"!! finalize with missing subsets: {missing}")
    weights = {s: (2 if s == "browser_use_sft_dataset" else 1) * n for s, n in counts.items()}
    total_w = sum(weights.values())
    probs = {s: w / total_w for s, w in weights.items()}

    lines = ["---", "configs:"]
    for s in counts:
        lines += [
            f"- config_name: {s}",
            "  data_files:",
            f"  - split: train\n    path: data/{s}/train.parquet",
            f"  - split: validation\n    path: data/{s}/validation.parquet",
        ]
    lines += [
        "---",
        "",
        "# nemotron_vl_sft_v31 (Blend v3.1)",
        "",
        "Interleave probabilities (p ∝ epochs × rows; browser at 2 epochs):",
        "```toml",
        "subsets = [" + ", ".join(f'"{s}"' for s in counts) + "]",
        "probabilities = [" + ", ".join(f"{probs[s]:.6f}" for s in counts) + "]",
        "```",
        "",
    ]
    for s, n in counts.items():
        lines.append(f"- {s}: {n} train rows (p={probs[s]:.4f})")
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-len(counts) :]))
    print("README.md written; paste the toml block into phase2_v31.toml")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slices", nargs="+", default=None)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.finalize:
        finalize()
        return
    rng = random.Random(SEED)
    manifest: dict = {}
    (OUT_DIR / "data").mkdir(parents=True, exist_ok=True)
    MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
    slices = build_slices(manifest, rng)
    for name, spec in slices.items():
        if args.slices and name not in args.slices:
            continue
        build_slice(spec, manifest)
        mpath = OUT_DIR / f"media_manifest_{name}.json"
        mpath.write_text(json.dumps({k: v for k, v in manifest.items()}))
        manifest.clear()
    print("DONE")


if __name__ == "__main__":
    main()
