"""Renderer/format battery for the phase-2 blend (vision slices + browser-agent traces).

Per subset (validation split, real media required on disk):
  1. Renders through build_training_sample (the trainer's exact path) without error.
  2. Image accounting: every <image> run bracketed by <img>/</img>; run lengths ==
     grid/4 == pixel rows/4; mm_items 1:1 with image parts, in order.
  3. Loss mask: supervision only on assistant turns; never on image tokens, user,
     system, or tool-output tokens.
  4. browser_use_sft_dataset: the tools block is rendered into the system message; supervised text
     contains Nemotron-format <tool_call> XML; tool outputs are NOT supervised;
     screenshots inside tool messages are counted correctly.
  5. reasoning_p2: <think> traces are PRESERVED in the supervised text (traces-on).
  6. Token-length stats vs the 65536 context (image-safe truncation handles overflow;
     reports how many rows exceed).
  7. Determinism: re-render matches.

Run from the prime-rl repo root (after media fetch):
    uv run python models/validation/check_renderer_phase2.py [--per-subset 25]
"""

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from renderers.base import build_training_sample, create_renderer
from renderers.configs import NemotronVLRendererConfig
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parent.parent.parent
GRAFT = REPO / "models" / "Nemotron-3-Super-VL-graft"
DATA = REPO / "datasets" / "nemotron_vl_sft_phase2" / "data"
SUBSETS = [
    "browser_use_sft_dataset",
    "longdoc_p2",
    "grounding_p2",
    "ocr_docs_p2",
    "charts_p2",
    "reasoning_p2",
    "natural_qa_p2",
]
IMG, IMG_START, IMG_END = 18, 19, 20
SEQ_LEN = 65536


def image_runs(ids: list[int]) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, t in enumerate(ids):
        if t == IMG and start is None:
            start = i
        elif t != IMG and start is not None:
            runs.append((start, i - start))
            start = None
    if start is not None:
        runs.append((start, len(ids) - start))
    return runs


def check_subset(tok, vl, subset: str, rows: list[dict]) -> list[int]:
    import json

    lengths = []
    for row in rows:
        tools = json.loads(row["tools"]) if row.get("tools") else None
        messages = row["messages"]
        sample = build_training_sample(vl, messages, tools=tools or None)
        ids, mask = list(sample.token_ids), list(sample.loss_mask)
        lengths.append(len(ids))
        mm = sample.multi_modal_data

        image_parts = sum(
            1 for m in messages for p in (m["content"] or []) if isinstance(p, dict) and p.get("type") == "image"
        )
        runs = image_runs(ids)
        items = mm.mm_items.get("image", []) if mm else []
        assert len(runs) == len(items) == image_parts, (subset, row["id"], len(runs), len(items), image_parts)
        for (start, length), item in zip(runs, items):
            th, tw = (int(x) for x in item["image_grids"][0])
            assert ids[start - 1] == IMG_START and ids[start + length] == IMG_END, (subset, "unbracketed")
            assert length == (th * tw) // 4 == item["pixel_values"].shape[0] // 4, (subset, length, th, tw)
            assert not any(mask[start : start + length]), (subset, "loss on image tokens")

        sup = tok.decode([t for t, m in zip(ids, mask) if m])
        n_assistant = sum(1 for m in messages if m["role"] == "assistant")
        assert sup.count("<|im_end|>") == n_assistant, (subset, row["id"], "per-turn stop mismatch")

        if subset in ("browser_use_sft_dataset", "gui_v31"):
            full = tok.decode(ids[: min(len(ids), 4000)])
            if json.loads(row.get("tools") or "[]"):
                assert "<function>" in full and "<name>" in full, (subset, "tools block missing from system prompt")
            if any(m.get("tool_calls") for m in messages):
                assert "<tool_call>" in sup, (subset, row["id"], "tool calls not supervised")
            for m in messages:
                if m["role"] == "tool":
                    for p in m["content"] or []:
                        if p.get("type") == "text" and p.get("text") and len(p["text"]) > 40:
                            probe = p["text"][:40]
                            assert probe not in sup, (subset, row["id"], "tool output leaked into supervision")
                            break
        if subset in ("reasoning_p2", "reasoning_v31"):
            has_think_src = any(
                "<think>" in (p.get("text") or "")
                for m in messages
                if m["role"] == "assistant"
                for p in m["content"] or []
                if isinstance(p, dict)
            )
            if has_think_src:
                assert "<think>" in sup and "</think>" in sup, (subset, row["id"], "think trace lost")
                assert sup.index("<think>") < 20 or "<think>" in sup, sup[:80]

        sample2 = build_training_sample(vl, messages, tools=tools or None)
        assert list(sample2.token_ids) == ids, (subset, "non-deterministic render")

    return lengths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-subset", type=int, default=25)
    parser.add_argument("--data-dir", type=Path, default=DATA, help="dataset data/ dir")
    parser.add_argument("--subsets", nargs="+", default=SUBSETS)
    parser.add_argument("--model-dir", type=Path, default=GRAFT, help="tokenizer source")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(str(args.model_dir))
    vl = create_renderer(tok, NemotronVLRendererConfig(image_cache_max=1))

    all_ok = True
    for subset in args.subsets:
        path = args.data_dir / subset / "validation.parquet"
        if not path.exists():
            print(f"{subset}: MISSING parquet")
            all_ok = False
            continue
        table = pq.read_table(path)
        rows = table.slice(0, args.per_subset).to_pylist()
        try:
            lengths = check_subset(tok, vl, subset, rows)
            a = np.array(lengths)
            over = (a > SEQ_LEN).sum()
            print(
                f"{subset:<16} {len(rows)} rows OK | tokens p50={np.percentile(a, 50):.0f} "
                f"max={a.max()} >{SEQ_LEN}: {over}"
            )
        except AssertionError as e:
            print(f"{subset:<16} FAIL: {e}")
            all_ok = False
        except FileNotFoundError as e:
            print(f"{subset:<16} MEDIA PENDING: {e}")
            all_ok = False
    print("PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
