"""Validation battery for NemotronVLRenderer.

A. Text-only byte parity: NemotronVLRenderer == Nemotron3Renderer == HF
   apply_chat_template across scenarios (system/no-system, multi-turn,
   generation prompt with thinking on/off).
B. Pixel parity vs NVIDIA's Omni image processor (the reference file shipped in
   the graft checkpoint): identical patch grids and BIT-IDENTICAL pixel values
   (after unpatchifying our Im2Patches layout) on real dataset images from
   every subset.
C. Structure + loss-mask invariants over real dataset samples (every subset):
   each <image> run bracketed by <img>/</img>; placeholder ranges == actual
   runs; token counts == grid/4 == pixel rows/4; mm_items aligned 1:1 with
   image parts in order; loss lands exactly on assistant bodies (starting
   <think></think>, ending <|im_end|>), never on image/user/system tokens.
D. Determinism: two renders of the same sample are identical.

Run from the prime-rl repo root:
    uv run python models/validation/check_renderer.py [--per-subset 30]
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from renderers.base import build_training_sample, create_renderer
from renderers.configs import Nemotron3RendererConfig, NemotronVLRendererConfig
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parent.parent.parent
GRAFT = REPO / "models" / "Nemotron-3-Super-VL-graft"
DATA = REPO / "datasets" / "nemotron_vl_sft" / "data"
SUBSETS = [
    "openimages_captions",
    "cc3m_captions_ocr",
    "openimages_grounding",
    "scenetext_qa",
    "longdoc_multiimage",
    "descriptive_long",
    "chart_doc_qa",
]
IMG, IMG_START, IMG_END = 18, 19, 20


def load_reference_processor():
    spec = importlib.util.spec_from_file_location("omni_ip", GRAFT / "omni_image_processing_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = json.loads((GRAFT / "preprocessor_config.json").read_text())
    return getattr(module, cfg["image_processor_type"])(
        **{k: v for k, v in cfg.items() if k not in ("image_processor_type", "auto_map")}
    )


def unpatchify(patches: np.ndarray, th: int, tw: int, p: int = 16) -> torch.Tensor:
    """Invert Im2Patches: (th*tw, 3*p*p) -> (3, th*p, tw*p)."""
    t = torch.from_numpy(patches).reshape(th, tw, 3, p, p)
    return t.permute(2, 0, 3, 1, 4).reshape(3, th * p, tw * p)


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


def check_text_parity(tok) -> None:
    base = create_renderer(tok, Nemotron3RendererConfig())
    vl = create_renderer(tok, NemotronVLRendererConfig())
    scenarios = [
        ([{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}], False),
        (
            [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "Name a prime."},
                {"role": "assistant", "content": "7"},
            ],
            False,
        ),
        ([{"role": "user", "content": "Go"}], True),
        ([{"role": "user", "content": [{"type": "text", "text": "list content"}]}], True),
    ]
    for messages, gen in scenarios:
        a = base.render(messages, add_generation_prompt=gen).token_ids
        b = vl.render(messages, add_generation_prompt=gen).token_ids
        assert a == b, f"text-only parity broken for {messages}"
        if any(isinstance(m.get("content"), list) for m in messages):
            continue  # the Jinja template stringifies content-part lists; renderers flatten them
        hf = tok.apply_chat_template(messages, add_generation_prompt=gen, tokenize=True)
        if not isinstance(hf, list):  # transformers >=5 returns BatchEncoding
            hf = hf["input_ids"]
        assert b == hf, (
            f"renderer != apply_chat_template for {messages}:\n{tok.convert_ids_to_tokens(b)}\nvs\n{tok.convert_ids_to_tokens(hf)}"
        )
    print(f"A. text-only parity (renderer == base == chat template): {len(scenarios)} scenarios OK")


def check_pixel_parity(vl, samples_by_subset) -> None:
    from PIL import Image

    ref_proc = load_reference_processor()
    checked = 0
    for subset, samples in samples_by_subset.items():
        images = []
        for row in samples:
            for msg in row["messages"]:
                for part in msg["content"]:
                    if part["type"] == "image":
                        images.append(part["image"])
        for path in images[:3]:
            pil = Image.open(REPO / path).convert("RGB")
            ref = ref_proc._preprocess([ref_proc._process_image(pil)])
            ref_pv = ref["pixel_values"]
            ref_pv = ref_pv[0] if isinstance(ref_pv, torch.Tensor) else ref_pv[0]
            (ref_tokens,) = ref["num_tokens"]

            patches, th, tw = vl._preprocess_image(pil)
            assert (th * 16, tw * 16) == tuple(ref_pv.shape[-2:]), (
                f"{path}: grid ({th},{tw}) vs reference {tuple(ref_pv.shape[-2:])}"
            )
            assert (th * tw) // 4 == ref_tokens, f"{path}: tokens {(th * tw) // 4} vs reference {ref_tokens}"
            ours = unpatchify(patches, th, tw)
            assert torch.equal(ours, ref_pv), f"{path}: pixel values differ (max|d|={(ours - ref_pv).abs().max():.3e})"
            checked += 1
    print(f"B. pixel parity vs Omni reference processor: {checked} images bit-identical")


def check_structure_and_mask(tok, vl, samples_by_subset) -> None:
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    n_samples = n_images = 0
    for subset, samples in samples_by_subset.items():
        for row in samples:
            s = build_training_sample(vl, row["messages"])
            ids, mask = list(s.token_ids), list(s.loss_mask)
            mm = s.multi_modal_data
            image_parts = [p for m in row["messages"] for p in m["content"] if p["type"] == "image"]

            runs = image_runs(ids)
            items = mm.mm_items.get("image", []) if mm else []
            phs = mm.mm_placeholders.get("image", []) if mm else []
            assert len(runs) == len(items) == len(phs) == len(image_parts), (
                subset,
                len(runs),
                len(items),
                len(phs),
                len(image_parts),
            )
            for (start, length), ph, item in zip(runs, phs, items):
                assert (ph.offset, ph.length) == (start, length), (subset, ph, start, length)
                assert ids[start - 1] == IMG_START and ids[start + length] == IMG_END, f"{subset}: unbracketed run"
                th, tw = (int(x) for x in item["image_grids"][0])
                assert length == (th * tw) // 4 == item["pixel_values"].shape[0] // 4, (subset, length, th, tw)
                assert item["pixel_values"].shape == (th * tw, 768)
                assert not any(mask[start : start + length]), f"{subset}: loss on image tokens"

            # Loss must land exactly on assistant bodies: starts with <think></think>
            # (thinking-off SFT target), ends with <|im_end|>, non-empty per assistant turn.
            n_assistant = sum(1 for m in row["messages"] if m["role"] == "assistant")
            sup_ids = [t for t, m in zip(ids, mask) if m]
            assert sup_ids, f"{subset}: empty supervision"
            assert sup_ids.count(im_end) == n_assistant, (subset, sup_ids.count(im_end), n_assistant)
            text = tok.decode(sup_ids)
            assert text.count("<think></think>") == n_assistant, f"{subset}: think-off prefix missing: {text[:80]}"
            expected_answers = [
                "".join(p["text"] for p in m["content"] if p["type"] == "text")
                for m in row["messages"]
                if m["role"] == "assistant"
            ]
            rebuilt = "".join(f"<think></think>{a.strip()}<|im_end|>" for a in expected_answers)
            assert text == rebuilt, f"{subset}: supervised text mismatch:\n{text[:200]}\nvs\n{rebuilt[:200]}"

            # Determinism.
            s2 = build_training_sample(vl, row["messages"])
            assert list(s2.token_ids) == ids and list(s2.loss_mask) == mask

            n_samples += 1
            n_images += len(image_parts)
    print(
        f"C+D. structure/mask/determinism: {n_samples} samples, {n_images} images across {len(samples_by_subset)} subsets OK"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-subset", type=int, default=30)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(str(GRAFT))
    vl = create_renderer(tok, NemotronVLRendererConfig(image_cache_max=1))

    samples_by_subset = {}
    for subset in SUBSETS:
        t = pq.read_table(DATA / subset / "validation.parquet", columns=["messages"])
        samples_by_subset[subset] = t.slice(0, args.per_subset).to_pylist()

    check_text_parity(tok)
    check_pixel_parity(vl, samples_by_subset)
    check_structure_and_mask(tok, vl, samples_by_subset)
    print("PASS")


if __name__ == "__main__":
    main()
