"""v3.1 image decontamination + cross-subset dedup (pHash).

1. Hash every image in the eval stack (--eval-roots) and the browser validation
   screenshots: these hashes are BANNED from training rows.
2. Hash every image referenced by the v3.1 train parquets (multiprocess).
3. Rewrite each train parquet dropping (a) rows referencing a banned hash
   (eval contamination) and (b) rows whose images all duplicate an earlier
   subset's images (cross-subset dedup; subset priority = the --subsets order,
   e.g. reasoning before general keeps the better-annotated row).

pHash: 32x32 grayscale -> 2D DCT (numpy) -> top-left 8x8 sans DC -> median
threshold -> 64-bit hex. Exact-match policy (standard for decontamination;
near-dup tolerance can be added by banning all hashes within hamming<=2).

Run on a compute node:
    uv run python scripts/dedup_v31.py --apply
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

OUT_DIR = Path(os.environ.get("OUT_ROOT", "/shared/hubert/datasets/nemotron_vl_sft_v31"))
REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_ROOTS_DEFAULT = ["/shared/hubert/raw/eval_stack"]
BROWSER_VAL = Path("/shared/hubert/datasets/nemotron_vl_sft_phase2/data/browser_use_sft_dataset/validation.parquet")

_DCT = None


def _dct_matrix(n: int = 32) -> np.ndarray:
    k = np.arange(n)
    m = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
    m[0] /= np.sqrt(2)
    return m


def phash(path: str) -> str | None:
    global _DCT
    if _DCT is None:
        _DCT = _dct_matrix()
    try:
        with Image.open(path) as im:
            a = np.asarray(im.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    except Exception:
        return None
    dct = _DCT @ a @ _DCT.T
    block = dct[:8, :8].flatten()[1:]
    bits = block > np.median(block)
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def hash_many(paths: list[str]) -> list[tuple[str, str | None]]:
    return [(p, phash(p)) for p in paths]


def collect_images(root: Path) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return [str(p) for p in root.rglob("*") if p.suffix.lower() in exts]


def row_image_paths(row: dict) -> list[str]:
    out = []
    for m in row["messages"]:
        for p in m["content"]:
            if p.get("image"):
                out.append(p["image"])
    return out


def parallel_hash(paths: list[str], workers: int) -> dict[str, str]:
    chunks = [paths[i::workers] for i in range(workers)]
    hashes: dict[str, str] = {}
    with ProcessPoolExecutor(workers) as ex:
        for result in ex.map(hash_many, chunks):
            for p, h in result:
                if h is not None:
                    hashes[p] = h
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-roots", nargs="+", default=EVAL_ROOTS_DEFAULT)
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=[
            "browser_use_sft_dataset",
            "reasoning_v31",
            "gui_v31",
            "docs_ocr_v31",
            "charts_v31",
            "perception_v31",
            "general_v31",
        ],
    )
    parser.add_argument("--workers", type=int, default=max(8, (os.cpu_count() or 16) - 8))
    parser.add_argument("--apply", action="store_true", help="rewrite parquets (otherwise report only)")
    args = parser.parse_args()

    banned: set[str] = set()
    for root in args.eval_roots:
        paths = collect_images(Path(root))
        print(f"eval root {root}: {len(paths)} images")
        banned |= set(parallel_hash(paths, args.workers).values())
    if BROWSER_VAL.exists():
        val_imgs = []
        for row in pq.read_table(BROWSER_VAL).to_pylist():
            val_imgs += row_image_paths(row)
        val_paths = [str(REPO_ROOT / p) for p in set(val_imgs) if (REPO_ROOT / p).exists()]
        print(f"browser val: {len(val_paths)} images")
        banned |= set(parallel_hash(val_paths, args.workers).values())
    print(f"banned hashes: {len(banned)}")

    seen_cross: dict[str, str] = {}  # hash -> first subset that claimed it
    report = {}
    for subset in args.subsets:
        f = OUT_DIR / "data" / subset / "train.parquet"
        if not f.exists():
            print(f"!! {subset}: no train.parquet, skipping")
            continue
        table = pq.read_table(f)
        rows = table.to_pylist()
        img_paths = sorted({p for row in rows for p in row_image_paths(row)})
        resolved = {p: str(REPO_ROOT / p) for p in img_paths if (REPO_ROOT / p).exists()}
        print(f"{subset}: {len(rows)} rows, {len(img_paths)} unique images ({len(resolved)} on disk)")
        hashes = parallel_hash(list(resolved.values()), args.workers)
        path_hash = {rel: hashes.get(abs_) for rel, abs_ in resolved.items()}

        kept, drop_eval, drop_dup = [], 0, 0
        for row in rows:
            hs = [path_hash.get(p) for p in row_image_paths(row)]
            hs = [h for h in hs if h]
            if any(h in banned for h in hs):
                drop_eval += 1
                continue
            if (
                subset != "browser_use_sft_dataset"
                and hs
                and all(h in seen_cross and seen_cross[h] != subset for h in hs)
            ):
                drop_dup += 1
                continue
            for h in hs:
                seen_cross.setdefault(h, subset)
            kept.append(row)
        report[subset] = {"rows": len(rows), "drop_eval": drop_eval, "drop_dup": drop_dup}
        print(f"  -> drop {drop_eval} contaminated / {drop_dup} cross-subset dups, keep {len(kept)}")
        if args.apply and (drop_eval or drop_dup):
            import pyarrow as pa

            pq.write_table(pa.Table.from_pylist(kept, schema=table.schema), f)
            print(f"  rewritten: {f}")

    (OUT_DIR / "dedup_report.json").write_text(json.dumps(report, indent=2))
    print("DONE", json.dumps(report))


if __name__ == "__main__":
    main()
