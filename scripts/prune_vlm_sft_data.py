"""Drop blend rows whose media files are missing on disk.

Some referenced media are unrecoverable (images removed from the OpenImages
bucket, PDFs gone from Digital Corpora, failed page renders). Run this after
all `fetch_vlm_sft_media.py` commands have finished; it rewrites each
train/validation parquet in place, keeping only rows whose image paths all
exist, and prints per-subset counts.

Run from the repo root:
    uv run python scripts/prune_vlm_sft_data.py
"""

from pathlib import Path

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "datasets" / "nemotron_vl_sft" / "data"


def main():
    total_kept = total_dropped = 0
    for parquet_path in sorted(DATA_DIR.glob("*/*.parquet")):
        table = pq.read_table(parquet_path)
        rows = table.to_pylist()
        kept = []
        for row in rows:
            paths = [p["image"] for m in row["messages"] for p in m["content"] if p["type"] == "image"]
            if all((REPO_ROOT / p).exists() for p in paths):
                kept.append(row)
        dropped = len(rows) - len(kept)
        total_kept += len(kept)
        total_dropped += dropped
        if dropped:
            import pyarrow as pa

            pq.write_table(pa.Table.from_pylist(kept, schema=table.schema), parquet_path)
        rel = parquet_path.relative_to(DATA_DIR)
        print(f"{str(rel):45s} kept={len(kept):6d} dropped={dropped:5d}")
    print(f"\nTOTAL kept={total_kept} dropped={total_dropped}")


if __name__ == "__main__":
    main()
