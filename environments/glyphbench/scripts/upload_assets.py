#!/usr/bin/env python
"""Upload generated assets (GIFs, etc.) to the GlyphBench HF dataset repo.

The README and per-suite previews are served from this dataset, so this is the
tool to refresh them after regenerating GIFs with `record_random_gifs.py`.

Usage:
    uv run --extra assets python scripts/upload_assets.py --repo owner/name
    uv run --extra assets python scripts/upload_assets.py --repo owner/name --replace
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo  # type: ignore[import-not-found]


def _load_local_env(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in {"HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "GLYPHBENCH_ASSET_REPO"}:
            os.environ.setdefault(key, value.strip().strip("'\""))


def main() -> None:
    _load_local_env()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        default=os.getenv("GLYPHBENCH_ASSET_REPO"),
        help="HF dataset repo id (owner/name); defaults to GLYPHBENCH_ASSET_REPO",
    )
    parser.add_argument("--src", type=Path, default=Path("gifs"), help="Local dir to upload")
    parser.add_argument("--dst", default="gifs", help="Destination path within the dataset repo")
    parser.add_argument("--private", action="store_true", help="Create repo as private")
    parser.add_argument("--commit-message", default="Update GlyphBench assets")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace the destination folder in the same commit as the upload.",
    )
    args = parser.parse_args()

    if not args.repo:
        parser.error("--repo or GLYPHBENCH_ASSET_REPO is required")
    if not args.src.is_dir():
        raise SystemExit(f"Source dir not found: {args.src}")
    files = [p for p in args.src.rglob("*") if p.is_file()]
    if not files:
        raise SystemExit(f"Source dir is empty: {args.src}")

    create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)

    api = HfApi()
    print(f"Uploading {len(files)} files from {args.src} -> {args.repo}/{args.dst}")
    api.upload_folder(
        folder_path=str(args.src),
        path_in_repo=args.dst,
        repo_id=args.repo,
        repo_type="dataset",
        commit_message=args.commit_message,
        delete_patterns="*" if args.replace else None,
    )
    print(
        f"\nDone. Public URL pattern:\n"
        f"  https://huggingface.co/datasets/{args.repo}/resolve/main/{args.dst}/<filename>"
    )


if __name__ == "__main__":
    main()
