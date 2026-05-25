"""Articraft dataset construction.

Scans ``data/records/rec_*/`` directories and emits a HuggingFace ``Dataset``
compatible with verifiers.  Cadquery records (~31.6%) are filtered out because
RL training targets pure SDK geometry only.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from datasets import Dataset

logger = logging.getLogger(__name__)

_CADQUERY_IMPORT_RE = re.compile(r"\b(import cadquery|from cadquery)\b")


def _load_record_json(record_dir: Path) -> dict[str, Any] | None:
    """Load and return the record.json payload, or None on failure."""
    path = record_dir / "record.json"
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _resolve_prompt_text(record: dict[str, Any], record_dir: Path) -> str | None:
    """Extract prompt text — prefer the prompt.txt file, fall back to display."""
    artifacts = record.get("artifacts", {})
    prompt_txt_rel = artifacts.get("prompt_txt")
    if prompt_txt_rel:
        prompt_path = record_dir / prompt_txt_rel
        if prompt_path.is_file():
            text = prompt_path.read_text(encoding="utf-8").strip()
            if text:
                return text

    display = record.get("display", {})
    preview = display.get("prompt_preview", "").strip()
    return preview or None


def _has_cadquery(record: dict[str, Any], record_dir: Path) -> bool:
    """Check if the record's model.py imports cadquery."""
    artifacts = record.get("artifacts", {})
    model_py_rel = artifacts.get("model_py")
    if not model_py_rel:
        return False
    model_path = record_dir / model_py_rel
    if not model_path.is_file():
        return False
    try:
        code = model_path.read_text(encoding="utf-8")
        return bool(_CADQUERY_IMPORT_RE.search(code))
    except OSError:
        return False


def build_dataset(
    articraft_root: str | Path,
    *,
    split: str = "all",
    eval_holdout: int = 50,
    sdk_package: str = "sdk",
) -> Dataset:
    """Construct a HF Dataset from articraft record directories.

    Args:
        articraft_root: Path to the articraft repo root.
        split: ``"all"``, ``"train"`` (drops last *eval_holdout*), or ``"eval"``.
        eval_holdout: Number of trailing records to reserve for eval.
        sdk_package: SDK package name (default ``"sdk"``).

    Returns:
        ``datasets.Dataset`` with ``prompt`` (empty list), ``answer`` (empty str),
        and ``info`` dict containing ``record_id``, ``prompt_text``, etc.
    """
    if split not in {"all", "train", "eval"}:
        raise ValueError(f"split must be 'all', 'train', or 'eval'; got {split!r}")

    root = Path(articraft_root).expanduser().resolve()
    records_dir = root / "data" / "records"
    if not records_dir.is_dir():
        raise FileNotFoundError(f"records directory not found: {records_dir}")

    record_dirs = sorted(
        [d for d in records_dir.iterdir() if d.is_dir() and d.name.startswith("rec_")],
        key=lambda d: d.name,
    )
    if not record_dirs:
        raise RuntimeError(f"no rec_* directories under {records_dir}")

    rows: list[dict[str, Any]] = []
    skipped_cadquery = 0
    skipped_no_prompt = 0

    for record_dir in record_dirs:
        record = _load_record_json(record_dir)
        if record is None:
            continue

        if _has_cadquery(record, record_dir):
            skipped_cadquery += 1
            continue

        prompt_text = _resolve_prompt_text(record, record_dir)
        if not prompt_text:
            skipped_no_prompt += 1
            continue

        record_id = record.get("record_id", record_dir.name)
        rows.append(
            {
                "prompt": [],
                "answer": "",
                "info": {
                    "record_id": record_id,
                    "prompt_text": prompt_text,
                    "category_slug": record.get("category_slug"),
                    "sdk_package": record.get("sdk_package", sdk_package),
                },
            }
        )

    logger.info(
        "dataset: %d records loaded, %d cadquery skipped, %d no-prompt skipped",
        len(rows),
        skipped_cadquery,
        skipped_no_prompt,
    )

    if not rows:
        raise RuntimeError(
            f"no valid records after filtering under {records_dir} "
            f"(cadquery={skipped_cadquery}, no_prompt={skipped_no_prompt})"
        )

    if split == "train":
        rows = rows[: max(0, len(rows) - eval_holdout)]
    elif split == "eval":
        rows = rows[max(0, len(rows) - eval_holdout) :]

    if not rows:
        raise RuntimeError(f"no records for split={split!r} (eval_holdout={eval_holdout})")

    return Dataset.from_list(rows)
