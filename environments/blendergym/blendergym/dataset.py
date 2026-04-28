"""BlenderGym dataset construction.

Scans ``data_root/<task_type><i>/`` directories produced by
``richard-guyunqi/BG_bench_data`` and emits a HuggingFace ``Dataset`` whose
``info`` column carries paths instead of base64 payloads — both to keep the
serialized dataset small and to make ``vf-eval --save-dataset`` outputs sane.

The actual b64 encoding (for VLM image messages) happens lazily during
``setup_state`` in :mod:`blendergym.env`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from datasets import Dataset


SUPPORTED_TASK_TYPES: tuple[str, ...] = (
    "placement",
    "blendshape",
    "geometry",
    "material",
    "lighting",
)

REQUIRED_FILES: tuple[tuple[str, ...], ...] = (
    ("blender_file.blend",),
    ("start.py",),
    ("renders", "start", "render1.png"),
    ("renders", "goal", "render1.png"),
)

_TASK_DIR_RE = re.compile(r"^(?P<type>[a-z]+)(?P<id>\d+)$")


def _iter_task_dirs(data_root: Path, task_type: str) -> list[tuple[int, Path]]:
    """Return ``[(numeric_id, dir_path)]`` for ``<task_type><N>`` dirs, sorted by id."""
    out: list[tuple[int, Path]] = []
    for entry in data_root.iterdir():
        if not entry.is_dir():
            continue
        m = _TASK_DIR_RE.match(entry.name)
        if not m or m.group("type") != task_type:
            continue
        out.append((int(m.group("id")), entry))
    out.sort(key=lambda pair: pair[0])
    return out


def _scan_task(task_dir: Path) -> dict[str, Any] | None:
    """Validate one task dir and return its row payload, or ``None`` on missing files."""
    missing = [
        "/".join(parts) for parts in REQUIRED_FILES if not (task_dir.joinpath(*parts)).exists()
    ]
    if missing:
        return None

    start_code_path = task_dir / "start.py"
    return {
        "task_dir": str(task_dir),
        "start_code": start_code_path.read_text(encoding="utf-8"),
        "init_image_path": str(task_dir / "renders" / "start" / "render1.png"),
        "goal_image_path": str(task_dir / "renders" / "goal" / "render1.png"),
        "blend_file_path": str(task_dir / "blender_file.blend"),
    }


def build_dataset(
    data_root: str | Path,
    task_types: Sequence[str] = ("placement",),
    *,
    split: str = "all",
    eval_holdout: int = 5,
) -> Dataset:
    """Construct a HF Dataset from ``data_root``.

    Args:
        data_root: directory containing ``<task_type><i>/`` subdirs (e.g. ``data/blendergym``).
        task_types: which task types to include. Phase-1/v1 only supports ``placement``.
        split: ``"all"``, ``"train"`` (drops the last ``eval_holdout`` ids per type), or
            ``"eval"`` (only the last ``eval_holdout`` ids per type). The ordering is by
            numeric task id, so the split is deterministic across runs.
        eval_holdout: number of trailing tasks per ``task_type`` to reserve for ``eval``.

    Returns:
        ``datasets.Dataset`` with columns ``prompt`` (list, empty), ``answer`` (str, ``""``),
        and ``info`` (dict with ``task_id`` / ``task_type`` plus the 5 plan-mandated fields).
    """
    if split not in {"all", "train", "eval"}:
        raise ValueError(f"split must be one of 'all', 'train', 'eval'; got {split!r}")
    if eval_holdout < 0:
        raise ValueError(f"eval_holdout must be >= 0, got {eval_holdout}")

    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"data_root not found: {root}")

    unsupported = sorted(set(task_types) - set(SUPPORTED_TASK_TYPES))
    if unsupported:
        raise ValueError(
            f"unsupported task_types {unsupported}; expected subset of {SUPPORTED_TASK_TYPES}"
        )

    rows: list[dict[str, Any]] = []
    for task_type in task_types:
        task_dirs = _iter_task_dirs(root, task_type)
        if not task_dirs:
            raise FileNotFoundError(f"no '{task_type}*' dirs under {root}")

        if split == "train":
            selected = task_dirs[: max(0, len(task_dirs) - eval_holdout)]
        elif split == "eval":
            selected = task_dirs[max(0, len(task_dirs) - eval_holdout) :]
        else:
            selected = task_dirs

        for task_num, task_dir in selected:
            payload = _scan_task(task_dir)
            if payload is None:
                continue
            task_id = f"{task_type}{task_num}"
            rows.append(
                {
                    "prompt": [],
                    "answer": "",
                    "info": {
                        "task_id": task_id,
                        "task_type": task_type,
                        **payload,
                    },
                }
            )

    if not rows:
        raise RuntimeError(
            f"no valid tasks discovered under {root} for task_types={list(task_types)} split={split}"
        )

    return Dataset.from_list(rows)
