"""Phase 2 dataset tests.

These cover the plan-mandated invariants:
- placement is non-empty
- every row exposes the 5 ``info`` fields plus ``task_id`` / ``task_type``
- ``goal_image_path`` ends in ``.png`` and the referenced file exists
- ``train`` + ``eval`` splits cover ``all`` exactly once
"""

from pathlib import Path

import pytest

from blendergym.dataset import (
    REQUIRED_FILES,
    SUPPORTED_TASK_TYPES,
    build_dataset,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data" / "blendergym"

REQUIRED_INFO_KEYS = {
    "task_id",
    "task_type",
    "task_dir",
    "start_code",
    "init_image_path",
    "goal_image_path",
    "blend_file_path",
}


pytestmark = pytest.mark.skipif(
    not (DATA_ROOT / "placement1").is_dir(),
    reason="data/blendergym/placement1 missing — Phase 0(a) not satisfied",
)


def test_supported_task_types_contains_placement():
    assert "placement" in SUPPORTED_TASK_TYPES


def test_required_files_match_viga_layout():
    expected = {
        ("blender_file.blend",),
        ("start.py",),
        ("renders", "start", "render1.png"),
        ("renders", "goal", "render1.png"),
    }
    assert set(REQUIRED_FILES) == expected


def test_build_dataset_placement_basic():
    ds = build_dataset(DATA_ROOT, task_types=("placement",))
    assert len(ds) >= 1, "expected at least one placement task"
    assert ds.column_names == ["prompt", "answer", "info"]

    for row in ds:
        info = row["info"]
        assert REQUIRED_INFO_KEYS.issubset(info.keys()), (
            f"missing keys: {REQUIRED_INFO_KEYS - set(info.keys())}"
        )
        assert info["task_type"] == "placement"
        assert info["task_id"].startswith("placement")
        assert isinstance(info["start_code"], str) and info["start_code"].strip()


def test_goal_image_path_is_png_and_exists():
    ds = build_dataset(DATA_ROOT, task_types=("placement",))
    for row in ds:
        goal = Path(row["info"]["goal_image_path"])
        assert goal.suffix == ".png", f"goal_image_path is not .png: {goal}"
        assert goal.is_file(), f"goal image missing: {goal}"


def test_paths_are_absolute():
    ds = build_dataset(DATA_ROOT, task_types=("placement",))
    for row in ds:
        info = row["info"]
        for key in ("task_dir", "init_image_path", "goal_image_path", "blend_file_path"):
            p = Path(info[key])
            assert p.is_absolute(), f"{key} is not absolute: {p}"


def test_split_partitions_all_exactly_once():
    full = build_dataset(DATA_ROOT, task_types=("placement",), split="all")
    train = build_dataset(DATA_ROOT, task_types=("placement",), split="train", eval_holdout=5)
    eval_ = build_dataset(DATA_ROOT, task_types=("placement",), split="eval", eval_holdout=5)
    full_ids = [r["info"]["task_id"] for r in full]
    train_ids = [r["info"]["task_id"] for r in train]
    eval_ids = [r["info"]["task_id"] for r in eval_]
    assert train_ids + eval_ids == full_ids
    assert len(eval_ids) == 5


def test_unknown_task_type_rejected():
    with pytest.raises(ValueError):
        build_dataset(DATA_ROOT, task_types=("does_not_exist",))


def test_missing_data_root_raises():
    with pytest.raises(FileNotFoundError):
        build_dataset("/tmp/__definitely_missing_blendergym_root__", task_types=("placement",))
