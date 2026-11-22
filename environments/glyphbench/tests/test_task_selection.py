"""Tests for task_selection.list_task_ids."""
from __future__ import annotations

from glyphbench.core.task_selection import list_task_ids


def test_no_args_returns_full_registry_minus_dummy():
    ids = list_task_ids()
    assert all(not i.startswith("glyphbench/__dummy") for i in ids)
    assert any(i.startswith("glyphbench/atari-") for i in ids)
    assert any(i.startswith("glyphbench/minigrid-") for i in ids)
    assert "glyphbench/minigrid-empty-5x5-v0" in ids
    assert "glyphbench/minigrid-memory-s17-v0" in ids
    assert "glyphbench/minihack-corridor-r5-v0" in ids
    assert "glyphbench/minihack-sink-distract-v0" in ids


def test_include_suites_whitelist():
    ids = list_task_ids(include_suites=["minigrid"])
    assert all(i.startswith("glyphbench/minigrid-") for i in ids)
    assert len(ids) > 0


def test_exclude_suites_blacklist():
    ids = list_task_ids(exclude_suites=["atari"])
    assert all(not i.startswith("glyphbench/atari-") for i in ids)
    assert any(i.startswith("glyphbench/minigrid-") for i in ids)


def test_include_tasks_exact_id():
    ids = list_task_ids(include_tasks=["glyphbench/minigrid-empty-5x5-v0"])
    assert ids == ["glyphbench/minigrid-empty-5x5-v0"]


def test_include_tasks_pattern():
    ids = list_task_ids(include_tasks=["glyphbench/atari-*-v0"])
    assert all(i.startswith("glyphbench/atari-") for i in ids)
    assert len(ids) >= 50


def test_exclude_tasks_pattern():
    ids = list_task_ids(exclude_tasks=["glyphbench/atari-*-v0"])
    assert all(not i.startswith("glyphbench/atari-") for i in ids)


def test_exclude_wins_on_conflict():
    ids = list_task_ids(
        include_suites=["atari"],
        exclude_tasks=["glyphbench/atari-pong-v0"],
    )
    assert "glyphbench/atari-pong-v0" not in ids
    assert "glyphbench/atari-breakout-v0" in ids


def test_include_dummy_explicit():
    ids = list_task_ids(include_tasks=["glyphbench/__dummy-v0"])
    assert ids == ["glyphbench/__dummy-v0"]


def test_empty_result_is_empty_list():
    ids = list_task_ids(include_suites=["nonexistent_suite"])
    assert ids == []


def test_combined_include_and_exclude():
    ids = list_task_ids(
        include_suites=["atari", "minigrid"],
        exclude_suites=["minigrid"],
    )
    assert all(i.startswith("glyphbench/atari-") for i in ids)


def test_returns_sorted():
    ids = list_task_ids(include_suites=["minigrid"])
    assert ids == sorted(ids)


def test_unknown_suite_in_exclude_is_silently_ignored():
    # Excluding a suite that doesn't exist shouldn't error; it's a no-op.
    ids = list_task_ids(exclude_suites=["does_not_exist"])
    assert any(i.startswith("glyphbench/atari-") for i in ids)
