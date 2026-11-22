"""Tests for the Pro harness memory subsystem."""

from __future__ import annotations

from glyphbench.pro_harness.memory import (
    Scratchpad,
    SpatialMemory,
    apply_memory,
)


def test_scratchpad_merge_preserves_unmentioned_sections():
    sp = Scratchpad()
    strategy_before = sp.sections["STRATEGY"]
    plan_before = sp.sections["PLAN"]
    changed = sp.update_from_text("TACTICAL: head north to the lake")
    assert changed
    assert sp.sections["TACTICAL"] == "head north to the lake"
    # Unmentioned sections preserved.
    assert sp.sections["STRATEGY"] == strategy_before
    assert sp.sections["PLAN"] == plan_before


def test_scratchpad_aliases_map_to_canonical():
    sp = Scratchpad()
    sp.update_from_text("AREA_NOTES: cleared the west room")
    assert "cleared the west room" in sp.sections["FLOOR_NOTES"]
    sp.update_from_text("CHECKLIST: craft pickaxe before descending")
    assert "pickaxe" in sp.sections["FLOOR_CHECKLIST"]


def test_scratchpad_caps_section_length():
    sp = Scratchpad()
    sp.update_from_text("TACTICAL: " + "x" * 5000)
    assert len(sp.sections["TACTICAL"]) <= 400


def test_scratchpad_initial_overrides():
    sp = Scratchpad({"STRATEGY": "win the game", "PLAN": "do stuff"})
    assert sp.sections["STRATEGY"] == "win the game"


def test_spatial_add_dedup_and_remove():
    sm = SpatialMemory()
    sm.add("0", "water", 20, 30, "lake")
    sm.add("0", "water", 20, 30, "still lake")  # same pos -> update in place
    assert sm.count() == 1
    sm.add("0", "chest", 5, 5)
    assert sm.count() == 2
    removed = sm.remove("0", kind="chest")
    assert removed == 1
    assert sm.count() == 1


def test_spatial_per_area_cap():
    sm = SpatialMemory()
    for i in range(40):
        sm.add("1", f"k{i}", i, 0)
    # capped at MAX_LANDMARKS_PER_AREA (24)
    assert sm.count() <= 24


def test_spatial_get_query():
    sm = SpatialMemory()
    sm.add("0", "water", 1, 1)
    sm.add("0", "chest", 2, 2)
    sm.add("1", "water", 3, 3)
    assert len(sm.get(area="0")) == 2
    assert len(sm.get(area="0", kind="water")) == 1
    assert len(sm.get(kind="water")) == 2


def test_apply_memory_full_block():
    sp = Scratchpad()
    sm = SpatialMemory()
    resp = (
        "<memory>\n"
        "TACTICAL: at a lake; refill then go to table\n"
        "PLAN:\n1. [x] place table\n2. [ ] mine stone\n"
        "MAP_ADD: area=0, type=water, row=20, col=30, note=big lake\n"
        "MAP_GET: area=0, type=water\n"
        "</memory>"
    )
    upd = apply_memory(resp, sp, sm)
    assert not upd.parse_failed
    assert upd.updated
    assert sm.count() == 1
    assert "water" in upd.query_results
    assert "lake" in sp.sections["TACTICAL"]
    assert "mine stone" in sp.sections["PLAN"]


def test_apply_memory_no_tag_is_parse_failed_but_lenient():
    sp = Scratchpad()
    sm = SpatialMemory()
    # No <memory> tag and nothing recognizable -> parse_failed, memory kept.
    upd = apply_memory("I think I should move left.", sp, sm)
    assert upd.parse_failed
    # The scratchpad's TACTICAL fallback may fire, but STRATEGY stays.
    assert sp.sections["STRATEGY"].startswith("Survive")


def test_apply_memory_strips_action_tags():
    sp = Scratchpad()
    sm = SpatialMemory()
    resp = "<memory>TACTICAL: go\n<action>MOVE_LEFT</action>\n</memory>"
    apply_memory(resp, sp, sm)
    assert "<action>" not in sp.sections["TACTICAL"]


def test_map_get_all_and_remove_feedback():
    sp = Scratchpad()
    sm = SpatialMemory()
    sm.add("0", "water", 1, 1)
    upd = apply_memory("<memory>MAP_GET_ALL\nMAP_REMOVE: area=0, type=water</memory>", sp, sm)
    assert "MAP_GET_ALL" in upd.query_results
    assert "removed 1" in upd.query_results
    assert sm.count() == 0
