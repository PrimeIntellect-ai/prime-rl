"""Tests for achievement counting + raw-score extraction (env-info quirks)."""

from __future__ import annotations

from glyphbench.pro_harness.adapters import (
    CraftaxAdapter,
    NetHackAdapter,
    get_adapter,
)
from glyphbench.pro_harness.harness import _coerce_name_list, _unlocked_achievements


def test_coerce_name_list_variants():
    assert _coerce_name_list(["a", "b"]) == ["a", "b"]
    assert _coerce_name_list(("a",)) == ["a"]
    assert _coerce_name_list("['collect_wood', 'place_table']") == [
        "collect_wood", "place_table"]
    assert _coerce_name_list("[]") == []
    assert _coerce_name_list("") == []
    assert _coerce_name_list(None) == []
    # numpy-style object with __iter__
    assert set(_coerce_name_list({"x", "y"})) == {"x", "y"}


def test_unlocked_from_explicit_list_even_when_flags_lag():
    # Fork env: per-name flags lag (0.0) but achievements_unlocked is immediate.
    info = {
        "num_achievements": 1,
        "achievements_unlocked": ["collect_wood"],
        "Achievements/collect_wood": 0.0,
        "Achievements/place_table": 0.0,
    }
    assert _unlocked_achievements(info) == {"collect_wood"}


def test_unlocked_from_flags():
    # Pure-python env: per-name flags are authoritative.
    info = {"Achievements/eat_cow": 1.0, "Achievements/place_table": 0.0}
    assert _unlocked_achievements(info) == {"eat_cow"}


def test_unlocked_union_and_stringified_list():
    info = {
        "Achievements/eat_cow": 1.0,
        "achievements_unlocked": "['collect_wood', 'place_table']",
    }
    assert _unlocked_achievements(info) == {"eat_cow", "collect_wood", "place_table"}


def test_unlocked_empty():
    assert _unlocked_achievements({"num_achievements": 0, "achievements_unlocked": "[]"}) == set()


def test_craftax_raw_score_is_achievement_count():
    ad = CraftaxAdapter()
    assert ad.raw_score_label() == "achievements"
    assert ad.raw_score({"num_achievements": 7}) == 7.0
    assert ad.raw_score({}) is None


def test_nethack_raw_score_is_game_score():
    ad = NetHackAdapter()
    assert ad.raw_score_label() == "score"
    assert ad.raw_score({"nethack_blstats_score": 1234}) == 1234.0
    assert ad.raw_score({"score": 5}) == 5.0
    assert ad.raw_score({}) is None


def test_generic_adapter_has_no_raw_score():
    ad = get_adapter("glyphbench/something-else-v0")
    assert ad.raw_score({"whatever": 1}) is None
