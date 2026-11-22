"""Tests for the robust Pro harness action parser."""

from __future__ import annotations

from glyphbench.core.action import ActionSpec
from glyphbench.pro_harness.parser import action_parse_region, parse_action

SPEC = ActionSpec(
    names=("NOOP", "MOVE_LEFT", "MOVE_RIGHT", "DO", "DESCEND", "SHOOT_ARROW"),
    descriptions=("a", "b", "c", "d", "e", "f"),
)
SYN = {
    "left": "MOVE_LEFT", "right": "MOVE_RIGHT", "mine": "DO", "attack": "DO",
    "go down": "DESCEND", "shoot": "SHOOT_ARROW",
}


def _parse(text):
    return parse_action(text, SPEC, noop="NOOP", synonyms=SYN)


def test_xml_tag_last_match_wins():
    idx, name, pf, reason = _parse(
        "maybe <action>MOVE_LEFT</action> ... actually <action>MOVE_RIGHT</action>"
    )
    assert name == "MOVE_RIGHT" and not pf


def test_action_region_after_think():
    txt = "<think>I'll write <action>MOVE_LEFT</action> later</think>\n<action>DESCEND</action>"
    idx, name, pf, _ = _parse(txt)
    assert name == "DESCEND" and not pf


def test_think_quoted_action_not_committed():
    # An action named only inside <think> must not be parsed as the commit.
    txt = "<think><action>SHOOT_ARROW</action></think>\nI choose to mine.\n<action>DO</action>"
    idx, name, pf, _ = _parse(txt)
    assert name == "DO"


def test_labelled_action():
    idx, name, pf, _ = _parse("Reasoning...\nAction: DESCEND")
    assert name == "DESCEND" and not pf


def test_standalone_line_synonym():
    idx, name, pf, _ = _parse("Here is my move:\nmine")
    assert name == "DO" and not pf


def test_case_insensitive_name():
    idx, name, pf, _ = _parse("<action>move_left</action>")
    assert name == "MOVE_LEFT" and not pf


def test_json_action():
    idx, name, pf, _ = _parse('{"action": "DESCEND"}')
    assert name == "DESCEND" and not pf


def test_longest_match_distinctive_only():
    # 'shoot' (synonym) -> SHOOT_ARROW via longest-match in prose.
    idx, name, pf, _ = _parse("I will now shoot at the orc from here.")
    assert name == "SHOOT_ARROW" and not pf


def test_prose_fallback_uses_last_occurrence_of_repeated_action():
    _, name, failed, _ = _parse(
        "I considered MOVE_LEFT, then MOVE_RIGHT, but will take MOVE_LEFT now."
    )
    assert name == "MOVE_LEFT" and not failed


def test_short_word_not_matched_in_prose():
    # 'do'/'left' must not be picked from prose by the last-resort strategy.
    idx, name, pf, reason = _parse("I really do not want to go left blindly here.")
    assert pf and name == "NOOP"


def test_parse_failure_returns_noop():
    idx, name, pf, reason = _parse("absolutely nothing actionable here ~~~")
    assert pf and name == "NOOP" and reason == "no_action"


def test_unknown_name_in_tag():
    idx, name, pf, reason = _parse("<action>FLY_AWAY</action>")
    assert pf and reason == "unknown_name"


def test_noop_family_alias_resolves():
    spec = ActionSpec(names=("WAIT", "MOVE_N"), descriptions=("a", "b"))
    idx, name, pf, _ = parse_action("<action>NOOP</action>", spec, noop="WAIT")
    assert name == "WAIT" and not pf


def test_action_parse_region_no_think():
    assert action_parse_region("hello") == "hello"
