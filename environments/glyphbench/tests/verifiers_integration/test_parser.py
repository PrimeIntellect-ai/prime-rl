from __future__ import annotations

import pytest

from glyphbench.core.action import ActionSpec
from glyphbench.verifiers_integration.parser import (
    NO_ACTION_TAG,
    UNKNOWN_NAME,
    GlyphbenchXMLParser,
)


@pytest.fixture
def spec() -> ActionSpec:
    return ActionSpec(
        names=("LEFT", "RIGHT", "UP", "DOWN", "NOOP"),
        descriptions=("l", "r", "u", "d", "n"),
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<think>reason</think><action>LEFT</action>", "LEFT"),
        ("<think>x</think><action>right</action>", "RIGHT"),
        ("<think>x</think><action>   UP\n</action>", "UP"),
        ("<action>LEFT</action>reasoning<action>RIGHT</action>", "RIGHT"),
        ("native reasoning mentions <action>LEFT</action>\n</think><action>UP</action>", "UP"),
    ],
)
def test_parser_accepts_only_final_committed_action_tag(
    spec: ActionSpec,
    text: str,
    expected: str,
) -> None:
    idx, name, failed, reason = GlyphbenchXMLParser().parse_action(
        text,
        spec,
        noop="NOOP",
    )

    assert idx == spec.index_of(expected)
    assert name == expected
    assert failed is False
    assert reason is None


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("<think>no action</think>", NO_ACTION_TAG),
        ("", NO_ACTION_TAG),
        ("<action>UP", NO_ACTION_TAG),
        ('{"action": "DOWN"}', NO_ACTION_TAG),
        ("I will go UP", NO_ACTION_TAG),
        ("</think>Final answer omitted the action tag.", NO_ACTION_TAG),
        ("<action>FLY</action>", UNKNOWN_NAME),
    ],
)
def test_parser_forfeits_malformed_or_unknown_outputs(
    spec: ActionSpec,
    text: str,
    reason: str,
) -> None:
    idx, name, failed, actual_reason = GlyphbenchXMLParser().parse_action(
        text,
        spec,
        noop="NOOP",
    )

    assert idx == spec.index_of("NOOP")
    assert name == "NOOP"
    assert failed is True
    assert actual_reason == reason
