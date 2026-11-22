"""Framework-independent protocol helpers for driving GlyphBench games.

This module is the stable boundary for rollout frameworks.  It owns the text
action protocol and re-exports the prompt renderers, while the Verifiers
adapters only translate framework lifecycle events to ``BaseGlyphEnv`` calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from glyphbench.core.action import ActionSpec

NO_ACTION_TAG = "no_action_tag"
UNKNOWN_NAME = "unknown_name"

_XML_ACTION_RE = re.compile(
    r"<\s*action\s*>(.*?)<\s*/\s*action\s*>", re.DOTALL | re.IGNORECASE
)
_THINK_CLOSE_RE = re.compile(r"<\s*/\s*think\s*>", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ActionParseResult:
    """Result of parsing one model reply against an environment action space."""

    index: int
    name: str
    failed: bool = False
    failure_reason: str | None = None


def action_parse_region(raw_text: str) -> str:
    """Return the response slice in which the final action is authoritative."""

    text = raw_text or ""
    closes = list(_THINK_CLOSE_RE.finditer(text))
    return text[closes[-1].end() :] if closes else text


def parse_action_response(
    raw_text: str,
    spec: ActionSpec,
    *,
    noop: str,
) -> ActionParseResult:
    """Parse the last complete ``<action>NAME</action>`` in a model reply.

    The returned fallback index/name are always valid for ``spec``.  Callers
    choose whether a failed parse freezes the game or applies that fallback.
    """

    matches = _XML_ACTION_RE.findall(action_parse_region(raw_text))
    if not matches or not matches[-1].strip():
        return _noop_result(spec, noop, NO_ACTION_TAG)
    candidate = matches[-1].strip()
    try:
        index = spec.index_of(candidate)
    except KeyError:
        return _noop_result(spec, noop, UNKNOWN_NAME)
    return ActionParseResult(index=index, name=spec.names[index])


def _noop_result(spec: ActionSpec, noop: str, reason: str) -> ActionParseResult:
    try:
        index = spec.index_of(noop)
    except KeyError:
        index = 0
    return ActionParseResult(
        index=index,
        name=spec.names[index],
        failed=True,
        failure_reason=reason,
    )


# Imported after the dependency-free parser is defined so the legacy
# ``glyphbench.verifiers_integration`` package can re-export these helpers
# without introducing an import cycle during its own initialization.
from glyphbench.verifiers_integration.prompting import (  # noqa: E402, I001
    build_system_prompt,
    render_user_turn,
)


__all__ = [
    "ActionParseResult",
    "NO_ACTION_TAG",
    "UNKNOWN_NAME",
    "action_parse_region",
    "build_system_prompt",
    "parse_action_response",
    "render_user_turn",
]
