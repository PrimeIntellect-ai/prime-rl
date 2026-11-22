"""GlyphbenchXMLParser: strict <action>NAME</action> only.

The parser extracts the LAST complete ``<action>NAME</action>`` match from
the model's action region and verifies NAME against the env's ``ActionSpec``.
For native-thinking outputs, the action region is the text after the final
``</think>``; this prevents literal ``<action>`` mentions inside private
reasoning from consuming the final action tag. For non-native instruct
outputs with no ``</think>``, the entire response remains the action region.
Any other shape (unclosed tag, JSON, bare-token, missing tag, unknown name)
forfeits the turn — see ``BaseGlyphEnv.forfeit_turn`` and the corresponding
behavior in ``verifiers_integration.env._apply_action_response``.

Last-match-wins is preserved so that models that quote ``<action>`` tags
inside their reasoning trace still get scored on the final committed tag.
"""

from __future__ import annotations

from typing import Any

import verifiers as vf

from glyphbench.core.action import ActionSpec
from glyphbench.protocol import NO_ACTION_TAG as NO_ACTION_TAG
from glyphbench.protocol import UNKNOWN_NAME as UNKNOWN_NAME
from glyphbench.protocol import action_parse_region as action_parse_region
from glyphbench.protocol import parse_action_response


class GlyphbenchXMLParser(vf.XMLParser):
    """Strict XML parser for the glyphbench verifiers integration.

    Returns ``(idx, name, parse_failed, parse_failure_reason)`` from
    ``parse_action``:

    - on success: ``(spec.index_of(name), spec.names[idx], False, None)``
    - on missing/unclosed/non-XML output:
      ``(noop_idx, noop_name, True, "no_action_tag")``
    - on tag found but NAME not in spec:
      ``(noop_idx, noop_name, True, "unknown_name")``

    The noop_idx/noop_name are returned for backward-compat with callers
    that need a non-None action; the verifiers integration ignores these
    and forfeits the turn instead.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            fields=["action"],
            answer_field="action",
            **kwargs,
        )

    def parse_action(
        self,
        raw_text: str,
        spec: ActionSpec,
        *,
        noop: str,
    ) -> tuple[int, str, bool, str | None]:
        result = parse_action_response(raw_text, spec, noop=noop)
        return (
            result.index,
            result.name,
            result.failed,
            result.failure_reason,
        )
