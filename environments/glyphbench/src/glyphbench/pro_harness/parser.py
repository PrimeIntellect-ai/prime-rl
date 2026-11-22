"""Robust action parsing for the Pro harness.

Unlike the strict ``<action>NAME</action>``-only parser used by the default
GlyphBench harness (which forfeits on any deviation), the Pro harness uses a
multi-strategy fallback chain so an off-the-shelf model rarely forfeits a
turn on formatting alone. This ports the original Craftax harness'
``_parse_action`` chain and resolves every candidate through the env's own
``ActionSpec.index_of`` (which already handles case-folding + NOOP/WAIT
aliases) plus any adapter-supplied synonyms.

Parsing always happens in the *action region* — the text after the final
``</think>`` — so action names quoted inside private reasoning never get
mistaken for the committed action. Last-match-wins within that region.
"""

from __future__ import annotations

import re

from glyphbench.core.action import ActionSpec
from glyphbench.protocol import action_parse_region

NO_ACTION = "no_action"
UNKNOWN_NAME = "unknown_name"

_XML_ACTION_RE = re.compile(r"<\s*action\s*>(.*?)<\s*/\s*action\s*>", re.IGNORECASE | re.DOTALL)
_LABELLED_RE = re.compile(
    r"(?:action|act|choose|select|move)\s*[:=]\s*`?\s*([A-Za-z][A-Za-z0-9_ ]*)",
    re.IGNORECASE,
)
_JSON_RE = re.compile(r'"action"\s*:\s*"([^"]+)"', re.IGNORECASE)


def _resolve(candidate: str, spec: ActionSpec, synonyms: dict[str, str]) -> int | None:
    cand = (candidate or "").strip().strip(".`*\"' ")
    if not cand:
        return None
    try:
        return spec.index_of(cand)
    except KeyError:
        pass
    syn = synonyms.get(cand.lower())
    if syn:
        try:
            return spec.index_of(syn)
        except KeyError:
            return None
    return None


def parse_action(
    raw_text: str,
    spec: ActionSpec,
    *,
    noop: str = "NOOP",
    synonyms: dict[str, str] | None = None,
) -> tuple[int, str, bool, str | None]:
    """Return ``(action_idx, action_name, parse_failed, failure_reason)``.

    On success ``parse_failed`` is False and ``action_name`` is the canonical
    spec name. On failure the noop index/name is returned with
    ``parse_failed=True`` and a reason of ``no_action`` / ``unknown_name``.
    """
    synonyms = synonyms or {}
    region = action_parse_region(raw_text)

    # 1) Explicit <action>NAME</action> — last complete tag wins.
    tags = _XML_ACTION_RE.findall(region)
    saw_tag = bool(tags)
    for cand in reversed(tags):
        idx = _resolve(cand, spec, synonyms)
        if idx is not None:
            return idx, spec.names[idx], False, None

    # 2) Labelled "Action: NAME" — last match wins.
    labelled = _LABELLED_RE.findall(region)
    for cand in reversed(labelled):
        idx = _resolve(cand, spec, synonyms)
        if idx is not None:
            return idx, spec.names[idx], False, None

    # 3) A standalone line that is exactly an action name — last wins.
    for line in reversed([ln.strip() for ln in region.splitlines() if ln.strip()]):
        idx = _resolve(line, spec, synonyms)
        if idx is not None:
            return idx, spec.names[idx], False, None

    # 4) JSON-ish {"action": "NAME"}.
    for cand in reversed(_JSON_RE.findall(region)):
        idx = _resolve(cand, spec, synonyms)
        if idx is not None:
            return idx, spec.names[idx], False, None

    # 5) Longest known action name / synonym appearing as a whole word.
    #    Only "distinctive" tokens (containing '_' or >=5 chars) are eligible
    #    so short names like DO / REST / UP can't match ordinary prose words.
    low = region.lower()
    vocab: list[tuple[str, str]] = [(n.lower(), n) for n in spec.names]
    vocab += [(k.lower(), v) for k, v in synonyms.items()]
    vocab = [(tok, tgt) for tok, tgt in vocab if "_" in tok or len(tok) >= 5]
    vocab.sort(key=lambda kv: len(kv[0]), reverse=True)
    best: tuple[int, str] | None = None  # (position, target_name)
    for token, target in vocab:
        # Prefer the match that appears latest in the region.
        for match in re.finditer(
            r"(?<![A-Za-z0-9_])" + re.escape(token) + r"(?![A-Za-z0-9_])", low
        ):
            if best is None or match.start() > best[0]:
                best = (match.start(), target)
    if best is not None:
        idx = _resolve(best[1], spec, synonyms)
        if idx is not None:
            return idx, spec.names[idx], False, None

    # Failure — return the env's noop for callers needing a concrete action.
    try:
        nidx = spec.index_of(noop)
    except KeyError:
        nidx = 0
    reason = UNKNOWN_NAME if (saw_tag or labelled) else NO_ACTION
    return nidx, spec.names[nidx], True, reason
