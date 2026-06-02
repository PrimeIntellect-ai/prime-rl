"""Per-seat visibility verification — what each seat saw, generalized to any agent count.

``Step.prompt`` is the contract's promise: *exactly* the messages a seat saw at
its turn (see ``schema.py``). This module audits that promise against a
``VisibilityPolicy``: it reconstructs the expected injected-this-turn delta for
each member's consecutive turns and flags two failure classes —

  - STRUCTURE : the prompt delta between a member's turn k-1 and k is not the
                shape the policy predicts (its own prior completion re-rendered
                as an assistant turn, plus protocol-injected user content from
                already-spoken seats).
  - LEAKAGE   : content attributed to a seat the policy forbids appears in a
                member's prompt — concretely, a not-yet-spoken (future) seat's
                contribution showing up in an earlier seat's view. A causal /
                acyclicity violation: turn ``t`` may only see turns ``< t``.

How cross-seat content is encoded in the rollout (measured on a real prime-rl
debate dump): another seat's contribution is interpolated into a ``user`` turn
as text, prefixed at position 0 with ``[<member_id>]``. A seat's own prior
completion appears as an ``assistant`` turn (re-rendered by the chat template,
so it is NOT byte-identical to ``Step.completion`` — the delta is matched on the
*newly present* messages, not on equality with the raw completion).

The policy is turn-order, not phase: ``DebateVisibilityPolicy`` encodes "a seat
may see any seat that produced a completion at an earlier turn index" — which is
what an open sequential debate (a proposes blind, b proposes after seeing a, both
critique, judge reads all) actually does, and which generalizes to N seats with
no debate-specific hard-coding.

Findings carry the FULL offending messages, never truncated — the bug lives in
the part you would have cut.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, Field

from rollout_viewer.schema import Episode, Message, Step

FindingKind = Literal["leakage", "structure"]

# Cross-seat content is prefixed ``[<member_id>]`` at the start of a user turn.
# ``member_id`` s in the wild are ``[a-z][a-z0-9_]*`` (debater_a, debater_b,
# judge). The pattern is intentionally anchored so substring "[link]"-style text
# inside an argument is not misread as an attribution; we only treat a prefix as
# a seat attribution when it names a known member of THIS episode (see
# ``_scan_message``).
_PREFIX_RE = re.compile(r"\[([a-zA-Z][a-zA-Z0-9_]*)\]")


class VisibilityFinding(BaseModel):
    """One per-seat visibility deviation. Carries the full offending messages."""

    kind: FindingKind
    member_id: str | None
    step_index: int
    phase: str | None = None
    detail: str
    # The seat(s) whose content was seen-but-forbidden (leakage) or the seats
    # the policy did allow (context for a structure finding).
    offending_seats: list[str] = Field(default_factory=list)
    allowed_seats: list[str] = Field(default_factory=list)
    # FULL messages — never truncated. For leakage: the prompt message(s) that
    # carry the forbidden attribution. For structure: the unexpected delta.
    messages: list[Message] = Field(default_factory=list)


class VisibilityPolicy(ABC):
    """Who-sees-whom across an episode's turns. Implementations encode a protocol."""

    @abstractmethod
    def allowed_sources(
        self, *, viewer: str | None, spoken_before: list[str | None], turn_index: int
    ) -> set[str | None]:
        """Seats whose content ``viewer`` may see at ``turn_index``.

        ``spoken_before`` is the ordered list of ``member_id`` s that produced a
        completion at turn indices ``< turn_index`` (duplicates kept — a seat may
        have spoken multiple times). ``viewer`` is always allowed to see itself.
        """


class DebateVisibilityPolicy(VisibilityPolicy):
    """Causal / turn-order visibility: a seat sees any seat that spoke earlier.

    This is the open-debate default and generalizes to any agent count: turn
    ``t`` may see the contributions of every turn ``< t`` and its own prior
    turns. Forbidden = a seat that has not yet spoken (a future turn) — a
    causality break.
    """

    def allowed_sources(
        self, *, viewer: str | None, spoken_before: list[str | None], turn_index: int
    ) -> set[str | None]:
        return set(spoken_before) | {viewer}


def prompt_delta(prev_prompt: list[Message], curr_prompt: list[Message]) -> list[Message]:
    """Messages newly present in ``curr_prompt`` vs ``prev_prompt`` (same member).

    The injected-this-turn delta: what the protocol added to a seat's view since
    its previous turn. Robust to reordering — membership is by ``(role, content)``,
    not position. Repeated identical messages are accounted for by multiplicity
    (a duplicated message in ``curr`` beyond its ``prev`` count counts as new).
    """
    prev_counts: dict[tuple[str, str], int] = {}
    for m in prev_prompt:
        prev_counts[_key(m)] = prev_counts.get(_key(m), 0) + 1

    delta: list[Message] = []
    seen: dict[tuple[str, str], int] = {}
    for m in curr_prompt:
        k = _key(m)
        seen[k] = seen.get(k, 0) + 1
        if seen[k] > prev_counts.get(k, 0):
            delta.append(m)
    return delta


def verify_episode(ep: Episode, policy: VisibilityPolicy = DebateVisibilityPolicy()) -> list[VisibilityFinding]:
    """Audit per-seat visibility for one episode against ``policy``.

    For every turn: detect LEAKAGE — content attributed (via the ``[seat]``
    prefix) to a seat the policy forbids at this turn. For each member's
    consecutive turns: detect STRUCTURE — a prompt delta that is not the shape
    the policy predicts (own prior completion + protocol injections from allowed
    seats only).

    Errored / filtered episodes are skipped (their trajectories may be partial by
    construction — the contract treats those as a separate, expected state).
    """
    if ep.error is not None or ep.is_filtered:
        return []

    seats = set(ep.members)
    findings: list[VisibilityFinding] = []
    spoken_before: list[str | None] = []
    # last completion-bearing turn index per member, for structure reconstruction.
    last_turn_of: dict[str | None, int] = {}

    for step in ep.steps:
        allowed = policy.allowed_sources(
            viewer=step.member_id,
            spoken_before=spoken_before,
            turn_index=step.index,
        )
        # --- LEAKAGE: any prompt message attributing content to a forbidden seat.
        for msg in step.prompt:
            scan = _scan_message(msg, seats)
            if scan.unscannable:
                # Fail loud: non-string content we cannot serialize to scannable
                # text silently disables leakage detection. Surface it as a
                # structural finding (full message, untruncated) instead.
                findings.append(
                    VisibilityFinding(
                        kind="structure",
                        member_id=step.member_id,
                        step_index=step.index,
                        phase=step.phase,
                        detail=(
                            f"prompt message for {step.member_id!r} at turn "
                            f"{step.index} has non-string content with no scannable "
                            f"text segments (content type "
                            f"{type(msg.content).__name__!r}); leakage attribution "
                            f"cannot be verified for it"
                        ),
                        allowed_seats=sorted(s for s in allowed if s is not None),
                        messages=[msg],
                    )
                )
                continue
            forbidden = {s for s in scan.seats if s not in allowed and s != step.member_id}
            if forbidden:
                findings.append(
                    VisibilityFinding(
                        kind="leakage",
                        member_id=step.member_id,
                        step_index=step.index,
                        phase=step.phase,
                        detail=(
                            f"{step.member_id!r} at turn {step.index} sees content "
                            f"attributed to {sorted(forbidden)}, which the policy "
                            f"forbids (allowed sources: "
                            f"{sorted(s for s in allowed if s is not None)})"
                        ),
                        offending_seats=sorted(forbidden),
                        allowed_seats=sorted(s for s in allowed if s is not None),
                        messages=[msg],
                    )
                )

        # --- STRUCTURE: delta vs this member's previous turn must be explainable.
        prev_idx = last_turn_of.get(step.member_id)
        if prev_idx is not None:
            delta = prompt_delta(ep.steps[prev_idx].prompt, step.prompt)
            structure_finding = _check_delta_structure(step=step, delta=delta, allowed=allowed, seats=seats)
            if structure_finding is not None:
                findings.append(structure_finding)

        last_turn_of[step.member_id] = step.index
        spoken_before.append(step.member_id)

    return findings


def _check_delta_structure(
    *, step: Step, delta: list[Message], allowed: set[str | None], seats: set[str]
) -> VisibilityFinding | None:
    """A member's turn-to-turn prompt delta must be: zero or more assistant turns
    (its own prior completions, re-rendered) followed by zero or more protocol
    ``user`` injections — none of which may attribute content to a forbidden seat
    (that case is already a leakage finding; here we flag a delta carrying an
    unexpected role, which would mean a seat-view was reshaped, not extended).
    """
    unexpected = [m for m in delta if m.role not in ("assistant", "user", "system")]
    # Newly-appearing system turns mid-episode are a structural anomaly: the
    # system prompt is fixed per seat and should never be injected after turn 0.
    new_system = [m for m in delta if m.role == "system"]
    bad = unexpected + new_system
    if not bad:
        return None
    return VisibilityFinding(
        kind="structure",
        member_id=step.member_id,
        step_index=step.index,
        phase=step.phase,
        detail=(
            f"prompt delta for {step.member_id!r} at turn {step.index} contains "
            f"{len(bad)} unexpected message(s) (roles="
            f"{sorted({m.role for m in bad})}); expected only re-rendered "
            f"assistant turns + protocol user injections"
        ),
        allowed_seats=sorted(s for s in allowed if s is not None),
        messages=bad,
    )


def verify_run(episodes: list[Episode]) -> dict[str, int]:
    """One-line summary: clean vs flagged episode counts + finding totals."""
    clean = 0
    flagged = 0
    leakage = 0
    structure = 0
    for ep in episodes:
        fs = verify_episode(ep)
        if fs:
            flagged += 1
            leakage += sum(1 for f in fs if f.kind == "leakage")
            structure += sum(1 for f in fs if f.kind == "structure")
        else:
            clean += 1
    return {
        "episodes": len(episodes),
        "clean": clean,
        "flagged": flagged,
        "leakage_findings": leakage,
        "structure_findings": structure,
    }


# --- helpers ------------------------------------------------------------------


def _key(m: Message) -> tuple[str, str]:
    """Identity for delta membership: role + content rendered to a stable string.

    ``content`` may be str, a structured list (multimodal), or None — all are
    serialized deterministically so list-content messages diff correctly.

    This is content-identity ONLY: it deliberately ignores ``Message.extra``
    (tool_calls / name / provider-specific keys). So a delta-based check
    (``prompt_delta``) will NOT flag a message whose only change between turns is
    in ``extra`` — two messages with identical ``(role, content)`` but different
    ``extra`` collapse to the same key. Visibility is about who-saw-what text,
    not tool-call plumbing; the delta is the right granularity for that.
    """
    if isinstance(m.content, str):
        c = m.content
    elif m.content is None:
        c = ""
    else:
        c = repr(m.content)
    return (m.role, c)


class _MessageScan(BaseModel):
    """Result of scanning a message for ``[seat]`` attributions.

    ``unscannable`` is True only when the content is non-string AND yields no
    text segments to scan — the fail-loud case (caller emits a structural
    finding). For string content, or list content with at least one text
    segment, ``unscannable`` is False and ``seats`` carries the attributions.
    """

    seats: set[str] = Field(default_factory=set)
    unscannable: bool = False


def _list_content_text(content: list[object]) -> str | None:
    """Concatenate the text-bearing segments of multimodal list content.

    Handles the OpenAI/verifiers chat convention where ``content`` is a list of
    parts: ``{"type": "text"|"input_text", "text": "..."}`` (and bare strings).
    Non-text parts (images, audio) carry no ``[seat]`` attribution and are
    skipped. Returns the joined scannable text, or ``None`` when the list has no
    text-bearing segment at all (the genuinely-unscannable case).
    """
    segments: list[str] = []
    for part in content:
        if isinstance(part, str):
            segments.append(part)
        elif isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str):
                segments.append(text)
    if not segments:
        return None
    return "\n".join(segments)


def _scan_message(msg: Message, seats: set[str]) -> _MessageScan:
    """Scan a message for ``[<seat>]`` attribution prefixes.

    Only prefixes naming a known member of the episode count — arbitrary
    ``[bracketed]`` text inside an argument is not a seat attribution. Scans the
    whole message (a single user turn can carry multiple seats' contributions).

    Multimodal (list) content is serialized to its text segments and scanned the
    same way — it never silently disables leakage detection. If list content has
    no scannable text segment (e.g. image-only), or content is some other
    non-string type, the scan is marked ``unscannable`` so the caller can fail
    loud with a structural finding rather than treating it as zero attributions.
    """
    if isinstance(msg.content, str):
        text: str | None = msg.content
    elif msg.content is None:
        text = ""
    elif isinstance(msg.content, list):
        text = _list_content_text(msg.content)
    else:
        text = None
    if text is None:
        return _MessageScan(unscannable=True)
    return _MessageScan(seats={tok for tok in _PREFIX_RE.findall(text) if tok in seats})
