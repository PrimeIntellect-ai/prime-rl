"""Persistent agent memory for the Pro harness.

Two stores, both updated from the model's memory-turn output:

  * ``Scratchpad`` — six free-text sections (STRATEGY / PLAN / LESSONS /
    TACTICAL / FLOOR_NOTES / FLOOR_CHECKLIST) with **merge-on-update**
    semantics: the model emits only the sections it wants to change; the
    rest are preserved. Each section is char-capped so memory can't grow
    without bound. Ported from the original ``StructuredScratchpad``.

  * ``SpatialMemory`` — a per-area landmark database keyed by floor / dungeon
    level. The model adds/removes/queries landmarks with ``MAP_ADD`` /
    ``MAP_REMOVE`` / ``MAP_GET`` / ``MAP_GET_ALL`` commands. ``MAP_GET``
    results are returned so the harness can inject them into the *next*
    action turn (the original's "pending query results" mechanism, which
    the current glyphbench pro harness dropped).

``apply_memory`` parses one memory-turn response (a ``<memory>...</memory>``
block, or a post-``---`` block, or — leniently — the whole text) and applies
both stores in one pass.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

MAX_LANDMARKS_PER_AREA = 20
MAX_TOTAL_LANDMARKS = 160

# (name, max_chars, default)
_SECTION_SPECS: tuple[tuple[str, int, str], ...] = (
    ("STRATEGY", 200,
     "Survive, build capability, and make steady progress toward the objective."),
    ("PLAN", 600,
     "1. [ ] Stabilize immediate survival\n"
     "2. [ ] Acquire the next required capability/resource\n"
     "3. [ ] Advance only once prepared"),
    ("LESSONS", 400, "(Durable mechanics you have confirmed by observation.)"),
    ("TACTICAL", 400, "(Current local situation and the immediate next intent.)"),
    ("FLOOR_NOTES", 800,
     "(Exploration log for the current area: cleared/unexplored regions, "
     "dead ends, hazards, resource and feature locations, routes.)"),
    ("FLOOR_CHECKLIST", 360,
     "(What must be done before leaving this area/floor.)"),
)
_SECTION_NAMES: tuple[str, ...] = tuple(s[0] for s in _SECTION_SPECS)
# Env-agnostic aliases so NetHack ("area"/"level") and free-form headers map
# onto the canonical names.
_SECTION_ALIASES: dict[str, str] = {
    "AREA_NOTES": "FLOOR_NOTES",
    "LOCAL_NOTES": "FLOOR_NOTES",
    "NOTES": "FLOOR_NOTES",
    "MAP_NOTES": "FLOOR_NOTES",
    "AREA_CHECKLIST": "FLOOR_CHECKLIST",
    "CHECKLIST": "FLOOR_CHECKLIST",
    "GOAL": "STRATEGY",
    "OBJECTIVE": "STRATEGY",
}
_ALL_HEADERS = _SECTION_NAMES + tuple(_SECTION_ALIASES)
_SECTION_RE = re.compile(
    r"^\s*(" + "|".join(re.escape(h) for h in _ALL_HEADERS) + r")\s*:\s*"
    r"(.*?)(?=\n\s*(?:" + "|".join(re.escape(h) for h in _ALL_HEADERS) + r")\s*:|\Z)",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)
_MEMORY_RE = re.compile(r"<\s*memory\s*>(.*?)<\s*/\s*memory\s*>", re.IGNORECASE | re.DOTALL)
_ACTION_RE = re.compile(r"<\s*action\s*>.*?<\s*/\s*action\s*>", re.IGNORECASE | re.DOTALL)
_THINK_RE = re.compile(r"<\s*think\s*>.*?<\s*/\s*think\s*>", re.IGNORECASE | re.DOTALL)


def _clip(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _clean_value(value: str | None) -> str:
    if value is None:
        return ""
    cleaned = re.sub(r"\s+", " ", str(value).strip().strip("`\"'"))
    return cleaned.rstrip(".")


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    m = re.search(r"-?\d+", str(value))
    return int(m.group(0)) if m else None


# ---------------------------------------------------------------------------
# Scratchpad
# ---------------------------------------------------------------------------
class Scratchpad:
    """Section-based scratchpad with merge-on-update semantics."""

    def __init__(self, initial: dict[str, str] | None = None) -> None:
        self.sections: dict[str, str] = {}
        self._max: dict[str, int] = {}
        for name, max_chars, default in _SECTION_SPECS:
            self.sections[name] = default
            self._max[name] = max_chars
        if initial:
            for k, v in initial.items():
                key = _SECTION_ALIASES.get(k.upper(), k.upper())
                if key in self.sections and v:
                    self.sections[key] = _clip(v, self._max[key])

    def update_from_text(self, text: str, *, allow_freeform_fallback: bool = True) -> bool:
        """Merge any section headers found in ``text``. Returns True if any
        section changed. Unmentioned sections are preserved.

        When no section headers are present and ``allow_freeform_fallback`` is
        True, non-trivial prose is stored as a TACTICAL note (the original's
        behaviour). The harness disables that fallback for untagged memory
        turns so a stray reasoning dump can't clobber TACTICAL.
        """
        if not text or not text.strip():
            return False
        updated = False
        matched_any = False
        for m in _SECTION_RE.finditer(text):
            matched_any = True
            raw = m.group(1).upper()
            name = _SECTION_ALIASES.get(raw, raw)
            content = m.group(2).strip()
            if name in self.sections and content:
                new = _clip(content, self._max[name])
                if new != self.sections[name]:
                    self.sections[name] = new
                    updated = True
        if not matched_any and allow_freeform_fallback:
            stripped = text.strip()
            if len(stripped) > 10:
                self.sections["TACTICAL"] = _clip(stripped, self._max["TACTICAL"])
                updated = True
        return updated

    def render(self) -> str:
        return "\n".join(f"{name}: {self.sections[name]}" for name in _SECTION_NAMES)

    def to_dict(self) -> dict[str, str]:
        return dict(self.sections)


# ---------------------------------------------------------------------------
# Spatial memory
# ---------------------------------------------------------------------------
@dataclass
class Landmark:
    kind: str
    row: int | None = None
    col: int | None = None
    note: str = ""


class SpatialMemory:
    """Per-area landmark DB with add / remove / query."""

    def __init__(self) -> None:
        self.data: dict[str, list[Landmark]] = defaultdict(list)

    def add(self, area: str, kind: str, row: int | None, col: int | None, note: str = "") -> None:
        area = _clip(_clean_value(area) or "current", 48)
        kind = _clip(_clean_value(kind) or "unknown", 48)
        note = _clip(_clean_value(note), 160)
        entries = self.data[area]
        # Dedup: same position (any kind) updates in place; else same kind+pos.
        for e in entries:
            if row is not None and col is not None and e.row == row and e.col == col:
                e.kind, e.note = kind, note
                return
            if e.kind == kind and e.row == row and e.col == col:
                e.note = note
                return
        if len(entries) >= MAX_LANDMARKS_PER_AREA:
            return
        if sum(len(v) for v in self.data.values()) >= MAX_TOTAL_LANDMARKS:
            return
        entries.append(Landmark(kind=kind, row=row, col=col, note=note))

    def remove(
        self, area: str | None, kind: str | None = None,
        row: int | None = None, col: int | None = None,
    ) -> int:
        removed = 0
        areas = [area] if area is not None else list(self.data.keys())
        for a in areas:
            a = _clip(_clean_value(a) or "current", 48) if a is not None else a
            if a not in self.data:
                continue
            before = len(self.data[a])
            self.data[a] = [
                e for e in self.data[a]
                if not (
                    (kind is None or e.kind == kind)
                    and (row is None or e.row == row)
                    and (col is None or e.col == col)
                )
            ]
            removed += before - len(self.data[a])
        return removed

    def get(self, area: str | None = None, kind: str | None = None) -> list[tuple[str, Landmark]]:
        out: list[tuple[str, Landmark]] = []
        areas = [area] if area is not None else sorted(self.data.keys())
        for a in areas:
            key = _clip(_clean_value(a) or "current", 48) if a is not None else a
            for e in self.data.get(key, []):
                if kind is None or e.kind == kind:
                    out.append((key, e))
        return out

    @staticmethod
    def _fmt(entries: list[tuple[str, Landmark]]) -> str:
        if not entries:
            return "(no matching landmarks)"
        lines = []
        for area, e in entries:
            pos = f" at ({e.row},{e.col})" if e.row is not None and e.col is not None else ""
            note = f" — {e.note}" if e.note else ""
            lines.append(f"  area {area}: {e.kind}{pos}{note}")
        return "\n".join(lines)

    def render(self) -> str:
        if not any(self.data.values()):
            return "(none saved)"
        lines: list[str] = []
        for area in sorted(self.data):
            if not self.data[area]:
                continue
            lines.append(f"Area {area}:")
            for e in self.data[area]:
                pos = f" ({e.row},{e.col})" if e.row is not None and e.col is not None else ""
                note = f" — {e.note}" if e.note else ""
                lines.append(f"  - {e.kind}{pos}{note}")
        return "\n".join(lines) if lines else "(none saved)"

    def count(self) -> int:
        return sum(len(v) for v in self.data.values())

    def to_dict(self) -> dict[str, list[dict]]:
        return {
            area: [
                {"kind": e.kind, "row": e.row, "col": e.col, "note": e.note}
                for e in entries
            ]
            for area, entries in self.data.items()
            if entries
        }


# ---------------------------------------------------------------------------
# Parsing a memory-turn response
# ---------------------------------------------------------------------------
@dataclass
class MemoryUpdate:
    parse_failed: bool = False
    updated: bool = False
    query_results: str = ""


def _extract_kv(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for m in re.finditer(r"(\w+)\s*=\s*([^,\n]+?)(?:\s*,|\s*$|\s+(?=\w+\s*=))", text):
        result[m.group(1).lower().strip()] = m.group(2).strip().rstrip(",")
    if not result:
        # "floor 0 type water row 20 col 30" fallback.
        tokens = text.replace(",", " ").split()
        keys = {"area", "floor", "level", "branch", "type", "kind", "name",
                "location", "row", "col", "r", "c", "x", "y"}
        i = 0
        while i < len(tokens) - 1:
            key = tokens[i].lower().strip(":")
            if key in keys:
                result[key] = tokens[i + 1].strip(",")
                i += 2
            else:
                i += 1
    # pos=(20,30)
    if "pos" in result and "row" not in result:
        parts = re.split(r"[,\s]+", result["pos"].strip("() "))
        if len(parts) >= 2:
            result.setdefault("row", parts[0])
            result.setdefault("col", parts[1])
    return result


def _area_from_kv(kv: dict[str, str], default: str | None = "current") -> str | None:
    area = kv.get("area") or kv.get("floor") or kv.get("level") or kv.get("branch")
    if area is None:
        return default
    return _clean_value(area) or default


def _kind_from_kv(kv: dict[str, str]) -> str:
    return _clean_value(
        kv.get("type") or kv.get("kind") or kv.get("location") or kv.get("name") or "unknown"
    )


def _apply_command_line(line: str, spatial: SpatialMemory, query_results: list[str]) -> bool:
    s = line.strip()
    low = s.lower()
    if not s:
        return False
    if low.startswith("map_get_all"):
        query_results.append("MAP_GET_ALL:\n" + SpatialMemory._fmt(spatial.get()))
        return True
    if low.startswith("map_get"):
        kv = _extract_kv(re.sub(r"^map_get\s*:?\s*", "", s, flags=re.IGNORECASE))
        area = _area_from_kv(kv, default=None)
        kind = _clean_value(kv.get("type") or kv.get("kind") or "") or None
        entries = spatial.get(area=area, kind=kind)
        label = []
        if area is not None:
            label.append(f"area={area}")
        if kind is not None:
            label.append(f"type={kind}")
        tag = f" ({', '.join(label)})" if label else ""
        query_results.append(f"MAP_GET{tag}:\n" + SpatialMemory._fmt(entries))
        return True
    if low.startswith(("map_add", "landmark_add", "landmark:")):
        payload = re.sub(r"^(map_add|landmark_add|landmark)\s*:?\s*", "", s, flags=re.IGNORECASE)
        kv = _extract_kv(payload)
        spatial.add(
            area=_area_from_kv(kv) or "current",
            kind=_kind_from_kv(kv),
            row=_optional_int(kv.get("row") or kv.get("r") or kv.get("x")),
            col=_optional_int(kv.get("col") or kv.get("c") or kv.get("y")),
            note=kv.get("note") or kv.get("desc") or "",
        )
        return True
    if low.startswith(("map_remove", "landmark_remove")):
        payload = re.sub(r"^(map_remove|landmark_remove)\s*:?\s*", "", s, flags=re.IGNORECASE)
        kv = _extract_kv(payload)
        n = spatial.remove(
            area=_area_from_kv(kv, default=None),
            kind=_clean_value(kv.get("type") or kv.get("kind") or "") or None,
            row=_optional_int(kv.get("row") or kv.get("r") or kv.get("x")),
            col=_optional_int(kv.get("col") or kv.get("c") or kv.get("y")),
        )
        if n:
            query_results.append(f"(removed {n} landmark{'s' if n != 1 else ''})")
        return True
    return False


def _extract_memory_body(text: str) -> tuple[str, bool]:
    """Return ``(body, clear_region)``.

    ``clear_region`` is True when the model delimited its memory update with a
    ``<memory>`` tag or a ``---`` block (a deliberate marker). When neither is
    present we still hand back the whole (think-stripped) text so explicit
    headers/commands can be salvaged, but ``clear_region`` is False so the
    caller treats it as a format failure and disables the free-form fallback.
    """
    raw = text or ""
    matches = _MEMORY_RE.findall(raw)
    if matches:
        return ("\n".join(m.strip() for m in matches), True)
    stripped = _THINK_RE.sub("", raw)
    delim = stripped.rfind("\n---")
    if delim == -1:
        delim = stripped.rfind("---")
    if delim != -1:
        return (stripped[delim:].lstrip("-").strip(), True)
    return (stripped.strip(), False)


def apply_memory(text: str, scratchpad: Scratchpad, spatial: SpatialMemory) -> MemoryUpdate:
    """Parse and apply one memory-turn response to both stores."""
    body, clear_region = _extract_memory_body(text)
    body = _ACTION_RE.sub("", body).strip()
    if not body:
        return MemoryUpdate(parse_failed=not clear_region, updated=False, query_results="")
    query_results: list[str] = []
    leftover: list[str] = []
    cmd_applied = False
    for line in body.splitlines():
        if _apply_command_line(line, spatial, query_results):
            cmd_applied = True
        else:
            leftover.append(line)
    section_updated = scratchpad.update_from_text(
        "\n".join(leftover).strip(), allow_freeform_fallback=clear_region
    )
    updated = cmd_applied or section_updated
    return MemoryUpdate(
        parse_failed=not clear_region,
        updated=updated,
        query_results="\n".join(query_results).strip(),
    )
