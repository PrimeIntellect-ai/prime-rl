"""Craftax LLM-first tutorial doc set + per-task composer.

Each .md file is a chapter that contains one or more named anchors marked
by `<!-- :section name -->` ... `<!-- :end -->` HTML comments. The composer
slices anchors per env. Bare chapter names expand to every anchor in that
chapter (in document order).
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from functools import cache
from pathlib import Path

_DOCS_DIR = Path(__file__).parent

_CHAPTERS: tuple[str, ...] = (
    "overview",
    "legend",
    "survival",
    "combat",
    "crafting",
    "magic",
    "items",
    "progression",
    "floors",
    "boss",
)

# Canonical anchor list (locked API). Renaming any entry is a breaking
# change across every craftax env that references it.
ALL_SECTIONS: tuple[str, ...] = (
    "overview",
    "legend:player",
    "legend:terrain",
    "legend:mobs:overworld",
    "legend:mobs:dungeon",
    "legend:mobs:boss",
    "legend:items",
    "legend:projectiles",
    "legend:hud",
    "survival:hp_food_drink",
    "survival:energy_sleep",
    "survival:rest",
    "survival:mana",
    "survival:day_night",
    "combat:melee",
    "combat:ranged_player",
    "combat:ranged_mob",
    "combat:armor",
    "combat:projectiles",
    "combat:elemental",
    "combat:sleep_penalty",
    "crafting:wood",
    "crafting:stone",
    "crafting:iron",
    "crafting:diamond",
    "crafting:placement",
    "crafting:arrows",
    "crafting:torches",
    "magic:books",
    "magic:spells",
    "magic:enchants",
    "items:resources",
    "items:gems",
    "items:potions",
    "items:bow",
    "progression:xp",
    "progression:attributes",
    "progression:achievements",
    "floors:0",
    "floors:1",
    "floors:2",
    "floors:3",
    "floors:4",
    "floors:5",
    "floors:6",
    "floors:7",
    "floors:8",
    "floors:navigation",
    "boss",
)

_ANCHOR_RE = re.compile(
    r"<!--\s*:section\s+([A-Za-z0-9_:]+)\s*-->(.*?)<!--\s*:end\s*-->",
    re.DOTALL,
)


def parse_anchors(text: str) -> dict[str, str]:
    """Parse `<!-- :section name -->...<!-- :end -->` blocks from `text`.

    Returns a dict mapping anchor name -> body (leading/trailing whitespace
    stripped). Raises ValueError on duplicate names or unclosed sections.
    """
    open_count = len(re.findall(r"<!--\s*:section\s+", text))
    close_count = len(re.findall(r"<!--\s*:end\s*-->", text))
    if open_count != close_count:
        raise ValueError(
            f"unclosed :section block (open={open_count}, close={close_count})"
        )

    out: dict[str, str] = {}
    for match in _ANCHOR_RE.finditer(text):
        name = match.group(1)
        body = match.group(2).strip()
        if name in out:
            raise ValueError(f"duplicate anchor {name!r}")
        out[name] = body
    return out


@cache
def _chapter_anchors(chapter: str) -> dict[str, str]:
    if chapter not in _CHAPTERS:
        raise ValueError(f"unknown chapter {chapter!r}")
    text = (_DOCS_DIR / f"{chapter}.md").read_text(encoding="utf-8")
    return parse_anchors(text)


@cache
def _chapter_order(chapter: str) -> tuple[str, ...]:
    """Return anchor names in the order they appear in the chapter file."""
    text = (_DOCS_DIR / f"{chapter}.md").read_text(encoding="utf-8")
    return tuple(m.group(1) for m in _ANCHOR_RE.finditer(text))


def _resolve(name: str) -> tuple[str, ...]:
    """Resolve a section spec to a tuple of anchor names.

    A bare chapter name expands to all anchors in that chapter.
    A `chapter:anchor` spec returns itself.
    An atomic anchor whose name equals its chapter (e.g. `overview`, `boss`)
    is also accepted directly.
    """
    if ":" not in name:
        if name in _CHAPTERS:
            order = _chapter_order(name)
            if not order:
                raise ValueError(
                    f"chapter {name!r} contains no anchors; cannot expand"
                )
            return order
        raise ValueError(f"unknown section {name!r}")
    chapter = name.split(":", 1)[0]
    if chapter not in _CHAPTERS:
        raise ValueError(f"unknown chapter {chapter!r} in section {name!r}")
    if name not in _chapter_anchors(chapter):
        raise ValueError(f"unknown anchor {name!r} in chapter {chapter!r}")
    return (name,)


def _anchor_chapter(name: str) -> str:
    """Return the chapter that owns this anchor."""
    if ":" in name:
        return name.split(":", 1)[0]
    if name in _CHAPTERS:
        return name
    raise ValueError(f"cannot determine chapter for anchor {name!r}")


def compose(sections: Iterable[str]) -> str:
    """Concatenate the requested anchors in caller-given order, joined by '\\n\\n'.

    Bare chapter names expand to every anchor in that chapter. Duplicate
    anchors (from overlapping specs) are emitted only once. Raises
    ValueError on unknown anchors (no silent fallback).
    """
    parts: list[str] = []
    seen: set[str] = set()
    for spec in sections:
        for name in _resolve(spec):
            if name in seen:
                continue
            seen.add(name)
            chapter = _anchor_chapter(name)
            body = _chapter_anchors(chapter)[name]
            parts.append(body)
    return "\n\n".join(parts)
