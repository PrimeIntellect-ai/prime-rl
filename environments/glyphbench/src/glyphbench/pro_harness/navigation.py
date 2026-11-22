"""Observation-grounded exploration memory for long-horizon grid play.

The tracker only consumes the same local text observation shown to the model.
It never reads the environment state or hidden map.  Its compact summary helps
an agent maintain coverage over thousands of turns without turning the harness
into a scripted policy: it reports where the agent has been, visible frontier
tiles, and last-seen durable features, but never selects an action.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass, field

_POS_RE = re.compile(r"Pos:\((-?\d+),\s*(-?\d+)\)")
_PLAYER = frozenset("@←→↑↓")
# Tiles known to be traversable from the public Craftax renderer semantics.
# Dynamic actors are included because their underlying tile is traversable
# once they move or are defeated.
_PASSABLE = frozenset("·░;:▼△@←→↑↓zGOLKTprcbskgWqadFiN")
_FEATURES = frozenset("▼△tf$⊙ⒺCID♦▲")
_FEATURE_NAMES = {
    "▼": "ladder_down", "△": "ladder_up", "t": "crafting_table",
    "f": "furnace", "$": "chest", "⊙": "fountain",
    "Ⓔ": "enchantment_table", "C": "coal", "I": "iron",
    "D": "diamond", "♦": "sapphire", "▲": "ruby",
}


@dataclass
class _AreaCoverage:
    seen: dict[tuple[int, int], str] = field(default_factory=dict)
    visits: dict[tuple[int, int], int] = field(default_factory=lambda: defaultdict(int))
    trail: deque[tuple[int, int]] = field(default_factory=lambda: deque(maxlen=24))
    current: tuple[int, int] | None = None


class ExplorationTracker:
    """Persistent visible-map coverage, partitioned by env area/floor."""

    def __init__(self) -> None:
        self.areas: dict[str, _AreaCoverage] = defaultdict(_AreaCoverage)

    def observe(self, area: str, obs_text: str) -> None:
        pos_match = _POS_RE.search(obs_text or "")
        if not pos_match:
            return
        pos = (int(pos_match.group(1)), int(pos_match.group(2)))
        grid_text = (obs_text or "").split("\n\n", 1)[0]
        rows = grid_text.splitlines()
        player_local: tuple[int, int] | None = None
        for rr, row in enumerate(rows):
            for cc, glyph in enumerate(row):
                if glyph in _PLAYER:
                    player_local = (rr, cc)
                    break
            if player_local is not None:
                break
        if player_local is None:
            return

        state = self.areas[str(area)]
        state.current = pos
        state.visits[pos] += 1
        state.trail.append(pos)
        pr, pc = player_local
        for rr, row in enumerate(rows):
            for cc, glyph in enumerate(row):
                # A blank is unobserved darkness, not evidence about a tile.
                if glyph == " ":
                    continue
                absolute = (pos[0] + rr - pr, pos[1] + cc - pc)
                # Store the underlying player tile as traversable ground.
                state.seen[absolute] = "·" if glyph in _PLAYER else glyph

    def render(self, area: str) -> str:
        state = self.areas.get(str(area))
        if state is None or state.current is None:
            return "(No coordinate-bearing observation has been mapped yet.)"
        current = state.current
        frontiers: list[tuple[int, int, int, int]] = []
        for (row, col), glyph in state.seen.items():
            if glyph not in _PASSABLE:
                continue
            unknown_neighbors = sum(
                (row + dr, col + dc) not in state.seen
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            )
            if unknown_neighbors:
                distance = abs(row - current[0]) + abs(col - current[1])
                frontiers.append((state.visits.get((row, col), 0), distance, row, col))
        frontiers.sort()

        features = [
            (_FEATURE_NAMES.get(glyph, glyph), row, col)
            for (row, col), glyph in state.seen.items()
            if glyph in _FEATURES
        ]
        features.sort(key=lambda item: abs(item[1] - current[0]) + abs(item[2] - current[1]))

        unique_trail: list[tuple[int, int]] = []
        for point in state.trail:
            if not unique_trail or point != unique_trail[-1]:
                unique_trail.append(point)
        trail = " -> ".join(f"({r},{c})" for r, c in unique_trail[-10:]) or "(none)"
        lines = [
            "Derived only from previously visible observations (no hidden map or policy).",
            f"Current {current}; visits here={state.visits[current]}; "
            f"unique positions={len(state.visits)}; visible tiles remembered={len(state.seen)}.",
            f"Recent route: {trail}",
        ]
        if frontiers:
            lines.append(
                "Nearest low-visit visible frontier tiles (each borders unseen space): "
                + ", ".join(
                    f"({r},{c}) d={dist} visits={visits}"
                    for visits, dist, r, c in frontiers[:6]
                )
            )
        else:
            lines.append("No visible frontier is currently known on this area.")
        if features:
            lines.append(
                "Last-seen durable features: "
                + ", ".join(f"{kind}@({r},{c})" for kind, r, c in features[:12])
            )
        lines.append(
            "Use this as coverage/navigation evidence; re-check the live observation because "
            "features and traversability can change."
        )
        return "\n".join(lines)
