"""Memory Match -- card-pair matching game with shape+colour binding.

Gym IDs:
  glyphbench/classics-memorymatch-easy-v0   (5x4 = 10 pairs)
  glyphbench/classics-memorymatch-hard-v0   (6x6 = 18 pairs)

Phase 3 audit fix (2026-05-03): mechanic A5 (bind-and-recall). Each card has
a shape AND a colour attribute; matches require BOTH attributes to agree.
Cards display a single-codepoint glyph that uniquely encodes the (shape,
colour) tuple; the agent must remember conjunctive features to find matches
efficiently. Mismatched flips still count toward the turn budget.
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.glyph_primitives import build_legend, grid_to_string, make_empty_grid
from glyphbench.core.observation import GridObservation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYM_FACE_DOWN = "▒"  # ▒

# Single-codepoint glyphs encoding (shape, colour). Indexing:
#   glyph = _ATTR_GLYPHS[shape_idx][colour_idx]
# Up to 5 shapes x 4 colours = 20 attribute combos -> covers easy/hard sizes.
_ATTR_GLYPHS: tuple[tuple[str, ...], ...] = (
    # 0: circle family
    ("○", "●", "◐", "◑"),  # ○ ● ◐ ◑
    # 1: square family
    ("□", "■", "▤", "▥"),  # □ ■ ▤ ▥
    # 2: triangle family
    ("△", "▲", "▶", "▷"),  # △ ▲ ▶ ▷
    # 3: diamond family
    ("◇", "◆", "◉", "◊"),  # ◇ ◆ ◉ ◊
    # 4: star / cross family
    ("☆", "★", "✚", "✝"),  # ☆ ★ ✚ ✝
)

_SHAPE_NAMES: tuple[str, ...] = ("circle", "square", "triangle", "diamond", "star")
_COLOUR_NAMES: tuple[str, ...] = ("hollow", "filled", "shaded-A", "shaded-B")

# Legacy alphabet kept available for other callers; not used in matching.
PAIR_LETTERS = tuple("ABCDEFGHIJKLMNOPQR")

# ---------------------------------------------------------------------------
# Build action spec
# ---------------------------------------------------------------------------


def _build_action_spec(total_cells: int) -> ActionSpec:
    names: list[str] = []
    descs: list[str] = []
    for i in range(total_cells):
        names.append(f"FLIP_{i}")
        descs.append(f"flip card at position {i}")
    return ActionSpec(names=tuple(names), descriptions=tuple(descs))


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class _MemoryMatchBase(BaseGlyphEnv):
    """Flip two cards per turn; matches require shape AND colour to agree."""

    noop_action_name: str = "FLIP_0"

    _rows: int = 4
    _cols: int = 5
    _difficulty: str = "easy"
    # Number of distinct shape families and colours used to build cards.
    _N_SHAPES: int = 5
    _N_COLOURS: int = 4

    def __init__(self, max_turns: int = 32) -> None:
        self._total_cells = self._rows * self._cols
        self._num_pairs = self._total_cells // 2
        self.action_spec = _build_action_spec(self._total_cells)
        super().__init__(max_turns=max_turns)
        # Per-card attributes.
        self._shapes: list[int] = []
        self._colours: list[int] = []
        self._matched: list[bool] = []
        self._first_flip: int | None = None
        self._second_flip: int | None = None
        self._on_first_flip: bool = True
        self._pairs_found: int = 0
        self._total_reward: float = 0.0
        self._match_msg: str = ""

    def env_id(self) -> str:
        return f"glyphbench/classics-memorymatch-{self._difficulty}-v0"

    def _reset(self, seed: int) -> GridObservation:
        # Sample _num_pairs distinct (shape, colour) attribute combos.
        all_combos = [
            (s, c)
            for s in range(self._N_SHAPES)
            for c in range(self._N_COLOURS)
        ]
        if len(all_combos) < self._num_pairs:
            raise RuntimeError(
                f"Not enough (shape, colour) combos for {self._num_pairs} pairs"
            )
        idx = list(range(len(all_combos)))
        self.rng.shuffle(idx)
        chosen_combos = [all_combos[i] for i in idx[: self._num_pairs]]

        # Build the two-of-each list and shuffle for placement.
        cards: list[tuple[int, int]] = []
        for s, c in chosen_combos:
            cards.append((s, c))
            cards.append((s, c))
        order = list(range(len(cards)))
        self.rng.shuffle(order)

        self._shapes = [0] * self._total_cells
        self._colours = [0] * self._total_cells
        for slot, src in enumerate(order):
            s, c = cards[src]
            self._shapes[slot] = s
            self._colours[slot] = c

        self._matched = [False] * self._total_cells
        self._first_flip = None
        self._second_flip = None
        self._on_first_flip = True
        self._pairs_found = 0
        self._total_reward = 0.0
        self._match_msg = ""
        return self._render_current_observation()

    def _glyph_at(self, idx: int) -> str:
        return _ATTR_GLYPHS[self._shapes[idx]][self._colours[idx]]

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        idx = action

        if idx >= self._total_cells or self._matched[idx]:
            return self._render_current_observation(), 0.0, False, False, info

        if self._on_first_flip:
            self._first_flip = idx
            self._second_flip = None
            self._on_first_flip = False
            return self._render_current_observation(), 0.0, False, False, info

        # Second flip.
        if idx == self._first_flip:
            return self._render_current_observation(), 0.0, False, False, info

        self._second_flip = idx
        reward = 0.0
        # A5 match: BOTH shape and colour must agree.
        first = self._first_flip
        assert first is not None
        match_shape = self._shapes[first] == self._shapes[idx]
        match_colour = self._colours[first] == self._colours[idx]
        matched = match_shape and match_colour

        if matched:
            self._matched[first] = True
            self._matched[idx] = True
            self._pairs_found += 1
            reward = 1.0 / self._num_pairs

        glyph_first = self._glyph_at(first)
        glyph_second = self._glyph_at(idx)
        if matched:
            self._match_msg = (
                f"Match! Both cards show {glyph_first}."
            )
        else:
            partial = (
                "(shape match only)"
                if match_shape
                else "(colour match only)" if match_colour
                else "(both differ)"
            )
            self._match_msg = (
                f"No match: {glyph_first} vs {glyph_second} {partial}. "
                "Cards flip back."
            )

        obs = self._render_current_observation()
        self._match_msg = ""
        self._first_flip = None
        self._second_flip = None
        self._on_first_flip = True

        terminated = self._pairs_found == self._num_pairs
        self._total_reward += reward
        info["pairs_found"] = self._pairs_found
        return obs, reward, terminated, False, info

    def _render_current_observation(self) -> GridObservation:
        grid = make_empty_grid(self._cols, self._rows, SYM_FACE_DOWN)
        for i in range(self._total_cells):
            r = i // self._cols
            c = i % self._cols
            if self._matched[i] or i == self._first_flip or i == self._second_flip:
                grid[r][c] = self._glyph_at(i)

        legend_map: dict[str, str] = {
            SYM_FACE_DOWN: "face-down card",
        }
        # Build legend for any (shape, colour) glyph currently in play.
        seen: set[tuple[int, int]] = set()
        for i in range(self._total_cells):
            key = (self._shapes[i], self._colours[i])
            if key in seen:
                continue
            seen.add(key)
            glyph = _ATTR_GLYPHS[key[0]][key[1]]
            legend_map[glyph] = (
                f"{_COLOUR_NAMES[key[1]]} {_SHAPE_NAMES[key[0]]}"
            )

        flip_phase = "first" if self._on_first_flip else "second"
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Pairs found: {self._pairs_found} / {self._num_pairs}    "
            f"Flip: {flip_phase}"
        )

        msg = ""
        if self._pairs_found == self._num_pairs:
            msg = "All pairs matched! You win!"
        elif self._match_msg:
            msg = self._match_msg
        elif not self._on_first_flip and self._first_flip is not None:
            msg = (
                f"Card at position {self._first_flip} shows "
                f"{self._glyph_at(self._first_flip)}. Now pick the second card."
            )

        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(legend_map),
            hud=hud,
            message=msg,
        )

    def system_prompt(self) -> str:
        per_pair_reward = 1.0 / self._num_pairs
        return (
            f"You are playing {self.env_id()}.\n\n"
            "TASK\n"
            "Find all matching pairs of cards by flipping two cards per turn.\n\n"
            "RULES\n"
            f"- The board is {self._rows}x{self._cols} with {self._num_pairs} "
            "pairs of cards.\n"
            "- Each card has a SHAPE attribute (circle, square, triangle, "
            "diamond, star) AND a COLOUR attribute (hollow, filled, shaded-A, "
            "shaded-B). The cell glyph encodes both attributes; consult the "
            "legend for each glyph's meaning.\n"
            "- The legend lists every card glyph in the deck from turn 0, but "
            "hidden card positions still have to be discovered by flipping.\n"
            "- Two cards form a matching PAIR only if their SHAPE and COLOUR "
            "BOTH agree (bind-and-recall: half-matches do not score).\n"
            "- Each step flips one card. Two consecutive flips form a turn: "
            "the first reveals one card; the second reveals the other and "
            "decides whether they match.\n"
            f"- Matched pairs stay face up permanently and pay "
            f"+{per_pair_reward:.4f} reward; mismatches flip both cards back "
            "and consume the steps.\n"
            "- Cards are numbered 0 to "
            f"{self._total_cells - 1} (left-to-right, top-to-bottom).\n"
            "- Flipping an already-matched card or the same card twice is a "
            "no-op (no reward, but still counts as a step).\n"
            "- Cumulative reward = 1.0 on finding all pairs.\n"
            "- Remember (shape, colour) of every card you've seen.\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Concrete variants
# ---------------------------------------------------------------------------


class MemoryMatchEasyEnv(_MemoryMatchBase):
    _rows = 4
    _cols = 5
    _difficulty = "easy"
    _N_SHAPES = 5
    _N_COLOURS = 4

    def __init__(self, max_turns: int = 32) -> None:
        super().__init__(max_turns=max_turns)


class MemoryMatchHardEnv(_MemoryMatchBase):
    _rows = 6
    _cols = 6
    _difficulty = "hard"
    _N_SHAPES = 5
    _N_COLOURS = 4

    def __init__(self, max_turns: int = 60) -> None:
        super().__init__(max_turns=max_turns)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
