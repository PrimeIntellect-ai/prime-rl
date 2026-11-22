"""Classic Minesweeper deduction game."""

from __future__ import annotations

from typing import Any

import numpy as np

from glyphbench.core.action import ActionSpec
from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.glyph_primitives import build_legend, grid_to_string, make_empty_grid
from glyphbench.core.observation import GridObservation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ROWS = 8
_COLS = 8
_NUM_MINES = 10
_TOTAL_CELLS = _ROWS * _COLS

_SYM_HIDDEN = "\u2592"  # ▒
_SYM_EMPTY = "\u00b7"   # ·
_SYM_MINE = "\u2731"     # ✱
_DIGIT_CHARS = " 12345678"  # index 0 unused, 1-8 used

# ---------------------------------------------------------------------------
# Action spec: REVEAL_0 .. REVEAL_63
# ---------------------------------------------------------------------------

_action_names = tuple(f"REVEAL_{i}" for i in range(_TOTAL_CELLS))
_action_descs = tuple(
    f"reveal cell at row {i // _COLS}, col {i % _COLS}" for i in range(_TOTAL_CELLS)
)

MINESWEEPER_ACTION_SPEC = ActionSpec(names=_action_names, descriptions=_action_descs)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class MinesweeperEnv(BaseGlyphEnv):
    """Classic Minesweeper on an 8x8 grid with 10 mines."""

    action_spec = MINESWEEPER_ACTION_SPEC
    noop_action_name = "REVEAL_0"  # no true noop; re-revealing is harmless

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)
        # Mine locations (boolean grid)
        self._mines: np.ndarray = np.zeros((_ROWS, _COLS), dtype=bool)
        # Neighbor counts
        self._counts: np.ndarray = np.zeros((_ROWS, _COLS), dtype=np.int8)
        # Revealed cells
        self._revealed: np.ndarray = np.zeros((_ROWS, _COLS), dtype=bool)
        self._dead: bool = False
        self._won: bool = False
        self._safe_total: int = _TOTAL_CELLS - _NUM_MINES

    def env_id(self) -> str:
        return "glyphbench/classics-minesweeper-v0"

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _place_mines(self) -> None:
        self._mines[:] = False
        indices = self.rng.choice(_TOTAL_CELLS, size=_NUM_MINES, replace=False)
        for idx in indices:
            r, c = divmod(int(idx), _COLS)
            self._mines[r, c] = True

    def _compute_counts(self) -> None:
        self._counts[:] = 0
        for r in range(_ROWS):
            for c in range(_COLS):
                if self._mines[r, c]:
                    continue
                cnt = 0
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < _ROWS and 0 <= nc < _COLS and self._mines[nr, nc]:
                            cnt += 1
                self._counts[r, c] = cnt

    def _make_first_reveal_safe(self, r: int, c: int) -> None:
        """Move a mine away from the first revealed cell if needed."""
        if not self._mines[r, c]:
            return
        self._mines[r, c] = False
        for idx in range(_TOTAL_CELLS):
            nr, nc = divmod(idx, _COLS)
            if (nr, nc) != (r, c) and not self._mines[nr, nc]:
                self._mines[nr, nc] = True
                self._compute_counts()
                return
        self._compute_counts()

    def _flood_reveal(self, r: int, c: int) -> int:
        """Flood-fill reveal from (r,c). Returns number of newly revealed cells."""
        if self._revealed[r, c]:
            return 0
        stack = [(r, c)]
        count = 0
        while stack:
            cr, cc = stack.pop()
            if self._revealed[cr, cc]:
                continue
            self._revealed[cr, cc] = True
            count += 1
            if self._counts[cr, cc] == 0:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < _ROWS and 0 <= nc < _COLS and not self._revealed[nr, nc] and not self._mines[nr, nc]:
                            stack.append((nr, nc))
        return count

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        self._dead = False
        self._won = False
        self._revealed[:] = False
        self._place_mines()
        self._compute_counts()
        self._safe_total = _TOTAL_CELLS - _NUM_MINES
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        r, c = divmod(action, _COLS)

        # Already revealed -- noop
        if self._revealed[r, c]:
            return self._render_current_observation(), 0.0, False, False, info

        # First reveal protection removes pure-luck opening deaths while
        # preserving the total number of mines.
        if not np.any(self._revealed):
            self._make_first_reveal_safe(r, c)

        # Hit a mine
        if self._mines[r, c]:
            self._dead = True
            self._revealed[r, c] = True
            info["mine_hit"] = True
            return self._render_current_observation(), -1.0, True, False, info

        # Safe reveal (with flood fill for 0-count cells). Pattern A: each
        # newly revealed safe cell = 1/_safe_total so cumulative reward =
        # 1.0 on full clear. No additional win bonus needed.
        newly = self._flood_reveal(r, c)
        reward = float(newly) / max(1, self._safe_total)

        # Check win
        revealed_count = int(np.sum(self._revealed))
        if revealed_count >= self._safe_total:
            self._won = True
            info["win"] = True
            return self._render_current_observation(), reward, True, False, info

        return self._render_current_observation(), reward, False, False, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        grid = make_empty_grid(_COLS, _ROWS, fill=_SYM_HIDDEN)
        syms: dict[str, str] = {
            _SYM_HIDDEN: "hidden cell",
            _SYM_EMPTY: "empty (0 mine neighbors)",
        }

        for r in range(_ROWS):
            for c in range(_COLS):
                if not self._revealed[r, c]:
                    continue
                if self._mines[r, c]:
                    grid[r][c] = _SYM_MINE
                    syms[_SYM_MINE] = "mine (boom!)"
                elif self._counts[r, c] == 0:
                    grid[r][c] = _SYM_EMPTY
                else:
                    ch = str(self._counts[r, c])
                    grid[r][c] = ch
                    syms[ch] = f"{ch} adjacent mines"

        legend = build_legend(syms)
        revealed_count = int(np.sum(self._revealed))
        mines_remaining = _NUM_MINES
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Revealed: {revealed_count} / {self._safe_total}    "
            f"Total mines: {mines_remaining}"
        )
        msg = ""
        if self._dead:
            msg = "You hit a mine! Game over."
        elif self._won:
            msg = "All safe cells revealed! You win!"

        return GridObservation(
            grid=grid_to_string(grid), legend=legend, hud=hud, message=msg
        )

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def system_prompt(self) -> str:
        return (
            f"You are playing {self.env_id()} -- classic Minesweeper.\n\n"
            "RULES\n"
            f"The board is {_ROWS}x{_COLS} with {_NUM_MINES} hidden mines.\n"
            "Each turn you reveal one cell by index (row*8+col).\n"
            "Your first reveal is guaranteed safe; if it was mined, that mine relocates before counts are revealed.\n"
            "If the cell is a mine, you die (-1 reward).\n"
            "If the cell has 0 adjacent mines, all connected 0-cells and their "
            "numbered borders are auto-revealed (flood fill).\n"
            "Reward is normalized so revealing all safe cells gives cumulative\n"
            "reward = 1.0 (each cell yields 1/safe_total).\n\n"
            "STRATEGY\n"
            "Use the numbers to deduce mine locations. A number N means exactly "
            "N of the 8 surrounding cells contain mines.\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
