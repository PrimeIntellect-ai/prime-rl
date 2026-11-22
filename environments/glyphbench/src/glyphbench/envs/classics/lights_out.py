"""Lights Out: toggle puzzle -- turn off all lights.

Gym IDs:
  glyphbench/classics-lightsout-easy-v0   (5x5)
  glyphbench/classics-lightsout-hard-v0   (7x7)
"""

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

SYM_ON = "\u25c9"    # ◉ light on
SYM_OFF = "\u25cb"   # ○ light off


def _make_lightsout_action_spec(size: int) -> ActionSpec:
    """Create an ActionSpec with PRESS_0 .. PRESS_(size*size-1).

    Indices are row-major: PRESS_i toggles cell (row=i // size, col=i % size).
    The previous (col, row) ordering was inconsistent with the env's _step
    and the system prompt — both use row-major.
    """
    n = size * size
    names = tuple(f"PRESS_{i}" for i in range(n))
    descriptions = tuple(
        f"toggle cell at row {i // size}, col {i % size} (and its 4-neighbours)"
        for i in range(n)
    )
    return ActionSpec(names=names, descriptions=descriptions)


# ---------------------------------------------------------------------------
# Base Lights Out Env
# ---------------------------------------------------------------------------


class _LightsOutBase(BaseGlyphEnv):
    """Lights Out: press cells to toggle them and neighbors. Turn all lights off."""

    noop_action_name: str = "PRESS_0"

    _grid_size: int = 5
    _difficulty: str = "easy"

    def __init__(self, max_turns: int = 100) -> None:
        self.action_spec = _make_lightsout_action_spec(self._grid_size)
        super().__init__(max_turns=max_turns)
        self._board: np.ndarray = np.zeros((self._grid_size, self._grid_size), dtype=np.int8)
        self._presses: int = 0

    def env_id(self) -> str:
        return f"glyphbench/classics-lightsout-{self._difficulty}-v0"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _toggle(self, row: int, col: int) -> None:
        """Flip cell (row, col) and its 4 orthogonal neighbors."""
        s = self._grid_size
        self._board[row][col] ^= 1
        if row > 0:
            self._board[row - 1][col] ^= 1
        if row < s - 1:
            self._board[row + 1][col] ^= 1
        if col > 0:
            self._board[row][col - 1] ^= 1
        if col < s - 1:
            self._board[row][col + 1] ^= 1

    def _generate_solvable(self) -> None:
        """Generate a solvable board by applying random presses to the solved state.

        Starting from all-off (solved), apply K random presses. Since each press
        is its own inverse, any sequence of presses can be undone.
        """
        s = self._grid_size
        self._board = np.zeros((s, s), dtype=np.int8)
        # Apply 5 to s*s random presses
        k = int(self.rng.integers(max(5, s), s * s + 1))
        cells = self.rng.integers(0, s * s, size=k)
        for c in cells:
            self._toggle(int(c) // s, int(c) % s)
        # If board happens to be already solved, press one more
        if not np.any(self._board):
            self._toggle(s // 2, s // 2)

    # ------------------------------------------------------------------
    # Reset / Step
    # ------------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        self._presses = 0
        self._generate_solvable()
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        s = self._grid_size

        idx = action  # PRESS_0 -> 0, etc.
        row, col = idx // s, idx % s
        self._toggle(row, col)
        self._presses += 1

        # Pattern A with bounded shaping: per-press penalty scaled so worst
        # case (no solve in entire budget) sums to -1.0; +1.0 on solve.
        reward = -1.0 / self.max_turns

        solved = not np.any(self._board)
        if solved:
            reward = 1.0

        info["lights_on"] = int(np.sum(self._board))
        info["presses"] = self._presses
        return self._render_current_observation(), reward, solved, False, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        s = self._grid_size
        grid = make_empty_grid(s, s, SYM_OFF)
        for r in range(s):
            for c in range(s):
                grid[r][c] = SYM_ON if self._board[r][c] else SYM_OFF

        legend = build_legend({
            SYM_ON: "light on",
            SYM_OFF: "light off",
        })

        lights_on = int(np.sum(self._board))
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Lights on: {lights_on} / {s * s}    "
            f"Presses: {self._presses}"
        )

        # Show cell indices for reference
        idx_lines = []
        for r in range(s):
            row_indices = " ".join(f"{r * s + c:2d}" for c in range(s))
            idx_lines.append(row_indices)
        cell_map = "Cell indices:\n" + "\n".join(idx_lines)

        return GridObservation(
            grid=grid_to_string(grid),
            legend=legend,
            hud=hud,
            message=cell_map,
        )

    def system_prompt(self) -> str:
        s = self._grid_size
        return (
            f"You are playing {self.env_id()}.\n\n"
            "TASK\n"
            "Turn off all the lights on the grid. Pressing a cell toggles it and "
            "its four orthogonal neighbors (up, down, left, right) between on and off.\n\n"
            "RULES\n"
            f"- The grid is {s}x{s} ({s * s} cells total).\n"
            "- Each press action is PRESS_N where N is the flattened cell index "
            "(row * width + col).\n"
            "- Toggling flips the pressed cell and its orthogonal neighbors.\n"
            "- The puzzle starts with a guaranteed-solvable configuration.\n"
            "- Reward: small per-press penalty (scaled so worst case sums to -1.0\n"
            "  if you exhaust the budget) and +1.0 when all lights are off.\n"
            "- The game ends when all lights are off or max steps is reached.\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Concrete variants
# ---------------------------------------------------------------------------


class LightsOutEasyEnv(_LightsOutBase):
    _grid_size = 5
    _difficulty = "easy"

    def __init__(self, max_turns: int = 100) -> None:
        super().__init__(max_turns=max_turns)


class LightsOutHardEnv(_LightsOutBase):
    _grid_size = 7
    _difficulty = "hard"

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

