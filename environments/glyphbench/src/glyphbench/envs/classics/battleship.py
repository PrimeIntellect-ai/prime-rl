"""Classic Battleship: guess ship locations on a hidden grid."""

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

_SIZE = 10
_TOTAL_CELLS = _SIZE * _SIZE

# F1 Mastermind-feedback budget: every shot costs a small penalty so random
# scattershot play exceeds the negative band; the 35-turn cap forces deduction
# from hit/miss results rather than blanket coverage of the grid.
_STEP_PENALTY = 0.005

_SYM_UNKNOWN = "\u2592"  # ▒
_SYM_MISS = "\u00b7"     # ·
_SYM_HIT = "\u2715"      # ✕
_SYM_SUNK = "\u25a3"     # ▣

# Ships: (name, length)
_SHIPS = [
    ("carrier", 5),
    ("battleship", 4),
    ("cruiser", 3),
    ("submarine", 3),
    ("destroyer", 2),
]

# Total cells across all ships -- used for [-1, 1] reward normalization.
_TOTAL_SHIP_CELLS = sum(length for _, length in _SHIPS)

# ---------------------------------------------------------------------------
# Action spec: FIRE_0 .. FIRE_99
# ---------------------------------------------------------------------------

_action_names = tuple(f"FIRE_{i}" for i in range(_TOTAL_CELLS))
_action_descs = tuple(
    f"fire at row {i // _SIZE}, col {i % _SIZE}" for i in range(_TOTAL_CELLS)
)

BATTLESHIP_ACTION_SPEC = ActionSpec(names=_action_names, descriptions=_action_descs)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class BattleshipEnv(BaseGlyphEnv):
    """Classic Battleship on a 10x10 grid."""

    action_spec = BATTLESHIP_ACTION_SPEC
    noop_action_name = "FIRE_0"  # re-firing acts as noop

    def __init__(self, max_turns: int = 35) -> None:
        super().__init__(max_turns=max_turns)
        # Hidden ship grid: -1 = empty, 0..4 = ship index
        self._ship_grid: np.ndarray = np.full((_SIZE, _SIZE), -1, dtype=np.int8)
        # Visible state: 0=unknown, 1=miss, 2=hit, 3=sunk
        self._visible: np.ndarray = np.zeros((_SIZE, _SIZE), dtype=np.int8)
        # Ship health: remaining hits per ship
        self._ship_health: list[int] = []
        self._ship_cells: list[list[tuple[int, int]]] = []
        self._all_sunk: bool = False
        self._total_hits: int = 0
        self._total_misses: int = 0
        self._ships_sunk: int = 0
        self._last_msg: str = ""

    def env_id(self) -> str:
        return "glyphbench/classics-battleship-v0"

    # ------------------------------------------------------------------
    # Ship placement
    # ------------------------------------------------------------------

    def _place_ships(self) -> None:
        """Randomly place all ships on the grid, no overlap."""
        self._ship_grid[:] = -1
        self._ship_health = []
        self._ship_cells = []

        for ship_idx, (name, length) in enumerate(_SHIPS):
            placed = False
            for _ in range(1000):  # safety limit
                horizontal = bool(self.rng.integers(2))
                if horizontal:
                    r = int(self.rng.integers(_SIZE))
                    c = int(self.rng.integers(_SIZE - length + 1))
                    cells = [(r, c + k) for k in range(length)]
                else:
                    r = int(self.rng.integers(_SIZE - length + 1))
                    c = int(self.rng.integers(_SIZE))
                    cells = [(r + k, c) for k in range(length)]

                # Check overlap
                if all(self._ship_grid[cr, cc] == -1 for cr, cc in cells):
                    for cr, cc in cells:
                        self._ship_grid[cr, cc] = ship_idx
                    self._ship_health.append(length)
                    self._ship_cells.append(cells)
                    placed = True
                    break

            if not placed:
                raise RuntimeError(f"Failed to place {name} after 1000 attempts")

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        self._visible[:] = 0
        self._all_sunk = False
        self._total_hits = 0
        self._total_misses = 0
        self._ships_sunk = 0
        self._last_msg = ""
        self._place_ships()
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        r, c = divmod(action, _SIZE)
        reward = 0.0

        # Already fired here -- still costs a step (Mastermind feedback budget).
        if self._visible[r, c] != 0:
            self._last_msg = (
                f"Already fired at ({r},{c}). No new info, still spent a shot."
            )
            return (
                self._render_current_observation(),
                -_STEP_PENALTY,
                False,
                False,
                info,
            )

        ship_idx = self._ship_grid[r, c]

        # Per-shot cost: every fired round subtracts -0.005 (F1).
        if ship_idx == -1:
            # Miss -- only the step penalty.
            self._visible[r, c] = 1
            self._total_misses += 1
            reward = -_STEP_PENALTY
            self._last_msg = f"Miss at ({r},{c})."
        else:
            # Hit: +1/_TOTAL_SHIP_CELLS (cumulative cap = 1.0 on full sink),
            # less the step penalty.
            self._visible[r, c] = 2
            self._total_hits += 1
            self._ship_health[ship_idx] -= 1
            reward = (1.0 / _TOTAL_SHIP_CELLS) - _STEP_PENALTY
            ship_name = _SHIPS[ship_idx][0]

            if self._ship_health[ship_idx] == 0:
                # Ship sunk -- mark all cells as sunk
                self._ships_sunk += 1
                for sr, sc in self._ship_cells[ship_idx]:
                    self._visible[sr, sc] = 3
                self._last_msg = f"Hit and sunk the {ship_name}!"

                # Check all sunk
                if self._ships_sunk == len(_SHIPS):
                    self._all_sunk = True
                    info["all_sunk"] = True
                    self._last_msg += " All ships sunk! You win!"
            else:
                self._last_msg = f"Hit at ({r},{c})!"

        info["hits"] = self._total_hits
        info["misses"] = self._total_misses
        info["ships_sunk"] = self._ships_sunk

        return (
            self._render_current_observation(),
            reward,
            self._all_sunk,
            False,
            info,
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        grid = make_empty_grid(_SIZE, _SIZE, fill=_SYM_UNKNOWN)
        syms: dict[str, str] = {
            _SYM_UNKNOWN: "unknown (not fired)",
            _SYM_MISS: "miss",
            _SYM_HIT: "hit",
            _SYM_SUNK: "sunk ship segment",
        }

        state_to_sym = {0: _SYM_UNKNOWN, 1: _SYM_MISS, 2: _SYM_HIT, 3: _SYM_SUNK}

        for r in range(_SIZE):
            for c in range(_SIZE):
                grid[r][c] = state_to_sym[int(self._visible[r, c])]

        legend = build_legend(syms)

        # Ship status
        ship_status_parts: list[str] = []
        for idx, (name, length) in enumerate(_SHIPS):
            hp = self._ship_health[idx]
            if hp == 0:
                ship_status_parts.append(f"{name}({length}): SUNK")
            else:
                ship_status_parts.append(f"{name}({length}): {hp}/{length}")
        ship_status = "  ".join(ship_status_parts)

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Hits: {self._total_hits}    Misses: {self._total_misses}    "
            f"Ships sunk: {self._ships_sunk}/{len(_SHIPS)}\n"
            f"Ships: {ship_status}"
        )

        return GridObservation(
            grid=grid_to_string(grid), legend=legend, hud=hud, message=self._last_msg
        )

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def system_prompt(self) -> str:
        ship_list = ", ".join(f"{name}({length})" for name, length in _SHIPS)
        return (
            f"You are playing {self.env_id()} -- classic Battleship.\n\n"
            "RULES\n"
            f"The opponent has placed ships on a {_SIZE}x{_SIZE} grid.\n"
            f"Ships: {ship_list}. No overlaps, placed horizontally or vertically.\n"
            "Each turn, fire at one cell. You will learn if it is a hit or miss.\n"
            "When all cells of a ship are hit, it sinks and is marked.\n"
            "The HUD shows current HP for each ship type after hits.\n"
            f"You have only {self.max_turns} shots — far fewer than the {_TOTAL_CELLS} "
            "cells on the grid, so blanket scanning cannot work.\n\n"
            "SCORING\n"
            f"  +{1.0 / _TOTAL_SHIP_CELLS:.4f} per hit ship cell before shot penalty\n"
            f"  -{_STEP_PENALTY} flat penalty on EVERY fired shot (hit, miss, or repeated)\n"
            "  A perfect 17-hit fleet clear scores about 0.915 after penalties.\n"
            "  Game ends when all ships are sunk or your shot budget is exhausted.\n"
            "  Firing at an already-fired cell still spends a shot and costs the penalty.\n\n"
            "STRATEGY\n"
            "Start by sampling spread-out cells to scan the grid. When you get a hit, "
            "probe adjacent cells to find the ship orientation, then follow it to sink "
            "the ship before wasting shots elsewhere. Deduce ship locations from prior "
            "miss patterns to skip cells that cannot host a ship.\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
