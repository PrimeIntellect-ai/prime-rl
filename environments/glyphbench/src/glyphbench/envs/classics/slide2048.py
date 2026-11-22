"""2048 sliding tile game (short-horizon variant).

Gym IDs:
  glyphbench/classics-2048-v0

Reward shape (Phase 3 audit fix, 2026-05-03): K5 (conditional bonus).
A new max-tile milestone in {4, 8, 16, 32, 64} fires +0.2 reward only when
the max-tile is in a CORNER cell after the slide. Cumulative reward caps at
1.0 on full win. The corner constraint enforces classic 2048 strategy
(anchor max-tile in a corner) and prevents random spawn-pressure merges
from saturating the milestone reward.

Design:
  - 4x4 board (classic). Target tile = 64. Reaching it wins.
  - Reward = +0.2 each time a NEW max-tile milestone in {4, 8, 16, 32, 64}
    is reached for the first time AND that max-tile sits in one of the four
    corner cells. Cumulative reward caps at 1.0 (5 * 0.2).
  - max_turns = 200.
  - No-legal-moves loss terminates with whatever reward was earned.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from glyphbench.core.action import ActionSpec
from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.glyph_primitives import build_legend
from glyphbench.core.observation import GridObservation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIZE = 4
TARGET_TILE = 64
DEFAULT_MAX_TURNS = 200

SLIDE_ACTION_SPEC = ActionSpec(
    names=("UP", "DOWN", "LEFT", "RIGHT"),
    descriptions=(
        "slide all tiles up",
        "slide all tiles down",
        "slide all tiles left",
        "slide all tiles right",
    ),
)

SYM_EMPTY = "·"  # ·

# Tile display: values -> symbol string. With a 3x3 / target-64 board the
# only commonly seen values are {2, 4, 8, 16, 32, 64}; the larger entries
# are kept as a fallback in case the player overshoots.
_TILE_SYMS: dict[int, str] = {
    0: SYM_EMPTY,
    2: "2",
    4: "4",
    8: "8",
    16: "16",
    32: "32",
    64: "64",
    128: "128",
    256: "256",
    512: "512",
    1024: "K",
    2048: "W",
}


def _tile_sym(val: int) -> str:
    if val == 0:
        return SYM_EMPTY
    if val in _TILE_SYMS:
        return _TILE_SYMS[val]
    if val >= 2048:
        return "W"
    if val >= 1024:
        return "K"
    return str(val)


# ---------------------------------------------------------------------------
# 2048 Env
# ---------------------------------------------------------------------------


class Slide2048Env(BaseGlyphEnv):
    """2048: slide tiles to merge and reach the target tile (64).

    Short-horizon variant on a 3x3 board. Reward is shaped: +0.2 per new
    max-tile milestone (4, 8, 16, 32, 64). Cumulative reward caps at 1.0.
    """

    action_spec = SLIDE_ACTION_SPEC
    noop_action_name: str = "UP"

    # Tile values that grant a milestone reward the first time they appear in
    # the anchor corner. Rewards are weighted toward the higher tiles so that
    # random play (which can occasionally merge an 8 into a corner) is bounded
    # well below the cap of 1.0; reaching 32 and 64 — which require disciplined
    # anchoring — is what produces most of the cumulative return.
    _MILESTONES: tuple[int, ...] = (8, 16, 32, 64)
    _MILESTONE_REWARDS: tuple[float, ...] = (0.05, 0.15, 0.30, 0.50)

    def __init__(self, max_turns: int = DEFAULT_MAX_TURNS) -> None:
        super().__init__(max_turns=max_turns)
        self._board: np.ndarray = np.zeros((SIZE, SIZE), dtype=np.int32)
        self._score: int = 0
        self._max_tile: int = 0
        self._milestones_hit: set[int] = set()
        # Tracks the corner used to earn the FIRST milestone. All subsequent
        # milestones must be earned with the max-tile in the same corner.
        self._anchor_corner: tuple[int, int] | None = None

    def env_id(self) -> str:
        return "glyphbench/classics-2048-v0"

    # ------------------------------------------------------------------
    # Board helpers
    # ------------------------------------------------------------------

    def _spawn_tile(self) -> None:
        """Spawn a new 2 or 4 tile on a random empty cell."""
        empty = list(zip(*np.where(self._board == 0), strict=False))
        if not empty:
            return
        idx = int(self.rng.integers(0, len(empty)))
        r, c = empty[idx]
        self._board[r][c] = 2 if self.rng.random() < 0.9 else 4

    @staticmethod
    def _slide_row_left(row: np.ndarray) -> tuple[np.ndarray, int, int]:
        """Slide a single row left and merge.

        Returns (new_row, merge_score, num_merges) where merge_score sums the
        merged tile values (used for HUD score) and num_merges counts merge
        events.
        """
        # Remove zeros
        tiles = row[row != 0].tolist()
        merged: list[int] = []
        score = 0
        num_merges = 0
        skip = False
        for i in range(len(tiles)):
            if skip:
                skip = False
                continue
            if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
                val = tiles[i] * 2
                merged.append(val)
                score += val
                num_merges += 1
                skip = True
            else:
                merged.append(tiles[i])
        # Pad with zeros
        merged.extend([0] * (SIZE - len(merged)))
        return np.array(merged, dtype=np.int32), score, num_merges

    def _slide(self, direction: str) -> tuple[bool, int, int]:
        """Slide the board in the given direction.

        Returns (changed, merge_score, num_merges).
        """
        board = self._board.copy()
        total_score = 0
        total_merges = 0

        if direction == "LEFT":
            for r in range(SIZE):
                self._board[r], s, m = self._slide_row_left(self._board[r])
                total_score += s
                total_merges += m
        elif direction == "RIGHT":
            for r in range(SIZE):
                self._board[r] = self._board[r][::-1]
                self._board[r], s, m = self._slide_row_left(self._board[r])
                self._board[r] = self._board[r][::-1]
                total_score += s
                total_merges += m
        elif direction == "UP":
            for c in range(SIZE):
                col = self._board[:, c].copy()
                self._board[:, c], s, m = self._slide_row_left(col)
                total_score += s
                total_merges += m
        elif direction == "DOWN":
            for c in range(SIZE):
                col = self._board[:, c][::-1].copy()
                new_col, s, m = self._slide_row_left(col)
                self._board[:, c] = new_col[::-1]
                total_score += s
                total_merges += m

        changed = not np.array_equal(board, self._board)
        return changed, total_score, total_merges

    _CORNERS: tuple[tuple[int, int], ...] = (
        (0, 0),
        (0, SIZE - 1),
        (SIZE - 1, 0),
        (SIZE - 1, SIZE - 1),
    )

    def _max_tile_corners(self) -> list[tuple[int, int]]:
        """Corners that hold the current max-tile value."""
        if self._max_tile <= 0:
            return []
        return [
            (r, c)
            for (r, c) in self._CORNERS
            if int(self._board[r, c]) == self._max_tile
        ]

    def _has_valid_moves(self) -> bool:
        """Check if any move can change the board."""
        if np.any(self._board == 0):
            return True
        # Check horizontal merges
        for r in range(SIZE):
            for c in range(SIZE - 1):
                if self._board[r][c] == self._board[r][c + 1]:
                    return True
        # Check vertical merges
        for r in range(SIZE - 1):
            for c in range(SIZE):
                if self._board[r][c] == self._board[r + 1][c]:
                    return True
        return False

    # ------------------------------------------------------------------
    # Reset / Step
    # ------------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        self._board = np.zeros((SIZE, SIZE), dtype=np.int32)
        self._score = 0
        self._max_tile = 0
        self._milestones_hit = set()
        self._anchor_corner = None
        self._spawn_tile()
        self._spawn_tile()
        self._max_tile = int(np.max(self._board))
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        name = self.action_spec.names[action]

        changed, merge_score, _num_merges = self._slide(name)

        if changed:
            self._spawn_tile()

        self._score += merge_score
        self._max_tile = int(np.max(self._board))

        # K5 (conditional bonus): a milestone fires only when the post-slide
        # max-tile is UNIQUE on the board AND sits in the anchor corner. The
        # first milestone may pick any corner, locking in the anchor; later
        # milestones must keep the unique max-tile in that exact corner. The
        # uniqueness constraint stops random play from collecting milestones
        # via the common pattern where two equal max-tiles sit on the board
        # and one of them happens to occupy a corner.
        reward = 0.0
        max_count = int(np.count_nonzero(self._board == self._max_tile))
        corners_with_max = self._max_tile_corners()
        if max_count == 1 and corners_with_max:
            if self._anchor_corner is None:
                target_corner = corners_with_max[0]
            elif self._anchor_corner in corners_with_max:
                target_corner = self._anchor_corner
            else:
                target_corner = None  # max moved out of the anchor corner

            if target_corner is not None:
                for m, mr in zip(self._MILESTONES, self._MILESTONE_REWARDS, strict=False):
                    if m <= self._max_tile and m not in self._milestones_hit:
                        self._milestones_hit.add(m)
                        reward += mr
                        if self._anchor_corner is None:
                            self._anchor_corner = target_corner

        # Win condition: target tile reached.
        won = self._max_tile >= TARGET_TILE
        # Loss condition: no legal moves remain (board locked).
        no_moves = not self._has_valid_moves()
        terminated = bool(won or no_moves)

        if won:
            info["won"] = True
        if no_moves and not won:
            info["no_legal_moves"] = True

        info["score"] = self._score
        info["max_tile"] = self._max_tile
        info["milestones_hit"] = sorted(self._milestones_hit)
        return self._render_current_observation(), reward, terminated, False, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        # Render as a padded grid for alignment
        cell_width = 4
        rows: list[str] = []
        for r in range(SIZE):
            cells: list[str] = []
            for c in range(SIZE):
                sym = _tile_sym(int(self._board[r][c]))
                cells.append(sym.rjust(cell_width))
            rows.append("".join(cells))
        grid_str = "\n".join(rows)

        # Collect unique tile symbols for legend
        sym_meanings: dict[str, str] = {SYM_EMPTY: "empty cell"}
        for r in range(SIZE):
            for c in range(SIZE):
                val = int(self._board[r][c])
                if val > 0:
                    sym = _tile_sym(val)
                    if sym not in sym_meanings:
                        sym_meanings[sym] = f"tile with value {val}"

        legend = build_legend(sym_meanings)

        if self._anchor_corner is None:
            anchor_str = "none yet"
        else:
            ar, ac = self._anchor_corner
            anchor_str = f"({ar},{ac})"
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Score: {self._score}    "
            f"Max tile: {self._max_tile} / {TARGET_TILE}    "
            f"Milestones: {len(self._milestones_hit)}/{len(self._MILESTONES)}    "
            f"Anchor corner: {anchor_str}"
        )

        return GridObservation(grid=grid_str, legend=legend, hud=hud, message="")

    def system_prompt(self) -> str:
        milestone_rows = "\n".join(
            f"    +{r:.2f} for max-tile = {v}"
            for v, r in zip(self._MILESTONES, self._MILESTONE_REWARDS, strict=False)
        )
        return (
            f"You are playing {self.env_id()}.\n\n"
            "TASK\n"
            f"Slide tiles on a {SIZE}x{SIZE} grid to merge matching numbers and reach the "
            f"target tile {TARGET_TILE}.\n\n"
            "RULES\n"
            "- Tiles slide in the chosen direction until they hit a wall or another tile.\n"
            "- Two tiles with the same value merge into one with double the value.\n"
            "- After each valid move, a new tile (2 or 4) spawns on a random empty cell.\n"
            f"- Reach a tile of value {TARGET_TILE} to win the game.\n"
            "- If no valid moves remain, the game ends with whatever reward you earned.\n\n"
            "REWARD\n"
            "- A milestone fires the first time, on a step's end, your current max-tile is "
            "UNIQUE on the board (no other tile shares that value) AND sits in a corner.\n"
            "- The first milestone you score locks in an ANCHOR CORNER (whichever corner "
            "held your unique max-tile when it scored). All later milestones must be earned "
            "with the unique max-tile in that same anchor corner.\n"
            "- If at any moment the max-tile is duplicated, or the max-tile drifts out of "
            "the anchor corner, milestones reached at that step do not score (you can earn "
            "them later once both conditions are restored).\n"
            "- Milestone rewards (cumulative cap = 1.0):\n"
            f"{milestone_rows}\n"
            f"- Tiles: numbers show the value (e.g. 2, 4, 8, ..., {TARGET_TILE}).\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

