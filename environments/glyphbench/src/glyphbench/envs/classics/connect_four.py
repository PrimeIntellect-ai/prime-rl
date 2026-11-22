"""Connect Four: drop tokens in columns vs a simple AI opponent.

Gym IDs:
  glyphbench/classics-connectfour-v0
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

COLS = 7
ROWS = 6

CONNECT_FOUR_ACTION_SPEC = ActionSpec(
    names=("DROP_0", "DROP_1", "DROP_2", "DROP_3", "DROP_4", "DROP_5", "DROP_6"),
    descriptions=(
        "drop your token in column 0 (leftmost)",
        "drop your token in column 1",
        "drop your token in column 2",
        "drop your token in column 3 (center)",
        "drop your token in column 4",
        "drop your token in column 5",
        "drop your token in column 6 (rightmost)",
    ),
)

EMPTY = 0
PLAYER = 1
OPPONENT = 2

SYM_EMPTY = "\u00b7"   # ·
SYM_PLAYER = "\u25cf"  # ●
SYM_OPP = "\u25cb"     # ○

_CELL_SYM = {EMPTY: SYM_EMPTY, PLAYER: SYM_PLAYER, OPPONENT: SYM_OPP}

# ---------------------------------------------------------------------------
# Connect Four Env
# ---------------------------------------------------------------------------


class ConnectFourEnv(BaseGlyphEnv):
    """Connect Four: drop tokens to get 4 in a row before the AI."""

    action_spec = CONNECT_FOUR_ACTION_SPEC
    noop_action_name: str = "DROP_0"

    def __init__(self, max_turns: int = 42) -> None:
        super().__init__(max_turns=max_turns)
        self._board: np.ndarray = np.zeros((ROWS, COLS), dtype=np.int8)
        self._message: str = ""

    def env_id(self) -> str:
        return "glyphbench/classics-connectfour-v0"

    # ------------------------------------------------------------------
    # Board helpers
    # ------------------------------------------------------------------

    def _drop(self, col: int, token: int) -> int:
        """Drop a token in a column. Returns the row it landed in, or -1 if full."""
        for row in range(ROWS - 1, -1, -1):
            if self._board[row][col] == EMPTY:
                self._board[row][col] = token
                return row
        return -1

    def _column_full(self, col: int) -> bool:
        return bool(self._board[0][col] != EMPTY)

    def _board_full(self) -> bool:
        return bool(np.all(self._board != EMPTY))

    def _check_winner(self, token: int) -> bool:
        """Check if the given token has 4 in a row."""
        b = self._board
        # Horizontal
        for r in range(ROWS):
            for c in range(COLS - 3):
                if b[r][c] == b[r][c + 1] == b[r][c + 2] == b[r][c + 3] == token:
                    return True
        # Vertical
        for r in range(ROWS - 3):
            for c in range(COLS):
                if b[r][c] == b[r + 1][c] == b[r + 2][c] == b[r + 3][c] == token:
                    return True
        # Diagonal down-right
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                if b[r][c] == b[r + 1][c + 1] == b[r + 2][c + 2] == b[r + 3][c + 3] == token:
                    return True
        # Diagonal up-right
        for r in range(3, ROWS):
            for c in range(COLS - 3):
                if b[r][c] == b[r - 1][c + 1] == b[r - 2][c + 2] == b[r - 3][c + 3] == token:
                    return True
        return False

    # ------------------------------------------------------------------
    # AI opponent
    # ------------------------------------------------------------------

    def _ai_move(self) -> int | None:
        """Simple AI: first check for winning move, then block player win, else random."""
        valid = [c for c in range(COLS) if not self._column_full(c)]
        if not valid:
            return None

        # Try to win
        for c in valid:
            row = self._drop(c, OPPONENT)
            if self._check_winner(OPPONENT):
                self._board[row][c] = EMPTY
                return c
            self._board[row][c] = EMPTY

        # Block player win
        for c in valid:
            row = self._drop(c, PLAYER)
            if self._check_winner(PLAYER):
                self._board[row][c] = EMPTY
                return c
            self._board[row][c] = EMPTY

        # Random
        return int(self.rng.choice(valid))

    # ------------------------------------------------------------------
    # Reset / Step
    # ------------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        self._board = np.zeros((ROWS, COLS), dtype=np.int8)
        self._message = "Your turn. Drop a token in a column."
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        col = action  # DROP_0 -> 0, etc.

        # Invalid move (column full) = NOOP
        if self._column_full(col):
            self._message = f"Column {col} is full. Choose another column."
            return self._render_current_observation(), 0.0, False, False, info

        # Player move
        self._drop(col, PLAYER)

        if self._check_winner(PLAYER):
            self._message = "You win! Four in a row!"
            info["outcome"] = "win"
            return self._render_current_observation(), 1.0, True, False, info

        if self._board_full():
            self._message = "Board full. It's a draw."
            info["outcome"] = "draw"
            return self._render_current_observation(), 0.0, True, False, info

        # AI move
        ai_col = self._ai_move()
        if ai_col is not None:
            self._drop(ai_col, OPPONENT)
            self._message = f"Opponent dropped in column {ai_col}."

            if self._check_winner(OPPONENT):
                self._message += " Opponent wins!"
                info["outcome"] = "loss"
                return self._render_current_observation(), -1.0, True, False, info

            if self._board_full():
                self._message += " Board full. It's a draw."
                info["outcome"] = "draw"
                return self._render_current_observation(), 0.0, True, False, info
        else:
            self._message = "Board full. It's a draw."
            info["outcome"] = "draw"
            return self._render_current_observation(), 0.0, True, False, info

        return self._render_current_observation(), 0.0, False, False, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        grid = make_empty_grid(COLS, ROWS, SYM_EMPTY)
        for r in range(ROWS):
            for c in range(COLS):
                grid[r][c] = _CELL_SYM[int(self._board[r][c])]

        legend = build_legend({
            SYM_EMPTY: "empty",
            SYM_PLAYER: "your token",
            SYM_OPP: "opponent token",
        })

        grid_str = grid_to_string(grid)
        hud = f"Step: {self._turn} / {self.max_turns}\nColumns: 0 1 2 3 4 5 6"

        return GridObservation(grid=grid_str, legend=legend, hud=hud, message=self._message)

    def system_prompt(self) -> str:
        return (
            f"You are playing {self.env_id()}.\n\n"
            "TASK\n"
            "Play Connect Four against an AI opponent. Drop tokens into columns "
            "to get four of your tokens in a row (horizontal, vertical, or diagonal).\n\n"
            "RULES\n"
            "- The board is 7 columns wide and 6 rows tall.\n"
            "- Tokens fall to the lowest empty position in the chosen column.\n"
            "- If a column is full, the action is a no-op.\n"
            "- You play as " + SYM_PLAYER + " (player), the AI plays as " + SYM_OPP + " (opponent).\n"
            "- First to get 4 in a row wins. Full board with no winner is a draw.\n"
            "- Reward: +1 for a win, -1 for a loss, 0 for a draw.\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
