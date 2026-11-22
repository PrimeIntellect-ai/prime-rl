"""Classic Tetris falling blocks game (short-horizon variant).

Gym IDs:
  glyphbench/classics-tetris-v0

Reward shape: Pattern B (milestone shaping) — +0.2 per line cleared
across 5 lines; cumulative reward caps at 1.0 on full clear.

Design (P4 rework, 2026-05-01):
  - Well = 6 wide x 10 tall (instead of classic 10x20).
  - Goal = clear 5 lines.
  - Reward = +0.2 per line cleared. Cumulative caps at 1.0 (5 lines).
  - max_turns = 300. Truncation gives whatever cumulative reward was earned.
  - Top-out (cannot spawn next piece) terminates with cumulative-so-far.

Random baseline (seeds 0-29, max_turns=300):
  success_rate=0% mean_length=19 mean_return=+0.000
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

_WIDTH = 6
_HEIGHT = 10

# Lines required to win the episode. Reward per line cleared is 1.0 / _TARGET_LINES,
# so cumulative reward caps at 1.0 when all _TARGET_LINES lines are cleared.
_TARGET_LINES = 5
_REWARD_PER_LINE = 1.0 / _TARGET_LINES

DEFAULT_MAX_TURNS = 300

_SYM_EMPTY = "\u00b7"  # ·
_SYM_ACTIVE = "\u2593"  # ▓

# Locked piece symbols per type
_PIECE_SYMS = {
    "I": "\u25b0",  # ▰
    "O": "\u25ae",  # ▮
    "T": "\u25b2",  # ▲
    "S": "\u25c6",  # ◆
    "Z": "\u25c7",  # ◇
    "J": "\u25c4",  # ◄
    "L": "\u25ba",  # ►
}

# Tetromino shapes: list of (row, col) offsets. Row 0 = top.
# Each piece has 4 rotations.
_SHAPES: dict[str, list[list[tuple[int, int]]]] = {
    "I": [
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(0, 0), (1, 0), (2, 0), (3, 0)],
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(0, 0), (1, 0), (2, 0), (3, 0)],
    ],
    "O": [
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ],
    "T": [
        [(0, 0), (0, 1), (0, 2), (1, 1)],
        [(0, 0), (1, 0), (2, 0), (1, 1)],
        [(1, 0), (1, 1), (1, 2), (0, 1)],
        [(0, 0), (1, 0), (2, 0), (1, -1)],
    ],
    "S": [
        [(0, 1), (0, 2), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(0, 1), (0, 2), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
    ],
    "Z": [
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 1), (1, 0), (1, 1), (2, 0)],
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 1), (1, 0), (1, 1), (2, 0)],
    ],
    "J": [
        [(0, 0), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (0, 1), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (0, 2), (1, 2)],
        [(0, 0), (1, 0), (2, 0), (2, -1)],
    ],
    "L": [
        [(0, 2), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 0)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
    ],
}

_PIECE_TYPES = list(_SHAPES.keys())

# ---------------------------------------------------------------------------
# Action spec
# ---------------------------------------------------------------------------

TETRIS_ACTION_SPEC = ActionSpec(
    names=("LEFT", "RIGHT", "ROTATE_CW", "ROTATE_CCW", "HARD_DROP", "NOOP"),
    descriptions=(
        "move piece one cell left",
        "move piece one cell right",
        "rotate piece 90 degrees clockwise",
        "rotate piece 90 degrees counter-clockwise",
        "instantly drop piece to the lowest valid position",
        "do nothing (piece drops 1 row by gravity)",
    ),
)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class TetrisEnv(BaseGlyphEnv):
    """Tetris on a 6 wide x 10 tall well; clear 5 lines to win.

    Reward = +0.2 per line cleared. Cumulative caps at 1.0 once 5 lines are
    cleared, at which point the episode terminates with ``won=True``.
    Top-out terminates without further reward.
    """

    action_spec = TETRIS_ACTION_SPEC
    noop_action_name = "NOOP"

    def __init__(self, max_turns: int = DEFAULT_MAX_TURNS) -> None:
        super().__init__(max_turns=max_turns)
        # Board: stores piece type char or empty string
        self._board: list[list[str]] = []
        # Current piece state
        self._piece_type: str = "I"
        self._piece_rot: int = 0
        self._piece_row: int = 0
        self._piece_col: int = 0
        self._game_over: bool = False
        self._score: int = 0
        self._lines_cleared: int = 0
        # Tracks cumulative reward earned so far (capped at 1.0).
        self._reward_total: float = 0.0
        self._won: bool = False

    def env_id(self) -> str:
        return "glyphbench/classics-tetris-v0"

    # ------------------------------------------------------------------
    # Piece helpers
    # ------------------------------------------------------------------

    def _get_cells(self, ptype: str, rot: int, row: int, col: int) -> list[tuple[int, int]]:
        """Return list of (row, col) for the piece at given position."""
        return [(row + dr, col + dc) for dr, dc in _SHAPES[ptype][rot % 4]]

    def _valid_position(self, ptype: str, rot: int, row: int, col: int) -> bool:
        for r, c in self._get_cells(ptype, rot, row, col):
            if r < 0 or r >= _HEIGHT or c < 0 or c >= _WIDTH:
                return False
            if self._board[r][c] != "":
                return False
        return True

    def _spawn_piece(self) -> bool:
        """Spawn a new random piece at the top. Returns False if blocked."""
        self._piece_type = _PIECE_TYPES[int(self.rng.integers(len(_PIECE_TYPES)))]
        self._piece_rot = 0
        self._piece_row = 0
        self._piece_col = _WIDTH // 2 - 1
        return self._valid_position(
            self._piece_type, self._piece_rot, self._piece_row, self._piece_col
        )

    def _lock_piece(self) -> None:
        """Lock the current piece into the board."""
        for r, c in self._get_cells(
            self._piece_type, self._piece_rot, self._piece_row, self._piece_col
        ):
            if 0 <= r < _HEIGHT and 0 <= c < _WIDTH:
                self._board[r][c] = self._piece_type

    def _clear_lines(self) -> int:
        """Clear full lines and return number cleared."""
        cleared = 0
        new_board: list[list[str]] = []
        for r in range(_HEIGHT):
            if all(self._board[r][c] != "" for c in range(_WIDTH)):
                cleared += 1
            else:
                new_board.append(self._board[r])
        # Add empty rows at top
        while len(new_board) < _HEIGHT:
            new_board.insert(0, [""] * _WIDTH)
        self._board = new_board
        return cleared

    def _drop_one(self) -> bool:
        """Try to drop piece one row. Returns True if successful."""
        if self._valid_position(
            self._piece_type, self._piece_rot, self._piece_row + 1, self._piece_col
        ):
            self._piece_row += 1
            return True
        return False

    def _hard_drop(self) -> None:
        """Drop piece as far as it goes."""
        while self._drop_one():
            pass

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        self._board = [[""] * _WIDTH for _ in range(_HEIGHT)]
        self._game_over = False
        self._score = 0
        self._lines_cleared = 0
        self._reward_total = 0.0
        self._won = False
        if not self._spawn_piece():
            self._game_over = True
        return self._render_current_observation()

    # ------------------------------------------------------------------
    # Reward helpers
    # ------------------------------------------------------------------

    def _reward_for_clear(self, cleared: int) -> float:
        """Flat +_REWARD_PER_LINE per line cleared. Cap cumulative at 1.0."""
        normalized = _REWARD_PER_LINE * cleared
        room = max(0.0, 1.0 - self._reward_total)
        gained = min(normalized, room)
        self._reward_total += gained
        return gained

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        reward = 0.0

        if self._game_over:
            return self._render_current_observation(), 0.0, True, False, info

        name = self.action_spec.names[action]

        # Apply action
        if name == "LEFT":
            if self._valid_position(
                self._piece_type, self._piece_rot, self._piece_row, self._piece_col - 1
            ):
                self._piece_col -= 1
        elif name == "RIGHT":
            if self._valid_position(
                self._piece_type, self._piece_rot, self._piece_row, self._piece_col + 1
            ):
                self._piece_col += 1
        elif name == "ROTATE_CW":
            new_rot = (self._piece_rot + 1) % 4
            if self._valid_position(
                self._piece_type, new_rot, self._piece_row, self._piece_col
            ):
                self._piece_rot = new_rot
        elif name == "ROTATE_CCW":
            new_rot = (self._piece_rot - 1) % 4
            if self._valid_position(
                self._piece_type, new_rot, self._piece_row, self._piece_col
            ):
                self._piece_rot = new_rot
        elif name == "HARD_DROP":
            self._hard_drop()
            # Lock immediately after hard drop
            self._lock_piece()
            cleared = self._clear_lines()
            if cleared > 0:
                self._lines_cleared += cleared
                reward = self._reward_for_clear(cleared)
                self._score += cleared
            # Win condition check before spawning the next piece.
            if self._lines_cleared >= _TARGET_LINES:
                self._won = True
                self._game_over = True
                info["won"] = True
            elif not self._spawn_piece():
                self._game_over = True
                info["game_over"] = True
            return self._render_current_observation(), reward, self._game_over, False, info
        # NOOP: just gravity

        # Gravity: drop one row
        if not self._drop_one():
            # Can't drop -> lock
            self._lock_piece()
            cleared = self._clear_lines()
            if cleared > 0:
                self._lines_cleared += cleared
                reward = self._reward_for_clear(cleared)
                self._score += cleared
            # Win condition check before spawning the next piece.
            if self._lines_cleared >= _TARGET_LINES:
                self._won = True
                self._game_over = True
                info["won"] = True
            elif not self._spawn_piece():
                self._game_over = True
                info["game_over"] = True

        return self._render_current_observation(), reward, self._game_over, False, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        grid = make_empty_grid(_WIDTH, _HEIGHT, fill=_SYM_EMPTY)
        syms: dict[str, str] = {_SYM_EMPTY: "empty"}

        # Draw locked pieces
        for r in range(_HEIGHT):
            for c in range(_WIDTH):
                pt = self._board[r][c]
                if pt != "":
                    sym = _PIECE_SYMS[pt]
                    grid[r][c] = sym
                    if sym not in syms:
                        syms[sym] = f"locked {pt}-piece"

        # Draw active piece
        if not self._game_over:
            for r, c in self._get_cells(
                self._piece_type, self._piece_rot, self._piece_row, self._piece_col
            ):
                if 0 <= r < _HEIGHT and 0 <= c < _WIDTH:
                    grid[r][c] = _SYM_ACTIVE
            syms[_SYM_ACTIVE] = "falling piece"

        legend = build_legend(syms)
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Score: {self._score}    "
            f"Lines: {self._lines_cleared} / {_TARGET_LINES}    "
            f"Piece: {self._piece_type}"
        )
        if self._won:
            msg = f"You cleared {_TARGET_LINES} lines! You win."
        elif self._game_over:
            msg = "Game over! A new piece cannot spawn."
        else:
            msg = ""

        return GridObservation(
            grid=grid_to_string(grid), legend=legend, hud=hud, message=msg
        )

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def system_prompt(self) -> str:
        return (
            f"You are playing {self.env_id()} -- short-horizon Tetris.\n\n"
            "TASK\n"
            f"Clear {_TARGET_LINES} lines on a {_WIDTH} wide x {_HEIGHT} tall well to win.\n\n"
            "RULES\n"
            "Tetrominoes (I, O, T, S, Z, J, L) fall from the top.\n"
            "Each turn: your action is applied, then gravity drops the piece 1 row.\n"
            "HARD_DROP is the exception: it drops and locks the current piece immediately,\n"
            "then a new piece spawns at the top without an extra gravity drop that turn.\n"
            "Rotations are rigid with no wall-kicks; some pieces have fewer unique orientations.\n"
            "If the piece cannot drop further, it locks in place and a new piece spawns.\n"
            "A horizontal line filled with locked cells is cleared and the rows above shift down.\n"
            f"The episode ends when you have cleared {_TARGET_LINES} lines (win), when a "
            "new piece cannot spawn (top-out), or when the step budget is exhausted.\n\n"
            "REWARD\n"
            f"+{_REWARD_PER_LINE:.2f} per line cleared. Cumulative reward caps at 1.0 once "
            f"{_TARGET_LINES} lines have been cleared.\n\n"
            "STRATEGY\n"
            "Stack pieces flat. Keep the well low. Save space for line clears.\n"
            "Use HARD_DROP to instantly place a piece at the bottom.\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
