"""Tron light cycle game.

Leave a trail, don't crash into walls or trails.

Gym IDs:
  glyphbench/classics-tron-v0
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

GRID_SIZE = 15

TRON_ACTION_SPEC = ActionSpec(
    names=("UP", "DOWN", "LEFT", "RIGHT"),
    descriptions=(
        "turn and move up",
        "turn and move down",
        "turn and move left",
        "turn and move right",
    ),
)

# Player directional glyphs
SYM_PLAYER = {
    "RIGHT": "\u25b8",  # ▸
    "DOWN": "\u25be",    # ▾
    "LEFT": "\u25c2",    # ◂
    "UP": "\u25b4",      # ▴
}
SYM_PLAYER_TRAIL = "\u2591"  # ░
SYM_OPPONENT = "\u25c6"      # ◆
SYM_OPPONENT_TRAIL = "\u2592"  # ▒
SYM_WALL = "\u2588"           # █
SYM_EMPTY = "\u00b7"          # ·

_DIR_DELTAS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}

_OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class TronEnv(BaseGlyphEnv):
    """Tron: light cycle game against an AI opponent."""

    action_spec = TRON_ACTION_SPEC
    noop_action_name: str = "UP"

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)
        self._player_pos: tuple[int, int] = (0, 0)
        self._player_dir: str = "RIGHT"
        self._player_trail: set[tuple[int, int]] = set()

        self._opp_pos: tuple[int, int] = (0, 0)
        self._opp_dir: str = "LEFT"
        self._opp_trail: set[tuple[int, int]] = set()

        self._player_crashed: bool = False
        self._opp_crashed: bool = False

    def env_id(self) -> str:
        return "glyphbench/classics-tron-v0"

    # ------------------------------------------------------------------
    # Reset / Step
    # ------------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        mid = GRID_SIZE // 2
        self._player_pos = (3, mid)
        self._player_dir = "RIGHT"
        self._player_trail = {self._player_pos}

        self._opp_pos = (GRID_SIZE - 4, mid)
        self._opp_dir = "LEFT"
        self._opp_trail = {self._opp_pos}

        self._player_crashed = False
        self._opp_crashed = False

        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        name = self.action_spec.names[action]

        # Prevent reversing
        if _OPPOSITE.get(name) == self._player_dir:
            name = self._player_dir
        self._player_dir = name

        # Move player
        dx, dy = _DIR_DELTAS[self._player_dir]
        px, py = self._player_pos
        new_px, new_py = px + dx, py + dy

        # Move opponent (AI)
        opp_dir = self._ai_choose_direction()
        self._opp_dir = opp_dir
        odx, ody = _DIR_DELTAS[opp_dir]
        ox, oy = self._opp_pos
        new_ox, new_oy = ox + odx, oy + ody

        # Check player crash
        player_crashed = self._is_blocked(new_px, new_py)
        # Check opponent crash
        opp_crashed = self._is_blocked(new_ox, new_oy)

        # Both move simultaneously, check head-on collision
        if (new_px, new_py) == (new_ox, new_oy):
            player_crashed = True
            opp_crashed = True

        # Update positions and trails
        if not player_crashed:
            self._player_pos = (new_px, new_py)
            self._player_trail.add(self._player_pos)
        else:
            self._player_crashed = True

        if not opp_crashed:
            self._opp_pos = (new_ox, new_oy)
            self._opp_trail.add(self._opp_pos)
        else:
            self._opp_crashed = True

        # Determine outcome
        terminated = False
        reward = 0.0

        if self._player_crashed and self._opp_crashed:
            terminated = True
            reward = 0.0
            info["outcome"] = "draw"
        elif self._player_crashed:
            terminated = True
            reward = -1.0
            info["outcome"] = "player_crashed"
        elif self._opp_crashed:
            terminated = True
            reward = 1.0
            info["outcome"] = "opponent_crashed"

        return self._render_current_observation(), reward, terminated, False, info

    # ------------------------------------------------------------------
    # AI opponent
    # ------------------------------------------------------------------

    def _is_blocked(self, x: int, y: int) -> bool:
        """Check if a cell is blocked (wall, trail, or out of bounds)."""
        if x <= 0 or x >= GRID_SIZE - 1 or y <= 0 or y >= GRID_SIZE - 1:
            return True
        return bool((x, y) in self._player_trail or (x, y) in self._opp_trail)

    def _ai_choose_direction(self) -> str:
        """Simple AI: prefer continuing straight, otherwise pick longest open path."""
        ox, oy = self._opp_pos
        current_dir = self._opp_dir

        # Get valid directions (not opposite of current)
        valid_dirs: list[str] = []
        for d in ("UP", "DOWN", "LEFT", "RIGHT"):
            if _OPPOSITE.get(d) == current_dir:
                continue
            ddx, ddy = _DIR_DELTAS[d]
            nx, ny = ox + ddx, oy + ddy
            if not self._is_blocked(nx, ny):
                valid_dirs.append(d)

        if not valid_dirs:
            # No valid moves; just continue current direction (will crash)
            return current_dir

        # Score each direction by an open-line count.
        scores: dict[str, int] = {}
        for d in valid_dirs:
            ddx, ddy = _DIR_DELTAS[d]
            score = 0
            cx, cy = ox + ddx, oy + ddy
            while not self._is_blocked(cx, cy):
                score += 1
                cx += ddx
                cy += ddy
            scores[d] = score

        # Break ties randomly among best directions
        best_score = max(scores.values())
        best_dirs = [d for d in valid_dirs if scores[d] == best_score]

        if len(best_dirs) > 1:
            idx = int(self.rng.integers(0, len(best_dirs)))
            return best_dirs[idx]
        return best_dirs[0]

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        grid = make_empty_grid(GRID_SIZE, GRID_SIZE, SYM_EMPTY)

        # Walls (border)
        for x in range(GRID_SIZE):
            grid[0][x] = SYM_WALL
            grid[GRID_SIZE - 1][x] = SYM_WALL
        for y in range(GRID_SIZE):
            grid[y][0] = SYM_WALL
            grid[y][GRID_SIZE - 1] = SYM_WALL

        # Trails
        for tx, ty in self._player_trail:
            if (tx, ty) != self._player_pos:
                grid[ty][tx] = SYM_PLAYER_TRAIL
        for tx, ty in self._opp_trail:
            if (tx, ty) != self._opp_pos:
                grid[ty][tx] = SYM_OPPONENT_TRAIL

        # Player and opponent heads
        px, py = self._player_pos
        grid[py][px] = SYM_PLAYER[self._player_dir]

        ox, oy = self._opp_pos
        grid[oy][ox] = SYM_OPPONENT

        legend_map = {
            SYM_PLAYER["RIGHT"]: "you, moving right",
            SYM_PLAYER["DOWN"]: "you, moving down",
            SYM_PLAYER["LEFT"]: "you, moving left",
            SYM_PLAYER["UP"]: "you, moving up",
            SYM_PLAYER_TRAIL: "your trail",
            SYM_OPPONENT: "opponent",
            SYM_OPPONENT_TRAIL: "opponent trail",
            SYM_WALL: "wall",
            SYM_EMPTY: "empty",
        }

        legend = build_legend(legend_map)

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Opponent dir: {self._opp_dir}"
        )

        msg = ""
        if self._player_crashed and self._opp_crashed:
            msg = "Both crashed! Draw."
        elif self._player_crashed:
            msg = "You crashed!"
        elif self._opp_crashed:
            msg = "Opponent crashed! You win!"

        return GridObservation(grid=grid_to_string(grid), legend=legend, hud=hud, message=msg)

    def system_prompt(self) -> str:
        return (
            f"You are playing {self.env_id()}.\n\n"
            "TASK\n"
            "Control a light cycle on a grid. You leave a trail behind you as you move. "
            "Avoid crashing into walls, your own trail, or the opponent's trail.\n\n"
            "RULES\n"
            f"- The grid is {GRID_SIZE}x{GRID_SIZE} with walls around the border.\n"
            "- Each step, you move one cell in your chosen direction, leaving a trail.\n"
            "- An AI opponent does the same.\n"
            "- You cannot reverse direction (e.g., moving RIGHT cannot switch to LEFT).\n"
            "- Crashing into any wall or trail ends the game.\n"
            "- If only you crash: -1 reward. If only the opponent crashes: +1 reward.\n"
            "- If both crash simultaneously: 0 reward (draw).\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
