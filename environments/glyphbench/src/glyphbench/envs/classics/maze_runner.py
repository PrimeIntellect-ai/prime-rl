"""Procedurally generated maze navigation.

Navigate from start to exit in a randomly generated maze.

Gym IDs:
  glyphbench/classics-maze-easy-v0      (9x9)
  glyphbench/classics-maze-medium-v0    (15x15)
  glyphbench/classics-maze-hard-v0      (21x21)
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAZE_ACTION_SPEC = ActionSpec(
    names=("UP", "DOWN", "LEFT", "RIGHT"),
    descriptions=(
        "move player up one cell",
        "move player down one cell",
        "move player left one cell",
        "move player right one cell",
    ),
)

SYM_PLAYER = "@"
SYM_WALL = "\u2588"   # █
SYM_PATH = "\u00b7"   # ·
SYM_EXIT = "\u2605"   # ★

_DIR_DELTAS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}

DIFFICULTY = {
    "easy": {"size": 9, "max_turns": 100},
    "medium": {"size": 15, "max_turns": 300},
    "hard": {"size": 21, "max_turns": 500},
}


# ---------------------------------------------------------------------------
# Maze generation (recursive backtracker)
# ---------------------------------------------------------------------------


def _generate_maze(
    width: int, height: int, rng: Any
) -> tuple[list[list[str]], tuple[int, int], tuple[int, int]]:
    """Generate a maze using recursive backtracker (iterative version).

    The grid must have odd dimensions. Walls are on even coordinates,
    passages on odd coordinates.

    Returns: (grid, start_pos, exit_pos)
    """
    assert width % 2 == 1 and height % 2 == 1, "Maze dimensions must be odd"

    # Start with all walls
    grid = [[SYM_WALL for _ in range(width)] for _ in range(height)]

    # Carve passages using iterative DFS (recursive backtracker)
    start_x, start_y = 1, 1
    grid[start_y][start_x] = SYM_PATH

    stack: list[tuple[int, int]] = [(start_x, start_y)]
    # Directions: (dx, dy) in cell-space (step 2 to skip wall between cells)
    directions = [(0, -2), (0, 2), (-2, 0), (2, 0)]

    while stack:
        cx, cy = stack[-1]

        # Find unvisited neighbors
        neighbors: list[tuple[int, int, int, int]] = []
        for ddx, ddy in directions:
            nx, ny = cx + ddx, cy + ddy
            if 1 <= nx < width - 1 and 1 <= ny < height - 1 and grid[ny][nx] == SYM_WALL:
                neighbors.append((nx, ny, cx + ddx // 2, cy + ddy // 2))

        if neighbors:
            # Pick random neighbor
            idx = int(rng.integers(0, len(neighbors)))
            nx, ny, wx, wy = neighbors[idx]
            # Carve passage
            grid[wy][wx] = SYM_PATH
            grid[ny][nx] = SYM_PATH
            stack.append((nx, ny))
        else:
            stack.pop()

    # Start: top-left passage
    start_pos = (1, 1)
    # Exit: bottom-right passage
    exit_pos = (width - 2, height - 2)
    grid[exit_pos[1]][exit_pos[0]] = SYM_EXIT

    return grid, start_pos, exit_pos


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class _MazeRunnerBase(BaseGlyphEnv):
    """Maze runner: navigate a procedurally generated maze from start to exit."""

    action_spec = MAZE_ACTION_SPEC
    noop_action_name: str = "UP"

    _grid_size: int = 9
    _difficulty: str = "easy"

    def __init__(self, max_turns: int = 100) -> None:
        super().__init__(max_turns=max_turns)
        self._player_pos: tuple[int, int] = (0, 0)
        self._exit_pos: tuple[int, int] = (0, 0)
        self._maze: list[list[str]] = []
        self._won: bool = False

    def env_id(self) -> str:
        return f"glyphbench/classics-maze-{self._difficulty}-v0"

    # ------------------------------------------------------------------
    # Reset / Step
    # ------------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        self._won = False
        self._maze, self._player_pos, self._exit_pos = _generate_maze(
            self._grid_size, self._grid_size, self.rng
        )
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        name = self.action_spec.names[action]

        dx, dy = _DIR_DELTAS[name]
        nx, ny = self._player_pos[0] + dx, self._player_pos[1] + dy

        # Check bounds and wall
        if (
            0 <= nx < self._grid_size
            and 0 <= ny < self._grid_size
            and self._maze[ny][nx] != SYM_WALL
        ):
            self._player_pos = (nx, ny)

        # Check exit
        if self._player_pos == self._exit_pos:
            self._won = True
            return self._render_current_observation(), 1.0, True, False, info

        # Pattern A with bounded shaping: -1/max_turns per step so that
        # cumulative reward bottoms at -1 if the exit is never reached.
        return (
            self._render_current_observation(),
            -1.0 / self.max_turns,
            False,
            False,
            info,
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        # Copy maze grid
        grid = [row[:] for row in self._maze]

        # Stamp player
        px, py = self._player_pos
        grid[py][px] = SYM_PLAYER

        legend = build_legend({
            SYM_PLAYER: "player (you)",
            SYM_WALL: "wall",
            SYM_PATH: "path",
            SYM_EXIT: "exit (reach here to win)",
        })

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Maze: {self._grid_size}x{self._grid_size} ({self._difficulty})"
        )

        msg = ""
        if self._won:
            msg = "You found the exit! You win!"

        return GridObservation(grid=grid_to_string(grid), legend=legend, hud=hud, message=msg)

    def system_prompt(self) -> str:
        return (
            f"You are playing {self.env_id()}.\n\n"
            "TASK\n"
            "Navigate through a procedurally generated maze from the start to the exit.\n\n"
            "RULES\n"
            f"- The maze is {self._grid_size}x{self._grid_size} ({self._difficulty} difficulty).\n"
            "- Find the exit (\u2605) somewhere in the maze.\n"
            "- Moving into a wall does nothing.\n"
            "- Reaching the exit gives +1 reward and ends the episode.\n"
            "- Each step costs a small penalty scaled so cumulative reward\n"
            "  bottoms at -1 if you never reach the exit. Find the exit fast.\n"
            f"- The episode ends after {self.max_turns} steps if the exit is not reached.\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Concrete variants
# ---------------------------------------------------------------------------


class MazeEasyEnv(_MazeRunnerBase):
    _grid_size = 9
    _difficulty = "easy"

    def __init__(self, max_turns: int = 100) -> None:
        super().__init__(max_turns=max_turns)


class MazeMediumEnv(_MazeRunnerBase):
    _grid_size = 15
    _difficulty = "medium"

    def __init__(self, max_turns: int = 300) -> None:
        super().__init__(max_turns=max_turns)


class MazeHardEnv(_MazeRunnerBase):
    _grid_size = 21
    _difficulty = "hard"

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__(max_turns=max_turns)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
