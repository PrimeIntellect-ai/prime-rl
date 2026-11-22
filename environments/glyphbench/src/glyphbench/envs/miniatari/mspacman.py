"""miniatari Ms. Pac-Man.

Identity: Eat every dot in a small maze while avoiding ghosts.
Win condition: collect every dot lining the corridors (~64 cells).
Reward: Pattern D, +1/N per dot eaten (N = total dots); -1 if a ghost
catches you.

Gym ID: glyphbench/miniatari-mspacman-v0

Random baseline (seed=0..29): success_rate=0%, mean_length=155, mean_return=-0.863
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


def _sign(a: int, b: int) -> int:
    return 1 if a > b else (-1 if a < b else 0)


# Static 14x10 maze layout. '#' is wall, '.' is dot location, ' ' is empty path.
_MAZE = [
    "##############",
    "#............#",
    "#.##.####.##.#",
    "#............#",
    "#.##.#  #.##.#",
    "#....#  #....#",
    "#.##.####.##.#",
    "#............#",
    "#............#",
    "##############",
]


class MiniMsPacmanEnv(MiniatariBase):
    """Mini Ms. Pac-Man: 14x10 walled maze, dots line every corridor cell, 2 ghosts.

    The maze is a fixed 14x10 layout with corridors. Dots (·) are placed
    on every open corridor cell except the ghost pen (rows 4-5, cols
    6-7) and the player start. 2 ghosts (g) spawn in the pen; for the
    first ~10 ticks they march UP out of the pen, then chase the
    player on the dominant axis (skipped if blocked). Pattern D:
    +1/N per dot eaten (N = total dots, ~64), -1 on ghost catch.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "UP", "DOWN"),
        descriptions=(
            "do nothing",
            "move left and face left",
            "move right and face right",
            "move up and face up",
            "move down and face down",
        ),
    )

    default_max_turns = 400

    _WIDTH = 14
    _HEIGHT = 10
    _GHOST_MOVE_EVERY = 2
    _N_GHOSTS = 2
    # Ghosts spend their first ~10 ticks marching up out of the pen so
    # they don't get stuck. After this they switch to the chase AI.
    _PEN_EXIT_TICKS = 10

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._walls: set[tuple[int, int]] = set()
        self._dots: set[tuple[int, int]] = set()
        self._ghosts: list[list[int]] = []
        self._tick_count: int = 0
        self._progress: int = 0
        self._n_dots: int = 0  # set in _generate_level

    def env_id(self) -> str:
        return "glyphbench/miniatari-mspacman-v0"

    def _is_wall(self, x: int, y: int) -> bool:
        return (x, y) in self._walls

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._tick_count = 0
        self._progress = 0
        self._walls = set()
        # Parse static maze layout: '#' is wall, all others are open path.
        candidate_dot_cells: list[tuple[int, int]] = []
        for y in range(self._HEIGHT):
            for x in range(self._WIDTH):
                ch = _MAZE[y][x]
                if ch == "#":
                    self._walls.add((x, y))
                else:
                    candidate_dot_cells.append((x, y))
        # Player at column 1, row 8 (open cell)
        self._player_x = 1
        self._player_y = 8
        self._player_dir = (1, 0)
        # Place a dot on EVERY open corridor cell except the ghost pen
        # and the player's start cell. This was previously capped at 16
        # which left rows 3-8 unsugared.
        ghost_pen = {(6, 4), (7, 4), (6, 5), (7, 5)}
        self._dots = set()
        for cell in candidate_dot_cells:
            if cell == (self._player_x, self._player_y):
                continue
            if cell in ghost_pen:
                continue
            self._dots.add(cell)
        self._n_dots = len(self._dots)
        # Ghost spawn (inside the pen).
        self._ghosts = [[6, 4], [7, 5]]

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._tick_count += 1

        # 1. Player move
        nx, ny = self._player_x, self._player_y
        if action_name == "LEFT":
            nx -= 1
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT":
            nx += 1
            self._player_dir = (1, 0)
        elif action_name == "UP":
            ny -= 1
            self._player_dir = (0, -1)
        elif action_name == "DOWN":
            ny += 1
            self._player_dir = (0, 1)
        if 0 <= nx < self._WIDTH and 0 <= ny < self._HEIGHT and not self._is_wall(nx, ny):
            self._player_x, self._player_y = nx, ny

        # 2. Eat dot
        cell = (self._player_x, self._player_y)
        if cell in self._dots:
            self._dots.discard(cell)
            reward += self._progress_reward(self._n_dots)
            self._progress += 1
            self._message = f"Dot! ({self._progress}/{self._n_dots})"
            if self._progress >= self._n_dots:
                self._on_won()
                return reward, self._game_over, info

        # 3. Ghosts move every K ticks
        if self._tick_count % self._GHOST_MOVE_EVERY == 0:
            for g in self._ghosts:
                if self._tick_count <= self._PEN_EXIT_TICKS:
                    # Pen-exit phase: step UP each ghost-move tick if
                    # possible. From the pen at row 4-5, this carries
                    # ghosts to row 3 (an open corridor row), after
                    # which the chase AI takes over.
                    if not self._is_wall(g[0], g[1] - 1):
                        g[1] -= 1
                    continue
                dx = _sign(self._player_x, g[0])
                dy = _sign(self._player_y, g[1])
                ngx, ngy = g[0], g[1]
                if abs(self._player_x - g[0]) >= abs(self._player_y - g[1]):
                    if dx != 0 and not self._is_wall(g[0] + dx, g[1]):
                        ngx = g[0] + dx
                    elif dy != 0 and not self._is_wall(g[0], g[1] + dy):
                        ngy = g[1] + dy
                else:
                    if dy != 0 and not self._is_wall(g[0], g[1] + dy):
                        ngy = g[1] + dy
                    elif dx != 0 and not self._is_wall(g[0] + dx, g[1]):
                        ngx = g[0] + dx
                g[0], g[1] = ngx, ngy

        # 4. Catch?
        for gx, gy in self._ghosts:
            if gx == self._player_x and gy == self._player_y:
                self._message = "A ghost caught you!"
                reward += self._death_reward()
                self._on_life_lost()
                return reward, True, info

        info["progress"] = self._progress
        info["dots_left"] = len(self._dots)
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        for wx, wy in self._walls:
            grid[wy][wx] = "█"
        for dx, dy in self._dots:
            grid[dy][dx] = "·"
        for gx, gy in self._ghosts:
            if 0 <= gx < self._WIDTH and 0 <= gy < self._HEIGHT:
                grid[gy][gx] = "g"
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            pch = self._DIR_CHARS.get(self._player_dir, "@")
            grid[self._player_y][self._player_x] = pch

        symbols = {
            " ": "open path",
            "█": "wall",
            "·": "dot",
            "g": "ghost",
        }
        pch = self._DIR_CHARS.get(self._player_dir, "@")
        symbols[pch] = f"you (facing {self._DIR_NAMES.get(self._player_dir, 'right')})"

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Eaten: {self._progress}/{self._n_dots}    "
            f"Score: {self._score:.3f}    "
            f"Ghosts: {len(self._ghosts)}"
        )
        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Ms. Pac-Man on a 14x10 walled maze with corridors. "
            "Dots (·) line every open corridor cell except the central "
            "ghost pen and your start cell (~64 dots total — see the "
            "HUD for the exact N this run). You (Y, arrow shows facing) "
            "start at column 1, row 8. 2 ghosts (g) start in the central "
            "pen and march UP to escape the pen for the first ~10 ticks; "
            "after that they step 1 cell every 2 ticks toward you on the "
            "dominant axis (blocked by walls). LEFT/RIGHT/UP/DOWN moves "
            "you 1 cell. Stepping onto a dot eats it for +1/N. Reward: "
            "+1/N per dot (N is the total dot count this episode). "
            "Being caught by a ghost is -1 terminal."
        )
