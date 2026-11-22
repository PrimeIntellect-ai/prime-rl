"""miniatari Amidar.

Identity: Paint cells of a small rectangle by walking on them while a
patrol enemy roams the path.
Win condition: paint all 10 cells of the rectangle perimeter.
Reward: Pattern D, +1/10 per painted cell; -1 if caught by patrol.

Gym ID: glyphbench/miniatari-amidar-v0

Random baseline (seed=0..29): success_rate=0%, mean_length=118, mean_return=-0.417
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniAmidarEnv(MiniatariBase):
    """Mini Amidar: 12x10 grid, walk a 3x4 rectangle perimeter.

    The painting rectangle has 10 perimeter cells (4 corners + 6 edge
    cells). Paint each by stepping onto it. The rectangle's position,
    the player spawn, and the patrol enemy's (e) start cell + direction
    are randomized per seed in _generate_level. The patrol traverses the
    perimeter 1 cell every 3 ticks. Pattern D: +1/10 per painted cell,
    -1 on catch.
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

    default_max_turns = 300

    _WIDTH = 12
    _HEIGHT = 10
    # Rectangle is always 3 wide x 4 tall (10 perimeter cells); only its
    # top-left origin is randomized per seed in _generate_level so the win
    # target and difficulty stay constant while the layout differs.
    _RECT_W = 3
    _RECT_H = 4
    _RECT = (3, 3, 5, 6)  # x0, y0, x1, y1 (inclusive); set per-seed at reset
    _N_CELLS = 10  # perimeter cells to paint
    _WIN_TARGET = _N_CELLS
    _PATROL_MOVE_EVERY = 3

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._painted: set[tuple[int, int]] = set()
        self._patrol_pos: list[int] = [0, 0]
        self._patrol_step: int = 0
        self._patrol_dir: int = 1  # +1 clockwise, -1 counter-clockwise
        self._tick_count: int = 0
        self._progress: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-amidar-v0"

    def _perimeter_cells(self) -> list[tuple[int, int]]:
        x0, y0, x1, y1 = self._RECT
        cells: list[tuple[int, int]] = []
        # Top row: (x0..x1, y0)
        for x in range(x0, x1 + 1):
            cells.append((x, y0))
        # Right col: (x1, y0+1..y1)
        for y in range(y0 + 1, y1 + 1):
            cells.append((x1, y))
        # Bottom row reversed: (x1-1..x0, y1)
        for x in range(x1 - 1, x0 - 1, -1):
            cells.append((x, y1))
        # Left col reversed: (x0, y1-1..y0+1)
        for y in range(y1 - 1, y0, -1):
            cells.append((x0, y))
        return cells

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._progress = 0
        self._tick_count = 0
        self._painted = set()
        rng = self.rng
        # Randomize the rectangle's top-left origin so the perimeter the
        # agent must paint sits in a different place each seed. Keep a
        # 1-cell margin on every side so the player always has a legal
        # spawn cell adjacent to the perimeter (never clipped by a wall).
        x0 = int(rng.integers(1, self._WIDTH - self._RECT_W))
        y0 = int(rng.integers(1, self._HEIGHT - self._RECT_H))
        x1 = x0 + self._RECT_W - 1
        y1 = y0 + self._RECT_H - 1
        self._RECT = (x0, y0, x1, y1)
        cells = self._perimeter_cells()

        # Player spawns one cell next to a randomly chosen perimeter cell, so
        # the starting distance to each target and the safe approach differ
        # per seed. The spawn is on the board but off the perimeter (so the
        # player must still step onto a track cell to start painting).
        spawn_targets = list(range(len(cells)))
        rng.shuffle(spawn_targets)
        for ci in spawn_targets:
            cx, cy = cells[ci]
            neighbours = [
                (ox, oy)
                for ox, oy in ((cx, cy - 1), (cx, cy + 1), (cx - 1, cy), (cx + 1, cy))
                if 0 <= ox < self._WIDTH and 0 <= oy < self._HEIGHT
                and not self._on_perim(ox, oy)
            ]
            if neighbours:
                sx, sy = neighbours[int(rng.integers(len(neighbours)))]
                self._player_x, self._player_y = sx, sy
                break
        else:  # pragma: no cover - margin guarantees a spawn exists
            self._player_x, self._player_y = x0, max(0, y0 - 1)
        self._player_dir = (0, 1)

        # Patrol: random start index along the perimeter and random
        # direction (clockwise vs counter-clockwise). Start it away from the
        # player so seed 0 isn't an instant catch.
        self._patrol_dir = 1 if rng.random() < 0.5 else -1
        far_idx = self._farthest_perim_index(cells, self._player_x, self._player_y)
        self._patrol_step = far_idx
        px, py = cells[self._patrol_step]
        self._patrol_pos = [px, py]

    @staticmethod
    def _farthest_perim_index(
        cells: list[tuple[int, int]], px: int, py: int
    ) -> int:
        best_i, best_d = 0, -1
        for i, (cx, cy) in enumerate(cells):
            d = abs(cx - px) + abs(cy - py)
            if d > best_d:
                best_d, best_i = d, i
        return best_i

    def _on_perim(self, x: int, y: int) -> bool:
        x0, y0, x1, y1 = self._RECT
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return False
        return y in (y0, y1) or x in (x0, x1)

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._tick_count += 1

        # 1. Player move
        nx, ny = self._player_x, self._player_y
        if action_name == "LEFT":
            nx = max(0, nx - 1)
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT":
            nx = min(self._WIDTH - 1, nx + 1)
            self._player_dir = (1, 0)
        elif action_name == "UP":
            ny = max(0, ny - 1)
            self._player_dir = (0, -1)
        elif action_name == "DOWN":
            ny = min(self._HEIGHT - 1, ny + 1)
            self._player_dir = (0, 1)
        self._player_x, self._player_y = nx, ny

        # 2. Paint cell if standing on perimeter and not yet painted
        if self._on_perim(self._player_x, self._player_y):
            cell = (self._player_x, self._player_y)
            if cell not in self._painted:
                self._painted.add(cell)
                reward += self._progress_reward(self._WIN_TARGET)
                self._progress += 1
                self._message = f"Painted! ({self._progress}/{self._WIN_TARGET})"
                if self._progress >= self._WIN_TARGET:
                    self._on_won()
                    return reward, self._game_over, info

        # 3. Patrol step
        if self._tick_count % self._PATROL_MOVE_EVERY == 0:
            cells = self._perimeter_cells()
            self._patrol_step = (self._patrol_step + self._patrol_dir) % len(cells)
            self._patrol_pos = list(cells[self._patrol_step])

        # 4. Catch?
        if (self._patrol_pos[0] == self._player_x and
                self._patrol_pos[1] == self._player_y):
            self._message = "Caught by the patrol!"
            reward += self._death_reward()
            self._on_life_lost()
            return reward, True, info

        info["progress"] = self._progress
        info["painted"] = len(self._painted)
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        # Render perimeter
        for cx, cy in self._perimeter_cells():
            if (cx, cy) in self._painted:
                grid[cy][cx] = "▓"
            else:
                grid[cy][cx] = "·"
        # Patrol
        px, py = self._patrol_pos
        if 0 <= px < self._WIDTH and 0 <= py < self._HEIGHT:
            grid[py][px] = "e"
        # Player
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            pch = self._DIR_CHARS.get(self._player_dir, "@")
            grid[self._player_y][self._player_x] = pch

        symbols = {
            " ": "open space",
            "·": "unpainted track",
            "▓": "painted track",
            "e": "patrol enemy",
        }
        pch = self._DIR_CHARS.get(self._player_dir, "@")
        symbols[pch] = f"you (facing {self._DIR_NAMES.get(self._player_dir, 'down')})"

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Painted: {self._progress}/{self._WIN_TARGET}    "
            f"Score: {self._score:.3f}"
        )
        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Amidar on a 12x10 grid. A 3x4 rectangle has 10 unpainted "
            "track cells (·) on its perimeter; its position varies each "
            "episode. Stepping onto a track cell paints it (▓). A patrol "
            "enemy (e) walks the perimeter, advancing 1 cell every 3 ticks "
            "(its start position and direction vary each episode). "
            "LEFT/RIGHT/UP/DOWN moves you 1 cell. Reward: +1/10 per newly "
            "painted cell. If the patrol's cell coincides with you, you take "
            "a -1 terminal penalty. Read the grid to locate the rectangle "
            "and the patrol before moving."
        )
