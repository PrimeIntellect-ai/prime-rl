"""miniatari Up'n Down.

Identity: Vertical road game; collect flags scattered across the road.
Win condition: collect 5 flags.
Reward: Pattern A, +1/5 per flag collected.
Loss: collide with an oncoming car (no -1; just terminates).

Gym ID: glyphbench/miniatari-upndown-v0

Random baseline (seed=0..29): success_rate=3%, mean_length=52, mean_return=+0.160
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniUpNDownEnv(MiniatariBase):
    """Mini Up'n Down: 14x10 grid, 5 flags + 3 oncoming cars.

    Player car (Y) drives on a 6-cell-wide road (columns 4..9). Each
    tick the agent picks LEFT, RIGHT, UP, DOWN, or JUMP. UP moves up,
    DOWN moves down. JUMP makes the next tick collision-immune (jumping
    over a car). 5 flags (F) sit at fixed grid cells across the road.
    Stepping onto a flag's cell collects it (+1/5). 3 oncoming cars (#)
    drift down 1 cell every 2 ticks; if a car overlaps the player when
    not jumping, the run ends.
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP", "DOWN", "LEFT", "RIGHT", "JUMP"),
        descriptions=(
            "do nothing",
            "move up one cell",
            "move down one cell",
            "move left one cell",
            "move right one cell",
            "jump (immune to a car for the next tick)",
        ),
    )

    default_max_turns = 300

    _WIDTH = 14
    _HEIGHT = 10
    _WIN_TARGET = 5
    _ROAD_X_MIN = 4
    _ROAD_X_MAX = 9
    _CAR_MOVE_EVERY = 2

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._flags: list[tuple[int, int]] = []
        self._cars: list[list[int]] = []  # [x, y]
        self._jump_immune: int = 0  # ticks remaining of immunity
        self._tick_count: int = 0
        self._progress: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-upndown-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._progress = 0
        self._tick_count = 0
        self._jump_immune = 0
        self._player_x = 6
        self._player_y = self._HEIGHT - 1
        rng = self.rng
        # Scatter 5 flags on the road, distinct cells, not on player start
        positions: set[tuple[int, int]] = set()
        flags: list[tuple[int, int]] = []
        for _ in range(self._WIN_TARGET):
            for _attempt in range(40):
                fx = int(rng.integers(self._ROAD_X_MIN, self._ROAD_X_MAX + 1))
                fy = int(rng.integers(0, self._HEIGHT - 2))
                if (fx, fy) in positions:
                    continue
                positions.add((fx, fy))
                flags.append((fx, fy))
                break
        self._flags = flags
        # 3 oncoming cars at top
        self._cars = []
        for _ in range(3):
            for _attempt in range(20):
                cx = int(rng.integers(self._ROAD_X_MIN, self._ROAD_X_MAX + 1))
                cy = int(rng.integers(0, 4))
                if (cx, cy) in positions:
                    continue
                self._cars.append([cx, cy])
                positions.add((cx, cy))
                break

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._tick_count += 1
        if self._jump_immune > 0:
            self._jump_immune -= 1

        # 1. Player move
        nx, ny = self._player_x, self._player_y
        if action_name == "UP":
            ny = max(0, ny - 1)
            self._player_dir = (0, -1)
        elif action_name == "DOWN":
            ny = min(self._HEIGHT - 1, ny + 1)
            self._player_dir = (0, 1)
        elif action_name == "LEFT":
            nx = max(self._ROAD_X_MIN, nx - 1)
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT":
            nx = min(self._ROAD_X_MAX, nx + 1)
            self._player_dir = (1, 0)
        elif action_name == "JUMP":
            self._jump_immune = 2
            self._message = "Jumping!"
        self._player_x, self._player_y = nx, ny

        # 2. Collect flag
        for i, (fx, fy) in enumerate(self._flags):
            if fx == self._player_x and fy == self._player_y:
                self._flags.pop(i)
                self._progress += 1
                reward += self._progress_reward(self._WIN_TARGET)
                self._message = f"Flag! ({self._progress}/{self._WIN_TARGET})"
                if self._progress >= self._WIN_TARGET:
                    self._on_won()
                    return reward, self._game_over, info
                break

        # 3. Move cars (every 2 ticks; downward; despawn at bottom)
        if self._tick_count % self._CAR_MOVE_EVERY == 0:
            new_cars: list[list[int]] = []
            for cx, cy in self._cars:
                cy += 1
                if cy >= self._HEIGHT:
                    continue
                new_cars.append([cx, cy])
            # Spawn replacement car at top with low chance
            if len(new_cars) < 3:
                rng = self.rng
                cx = int(rng.integers(self._ROAD_X_MIN, self._ROAD_X_MAX + 1))
                if not any(c[0] == cx and c[1] == 0 for c in new_cars):
                    new_cars.append([cx, 0])
            self._cars = new_cars

        # 4. Collision check
        for cx, cy in self._cars:
            if cx == self._player_x and cy == self._player_y:
                if self._jump_immune > 0:
                    self._message = "Jumped over a car!"
                else:
                    self._message = "Hit by a car!"
                    self._game_over = True
                    self._won = False
                    return reward, True, info

        info["progress"] = self._progress
        info["flags_left"] = len(self._flags)
        info["cars"] = len(self._cars)
        info["jump_immune"] = self._jump_immune
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        # Road borders
        for y in range(self._HEIGHT):
            grid[y][self._ROAD_X_MIN - 1] = "│"
            grid[y][self._ROAD_X_MAX + 1] = "│"
        # Flags
        for fx, fy in self._flags:
            if 0 <= fx < self._WIDTH and 0 <= fy < self._HEIGHT:
                grid[fy][fx] = "F"
        # Cars
        for cx, cy in self._cars:
            if 0 <= cx < self._WIDTH and 0 <= cy < self._HEIGHT:
                grid[cy][cx] = "#"
        # Player
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            pch = "J" if self._jump_immune > 0 else "Y"
            grid[self._player_y][self._player_x] = pch

        symbols = {
            " ": "road",
            "│": "road edge",
            "F": "flag (collect)",
            "#": "oncoming car",
            "Y": "your car",
            "J": "your car (jumping; immune)",
        }
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Flags: {self._progress}/{self._WIN_TARGET}    "
            f"Score: {self._score:.3f}    "
            f"Jump immune: {self._jump_immune}    "
            f"Flags left: {len(self._flags)}    "
            f"Cars: {len(self._cars)}"
        )
        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Up'n Down on a 14x10 grid. The road occupies columns "
            "4..9 (│ edges). Your car (Y) starts at the bottom. 5 flags "
            "(F) sit at fixed cells across the road; touch one to collect "
            "(+1/5). 3 oncoming cars (#) drift down 1 cell every 2 ticks. "
            "UP/DOWN/LEFT/RIGHT moves you 1 cell on the road. JUMP grants "
            "2 ticks of car immunity (J marker). Collide with a car while "
            "not jumping and the run ends. Collect 5 flags to win. Reward: "
            "+1/5 per flag."
        )
