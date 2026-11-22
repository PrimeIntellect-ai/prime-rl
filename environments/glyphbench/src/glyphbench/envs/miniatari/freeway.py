"""miniatari Freeway.

Identity: Cross a 3-lane highway dodging cars; chicken crosses the road.
Win condition: agent reaches the top side of the road 4 times.
Reward: Pattern A, +1/4 per successful crossing.
Loss: time runs out. A car collision sends the chicken back to the start of
the current crossing, so reckless straight-line dashes lose progress and
burn the (tight) time budget.

Gym ID: glyphbench/miniatari-freeway-v0

Random baseline (seed=0..29): success_rate=0%, mean_length=60, mean_return~0.0
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniFreewayEnv(MiniatariBase):
    """Mini Freeway: 14x10 grid, 3 lanes of moving cars.

    Agent (chicken) starts at the bottom edge (y=9) and must reach the top
    edge (y=0). Each crossing teleports the chicken back to the bottom.
    Cars occupy rows 2,4,6 (lanes), each moving in opposite directions at
    1 cell/tick. Lane 1 (row 2): cars move right at +1 cell/tick.
    Lane 2 (row 4): cars move left at -1 cell/tick.
    Lane 3 (row 6): cars move right at +1 cell/tick.
    Hitting a car sends the chicken back to the start of the current crossing
    (no death penalty, but lost progress under a tight time budget).
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP", "DOWN"),
        descriptions=(
            "stay still",
            "move chicken up one row",
            "move chicken down one row",
        ),
    )

    # Tight budget: a flawless run needs ~36 ticks (4 crossings x 9 rows); a
    # timing-aware agent that waits out cars needs <=57. The budget leaves
    # only a few ticks of slack so collisions (which reset the chicken to the
    # start of the crossing) are genuinely costly. With this budget a blind
    # UP-spam wins ~31% of seeds while a dodging policy wins ~100%.
    default_max_turns = 60

    _WIDTH = 14
    _HEIGHT = 10
    _START_Y = 9
    _GOAL_Y = 0
    _LANE_ROWS = (2, 4, 6)
    _WIN_TARGET = 4

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        # cars per lane: list of (x_pos, direction)
        self._cars: list[list[int]] = []  # [lane_idx][car_idx] -> x
        self._lane_dirs: list[int] = []
        self._progress: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-freeway-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._progress = 0
        self._player_x = self._WIDTH // 2
        self._player_y = self._START_Y
        # 3 cars per lane, evenly spaced; alternating directions
        self._lane_dirs = [1, -1, 1]
        self._cars = []
        for _li, _row in enumerate(self._LANE_ROWS):
            spacing = self._WIDTH // 3
            offset = int(self.rng.integers(0, spacing))
            cars_in_lane = [
                (offset + spacing * i) % self._WIDTH for i in range(3)
            ]
            self._cars.append(cars_in_lane)

    def _car_at(self, x: int, y: int) -> bool:
        for li, row in enumerate(self._LANE_ROWS):
            if row == y:
                return x in self._cars[li]
        return False

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        # 1. Move chicken
        if action_name == "UP":
            self._player_y = max(0, self._player_y - 1)
            self._player_dir = (0, -1)
        elif action_name == "DOWN":
            self._player_y = min(self._HEIGHT - 1, self._player_y + 1)
            self._player_dir = (0, 1)

        # 2. Check goal
        if self._player_y <= self._GOAL_Y:
            reward += self._progress_reward(self._WIN_TARGET)
            self._progress += 1
            self._message = f"Crossed! ({self._progress}/{self._WIN_TARGET})"
            if self._progress >= self._WIN_TARGET:
                self._on_won()
                return reward, self._game_over, info
            else:
                # Reset to start
                self._player_y = self._START_Y

        hit_before_cars_move = self._car_at(self._player_x, self._player_y)

        # 3. Move cars
        for li, _row in enumerate(self._LANE_ROWS):
            d = self._lane_dirs[li]
            self._cars[li] = [
                (cx + d) % self._WIDTH for cx in self._cars[li]
            ]

        # 4. Car collision (before or after cars move): lose the current
        # crossing's progress by snapping back to the start row.
        if hit_before_cars_move or self._car_at(self._player_x, self._player_y):
            self._message = "Hit by a car!"
            self._player_y = self._START_Y

        info["progress"] = self._progress
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        # Goal stripe
        for x in range(self._WIDTH):
            grid[self._GOAL_Y][x] = "─"
        # Lane stripes
        for row in (1, 3, 5, 7):
            for x in range(self._WIDTH):
                grid[row][x] = "·"
        # Cars
        for li, row in enumerate(self._LANE_ROWS):
            for cx in self._cars[li]:
                if 0 <= cx < self._WIDTH:
                    grid[row][cx] = "#"
        # Chicken
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            grid[self._player_y][self._player_x] = "Y"

        symbols = {
            " ": "road lane",
            "·": "lane stripe",
            "─": "goal line (north shore)",
            "#": "car",
            "Y": "you (chicken)",
        }

        lane_dir_str = "  ".join(
            f"L{i+1}(row{r}):{'+' if d == 1 else '-'}"
            for i, (r, d) in enumerate(zip(self._LANE_ROWS, self._lane_dirs, strict=False))
        )
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Crossings: {self._progress}/{self._WIN_TARGET}    "
            f"Score: {self._score:.3f}\n"
            f"Chicken: ({self._player_x},{self._player_y})    "
            f"Lanes: {lane_dir_str}"
        )

        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Freeway: a 14x10 highway with 3 lanes of cars (rows 2, 4, "
            "6). You are the chicken (Y) starting at row 9. Move UP/DOWN to "
            "navigate; reach row 0 to score a crossing. After scoring you "
            "respawn at the bottom. Cars (#) move 1 cell per tick: row 2 "
            "right, row 4 left, row 6 right (each wraps around). Getting "
            "hit by a car sends you back to row 9, costing you the progress "
            "of the current crossing -- time is tight, so dodge carefully. "
            "Cross the highway 4 times to win. Reward: +1/4 per crossing."
        )
