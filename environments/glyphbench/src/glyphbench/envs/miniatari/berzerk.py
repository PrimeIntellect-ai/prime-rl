"""miniatari Berzerk.

Identity: Navigate a small electrified room shooting robots; don't touch walls.
Win condition: clear all 5 robots in the room.
Reward: Pattern D, +1/5 per robot destroyed; -1 on touching an electrified wall.

Gym ID: glyphbench/miniatari-berzerk-v0

Random baseline (seed=0..29): success_rate=0%, mean_length=9, mean_return=-0.973
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


def _sign(a: int, b: int) -> int:
    return 1 if a > b else (-1 if a < b else 0)


class MiniBerzerkEnv(MiniatariBase):
    """Mini Berzerk: 16x10 grid, walled small room with 5 robots.

    The room is enclosed by electrified walls (█) on the border. Player
    (Y, arrow facing) starts near the center. 5 robots (r) start in
    distinct interior positions and step 1 cell every 3 ticks toward the
    player on the dominant axis. FIRE shoots in your facing direction up
    to 5 cells; first robot in line is destroyed. Stepping into a wall
    cell is a -1 terminal (the room is electrified). Robots are also
    destroyed if they walk into a wall (game logic checks). Clear all 5
    to win.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "UP", "DOWN", "FIRE"),
        descriptions=(
            "do nothing",
            "move left and face left",
            "move right and face right",
            "move up and face up",
            "move down and face down",
            "fire in your current facing direction",
        ),
    )

    default_max_turns = 200

    _WIDTH = 16
    _HEIGHT = 10
    _WIN_TARGET = 5
    _BULLET_RANGE = 5
    _ROBOT_MOVE_EVERY = 3

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        # robots: [x, y]
        self._robots: list[list[int]] = []
        self._tick_count: int = 0
        self._progress: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-berzerk-v0"

    def _is_wall(self, x: int, y: int) -> bool:
        return x <= 0 or x >= self._WIDTH - 1 or y <= 0 or y >= self._HEIGHT - 1

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._progress = 0
        self._tick_count = 0
        # Borders
        for x in range(self._WIDTH):
            self._set_cell(x, 0, "█")
            self._set_cell(x, self._HEIGHT - 1, "█")
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "█")
            self._set_cell(self._WIDTH - 1, y, "█")
        self._player_x = self._WIDTH // 2
        self._player_y = self._HEIGHT // 2
        self._player_dir = (0, -1)
        # Place 5 robots at distinct interior cells
        rng = self.rng
        used: set[tuple[int, int]] = {(self._player_x, self._player_y)}
        self._robots = []
        for _ in range(self._WIN_TARGET):
            for _attempt in range(40):
                x = int(rng.integers(2, self._WIDTH - 2))
                y = int(rng.integers(2, self._HEIGHT - 2))
                if (x, y) in used:
                    continue
                if abs(x - self._player_x) + abs(y - self._player_y) < 3:
                    continue
                used.add((x, y))
                self._robots.append([x, y])
                break

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._tick_count += 1

        # 1. Player move (no clamping; stepping into wall is fatal)
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
        self._player_x, self._player_y = nx, ny
        if self._is_wall(self._player_x, self._player_y):
            self._message = "Touched the electrified wall!"
            reward += self._death_reward()
            self._on_life_lost()
            return reward, True, info

        # 2. Fire
        if action_name == "FIRE":
            bdx, bdy = self._player_dir
            if bdx == 0 and bdy == 0:
                bdy = -1
            bx, by = self._player_x, self._player_y
            target: int | None = None
            for _ in range(self._BULLET_RANGE):
                bx += bdx
                by += bdy
                if self._is_wall(bx, by):
                    break
                hit = False
                for i, (ex, ey) in enumerate(self._robots):
                    if ex == bx and ey == by:
                        target = i
                        hit = True
                        break
                if hit:
                    break
            if target is not None:
                self._robots.pop(target)
                reward += self._progress_reward(self._WIN_TARGET)
                self._progress += 1
                self._message = f"Robot down! ({self._progress}/{self._WIN_TARGET})"
                if self._progress >= self._WIN_TARGET:
                    self._on_won()
                    return reward, self._game_over, info

        # 3. Robots move (every K ticks, toward player on dominant axis)
        if self._tick_count % self._ROBOT_MOVE_EVERY == 0:
            survivors: list[list[int]] = []
            for r in self._robots:
                dx = _sign(self._player_x, r[0])
                dy = _sign(self._player_y, r[1])
                nrx, nry = r[0], r[1]
                if abs(self._player_x - r[0]) >= abs(self._player_y - r[1]):
                    nrx += dx
                else:
                    nry += dy
                if self._is_wall(nrx, nry):
                    # Robot walks into a wall and dies (no reward)
                    continue
                survivors.append([nrx, nry])
            self._robots = survivors

        # 4. Robot collision with player
        for ex, ey in self._robots:
            if ex == self._player_x and ey == self._player_y:
                self._message = "A robot caught you!"
                reward += self._death_reward()
                self._on_life_lost()
                return reward, True, info

        info["progress"] = self._progress
        info["robots_left"] = len(self._robots)
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        for x in range(self._WIDTH):
            grid[0][x] = "█"
            grid[self._HEIGHT - 1][x] = "█"
        for y in range(self._HEIGHT):
            grid[y][0] = "█"
            grid[y][self._WIDTH - 1] = "█"
        for ex, ey in self._robots:
            if 0 <= ex < self._WIDTH and 0 <= ey < self._HEIGHT:
                grid[ey][ex] = "r"
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            pch = self._DIR_CHARS.get(self._player_dir, "@")
            grid[self._player_y][self._player_x] = pch

        symbols = {
            " ": "floor",
            "█": "electrified wall (deadly)",
            "r": "robot",
        }
        pch = self._DIR_CHARS.get(self._player_dir, "@")
        symbols[pch] = f"you (facing {self._DIR_NAMES.get(self._player_dir, 'up')})"

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Killed: {self._progress}/{self._WIN_TARGET}    "
            f"Score: {self._score:.3f}    "
            f"Robots: {len(self._robots)}"
        )
        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Berzerk on a 16x10 walled room. The border (█) is "
            "electrified — touching any wall cell is -1 terminal. You "
            "(arrow shows facing) start near the center. 5 robots (r) "
            "start at scattered interior positions and step 1 cell every "
            "3 ticks toward you on the dominant axis. LEFT/RIGHT/UP/DOWN "
            "moves you 1 cell and sets your facing. FIRE shoots up to 5 "
            "cells in your facing direction; first robot in line is "
            "destroyed. Reward: +1/5 per robot. Touching a wall or being "
            "caught by a robot is -1 terminal."
        )
