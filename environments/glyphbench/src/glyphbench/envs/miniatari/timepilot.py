"""miniatari Time Pilot.

Identity: 360-degree aerial shooter; enemies converge on you from every edge.
Win condition: destroy 5 enemies in this era.
Reward: Pattern A, +1/5 per enemy destroyed.
Loss: collision with an enemy ends the run (no -1; just terminates).

Gym ID: glyphbench/miniatari-timepilot-v0

Random baseline (seed=0..29): success_rate=0%, mean_length=13, mean_return=+0.013
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


def _sign(a: int, b: int) -> int:
    return 1 if a > b else (-1 if a < b else 0)


class MiniTimePilotEnv(MiniatariBase):
    """Mini Time Pilot: 14x12 grid, 5 enemies converging from edges.

    Player ship (Y, arrow shows facing) starts at center. 5 enemies (E)
    spawn at random edge positions and step 1 cell every 2 ticks toward
    the player along the dominant axis. FIRE shoots up to 4 cells in the
    facing direction; the first enemy in line is destroyed. Movement
    actions translate the ship 1 cell AND set the facing. Colliding with
    an enemy ends the run. Destroy all 5 to win.
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP", "DOWN", "LEFT", "RIGHT", "FIRE"),
        descriptions=(
            "do nothing",
            "move up and face up",
            "move down and face down",
            "move left and face left",
            "move right and face right",
            "fire in your current facing direction",
        ),
    )

    default_max_turns = 200

    _WIDTH = 14
    _HEIGHT = 12
    _WIN_TARGET = 5
    _BULLET_RANGE = 4
    _ENEMY_MOVE_EVERY = 2

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._enemies: list[list[int]] = []
        self._tick_count: int = 0
        self._progress: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-timepilot-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._progress = 0
        self._tick_count = 0
        self._player_x = self._WIDTH // 2
        self._player_y = self._HEIGHT // 2
        self._player_dir = (0, -1)  # face up
        rng = self.rng
        # 5 enemies on edges
        self._enemies = []
        used: set[tuple[int, int]] = set()
        for _ in range(self._WIN_TARGET):
            for _attempt in range(20):
                side = int(rng.integers(0, 4))
                if side == 0:  # top
                    x = int(rng.integers(0, self._WIDTH))
                    y = 0
                elif side == 1:  # bottom
                    x = int(rng.integers(0, self._WIDTH))
                    y = self._HEIGHT - 1
                elif side == 2:  # left
                    x = 0
                    y = int(rng.integers(0, self._HEIGHT))
                else:  # right
                    x = self._WIDTH - 1
                    y = int(rng.integers(0, self._HEIGHT))
                if (x, y) in used or (x == self._player_x and y == self._player_y):
                    continue
                used.add((x, y))
                self._enemies.append([x, y])
                break

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._tick_count += 1

        # 1. Player move
        nx, ny = self._player_x, self._player_y
        if action_name == "UP":
            ny = max(0, ny - 1)
            self._player_dir = (0, -1)
        elif action_name == "DOWN":
            ny = min(self._HEIGHT - 1, ny + 1)
            self._player_dir = (0, 1)
        elif action_name == "LEFT":
            nx = max(0, nx - 1)
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT":
            nx = min(self._WIDTH - 1, nx + 1)
            self._player_dir = (1, 0)
        self._player_x, self._player_y = nx, ny

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
                if bx < 0 or bx >= self._WIDTH or by < 0 or by >= self._HEIGHT:
                    break
                hit = False
                for i, (ex, ey) in enumerate(self._enemies):
                    if ex == bx and ey == by:
                        target = i
                        hit = True
                        break
                if hit:
                    break
            if target is not None:
                self._enemies.pop(target)
                reward += self._progress_reward(self._WIN_TARGET)
                self._progress += 1
                self._message = f"Enemy down! ({self._progress}/{self._WIN_TARGET})"
                if self._progress >= self._WIN_TARGET:
                    self._on_won()
                    return reward, self._game_over, info

        # 3. Move enemies (every K ticks, 1 cell on the dominant axis toward player)
        if self._tick_count % self._ENEMY_MOVE_EVERY == 0:
            for e in self._enemies:
                dx = _sign(self._player_x, e[0])
                dy = _sign(self._player_y, e[1])
                if abs(self._player_x - e[0]) >= abs(self._player_y - e[1]):
                    e[0] += dx
                else:
                    e[1] += dy

        # 4. Collision
        for ex, ey in self._enemies:
            if ex == self._player_x and ey == self._player_y:
                self._message = "Enemy rammed your ship!"
                self._game_over = True
                self._won = False
                return reward, True, info

        info["progress"] = self._progress
        info["enemies_left"] = len(self._enemies)
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        for ex, ey in self._enemies:
            if 0 <= ex < self._WIDTH and 0 <= ey < self._HEIGHT:
                grid[ey][ex] = "E"
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            pch = self._DIR_CHARS.get(self._player_dir, "@")
            grid[self._player_y][self._player_x] = pch

        symbols = {
            " ": "sky",
            "E": "enemy ship",
        }
        pch = self._DIR_CHARS.get(self._player_dir, "@")
        symbols[pch] = f"your ship (facing {self._DIR_NAMES.get(self._player_dir, 'up')})"

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Killed: {self._progress}/{self._WIN_TARGET}    "
            f"Score: {self._score:.3f}    "
            f"Enemies: {len(self._enemies)}"
        )
        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Time Pilot on a 14x12 grid. You pilot a ship (arrow shows "
            "facing) starting at the center. 5 enemy ships (E) spawn at "
            "random edge positions and converge on you, stepping 1 cell "
            "every 2 ticks toward your position along the dominant axis. "
            "Movement actions translate you 1 cell AND set your facing. "
            "FIRE shoots up to 4 cells in your facing direction; the first "
            "enemy in line is destroyed. Colliding with any enemy ends the "
            "run. Destroy all 5 to clear the era. Reward: +1/5 per enemy."
        )
