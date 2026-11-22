"""miniatari Asterix.

Identity: Pacman-like; collect helmets while avoiding enemies.
Win condition: collect 5 helmets.
Reward: Pattern D, +1/5 per helmet; -1 if caught by an enemy.

Gym ID: glyphbench/miniatari-asterix-v0

Random baseline (seed=0..29): success_rate=0%, mean_length=17, mean_return=-0.907
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


def _sign(a: int, b: int) -> int:
    return 1 if a > b else (-1 if a < b else 0)


class MiniAsterixEnv(MiniatariBase):
    """Mini Asterix: 14x10 open grid, 5 helmets, 2 chasing enemies.

    Player moves freely on a 14x10 grid surrounded by border walls.
    5 helmets (h) are placed at distinct interior cells; each gives
    +1/5 when collected. 2 enemies (e) wander toward the player at 1
    cell every 3 ticks on the dominant axis. Pattern D: +1/5 per helmet,
    -1 if either enemy reaches the player's cell.
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

    _WIDTH = 14
    _HEIGHT = 10
    _N_HELMETS = 5
    _WIN_TARGET = _N_HELMETS
    _N_ENEMIES = 2
    _ENEMY_MOVE_EVERY = 3

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._helmets: set[tuple[int, int]] = set()
        self._enemies: list[list[int]] = []
        self._tick_count: int = 0
        self._progress: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-asterix-v0"

    def _is_wall(self, x: int, y: int) -> bool:
        return x <= 0 or x >= self._WIDTH - 1 or y <= 0 or y >= self._HEIGHT - 1

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._progress = 0
        self._tick_count = 0
        for y in range(self._HEIGHT):
            for x in range(self._WIDTH):
                if self._is_wall(x, y):
                    self._set_cell(x, y, "█")
        self._player_x = self._WIDTH // 2
        self._player_y = self._HEIGHT // 2
        self._player_dir = (1, 0)

        rng = self.rng
        used: set[tuple[int, int]] = {(self._player_x, self._player_y)}
        # Helmets
        self._helmets = set()
        while len(self._helmets) < self._N_HELMETS:
            x = int(rng.integers(1, self._WIDTH - 1))
            y = int(rng.integers(1, self._HEIGHT - 1))
            if (x, y) in used or self._is_wall(x, y):
                continue
            used.add((x, y))
            self._helmets.add((x, y))
        # Enemies far from the player
        self._enemies = []
        while len(self._enemies) < self._N_ENEMIES:
            x = int(rng.integers(1, self._WIDTH - 1))
            y = int(rng.integers(1, self._HEIGHT - 1))
            if (x, y) in used or self._is_wall(x, y):
                continue
            if abs(x - self._player_x) + abs(y - self._player_y) < 5:
                continue
            used.add((x, y))
            self._enemies.append([x, y])

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
        if not self._is_wall(nx, ny):
            self._player_x, self._player_y = nx, ny

        # 2. Collect helmet
        cell = (self._player_x, self._player_y)
        if cell in self._helmets:
            self._helmets.discard(cell)
            reward += self._progress_reward(self._WIN_TARGET)
            self._progress += 1
            self._message = f"Helmet! ({self._progress}/{self._WIN_TARGET})"
            if self._progress >= self._WIN_TARGET:
                self._on_won()
                return reward, self._game_over, info

        # 3. Enemies move
        if self._tick_count % self._ENEMY_MOVE_EVERY == 0:
            for e in self._enemies:
                dx = _sign(self._player_x, e[0])
                dy = _sign(self._player_y, e[1])
                nex, ney = e[0], e[1]
                if abs(self._player_x - e[0]) >= abs(self._player_y - e[1]):
                    if not self._is_wall(e[0] + dx, e[1]):
                        nex = e[0] + dx
                    elif not self._is_wall(e[0], e[1] + dy):
                        ney = e[1] + dy
                else:
                    if not self._is_wall(e[0], e[1] + dy):
                        ney = e[1] + dy
                    elif not self._is_wall(e[0] + dx, e[1]):
                        nex = e[0] + dx
                e[0], e[1] = nex, ney

        # 4. Catch?
        for ex, ey in self._enemies:
            if ex == self._player_x and ey == self._player_y:
                self._message = "An enemy caught you!"
                reward += self._death_reward()
                self._on_life_lost()
                return reward, True, info

        info["progress"] = self._progress
        info["helmets_left"] = len(self._helmets)
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        for y in range(self._HEIGHT):
            for x in range(self._WIDTH):
                if self._is_wall(x, y):
                    grid[y][x] = "█"
        for hx, hy in self._helmets:
            if 0 <= hx < self._WIDTH and 0 <= hy < self._HEIGHT:
                grid[hy][hx] = "h"
        for ex, ey in self._enemies:
            if 0 <= ex < self._WIDTH and 0 <= ey < self._HEIGHT:
                grid[ey][ex] = "e"
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            pch = self._DIR_CHARS.get(self._player_dir, "@")
            grid[self._player_y][self._player_x] = pch

        symbols = {
            " ": "open ground",
            "█": "wall",
            "h": "helmet",
            "e": "enemy",
        }
        pch = self._DIR_CHARS.get(self._player_dir, "@")
        symbols[pch] = f"you (facing {self._DIR_NAMES.get(self._player_dir, 'right')})"

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Collected: {self._progress}/{self._WIN_TARGET}    "
            f"Score: {self._score:.3f}    "
            f"Helmets: {len(self._helmets)}    "
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
            "Mini Asterix on a 14x10 open arena with border walls. 5 "
            "helmets (h) sit at scattered interior cells. 2 enemies (e) "
            "step 1 cell every 3 ticks toward you on the dominant axis. "
            "LEFT/RIGHT/UP/DOWN moves you 1 cell. Stepping onto a helmet "
            "collects it for +1/5. Reward: +1/5 per helmet. Being caught "
            "by an enemy is -1 terminal."
        )
