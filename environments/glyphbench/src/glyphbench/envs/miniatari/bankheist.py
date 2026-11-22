"""miniatari Bank Heist.

Identity: Drive between banks robbing them while cops chase you.
Win condition: rob 3 banks.
Reward: Pattern D, +1/3 per bank robbed; -1 if caught by a cop.

Gym ID: glyphbench/miniatari-bankheist-v0

Random baseline (seed=0..29): success_rate=0%, mean_length=20, mean_return=-0.989
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


def _sign(a: int, b: int) -> int:
    return 1 if a > b else (-1 if a < b else 0)


class MiniBankHeistEnv(MiniatariBase):
    """Mini Bank Heist: 14x10 city grid, 3 banks, 2 cops.

    Player car (arrow facing) drives across a 14x10 city. 3 banks (B)
    sit at scattered cells; stepping onto one robs it (+1/3). 2 cops (c)
    spawn at fixed corners and step toward the player at 1 cell every
    3 ticks. Pattern D: +1/3 per bank, -1 on catch. Border walls only.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "UP", "DOWN"),
        descriptions=(
            "do nothing",
            "drive left and face left",
            "drive right and face right",
            "drive up and face up",
            "drive down and face down",
        ),
    )

    default_max_turns = 400

    _WIDTH = 14
    _HEIGHT = 10
    _N_BANKS = 3
    _WIN_TARGET = _N_BANKS
    _N_COPS = 2
    _COP_MOVE_EVERY = 3

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._banks: set[tuple[int, int]] = set()
        self._cops: list[list[int]] = []
        self._tick_count: int = 0
        self._progress: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-bankheist-v0"

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
        # Banks
        self._banks = set()
        attempts = 0
        while len(self._banks) < self._N_BANKS and attempts < 200:
            attempts += 1
            x = int(rng.integers(1, self._WIDTH - 1))
            y = int(rng.integers(1, self._HEIGHT - 1))
            if (x, y) in used:
                continue
            if abs(x - self._player_x) + abs(y - self._player_y) < 3:
                continue
            used.add((x, y))
            self._banks.add((x, y))
        # Cops at the four corners; pick 2
        corners = [
            (1, 1), (self._WIDTH - 2, 1),
            (1, self._HEIGHT - 2), (self._WIDTH - 2, self._HEIGHT - 2),
        ]
        rng.shuffle(corners)
        self._cops = [list(p) for p in corners[: self._N_COPS]]

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

        # 2. Rob bank
        cell = (self._player_x, self._player_y)
        if cell in self._banks:
            self._banks.discard(cell)
            reward += self._progress_reward(self._WIN_TARGET)
            self._progress += 1
            self._message = f"Robbed! ({self._progress}/{self._WIN_TARGET})"
            if self._progress >= self._WIN_TARGET:
                self._on_won()
                return reward, self._game_over, info

        # 3. Cops move
        if self._tick_count % self._COP_MOVE_EVERY == 0:
            for c in self._cops:
                dx = _sign(self._player_x, c[0])
                dy = _sign(self._player_y, c[1])
                ncx, ncy = c[0], c[1]
                if abs(self._player_x - c[0]) >= abs(self._player_y - c[1]):
                    if not self._is_wall(c[0] + dx, c[1]):
                        ncx = c[0] + dx
                    elif not self._is_wall(c[0], c[1] + dy):
                        ncy = c[1] + dy
                else:
                    if not self._is_wall(c[0], c[1] + dy):
                        ncy = c[1] + dy
                    elif not self._is_wall(c[0] + dx, c[1]):
                        ncx = c[0] + dx
                c[0], c[1] = ncx, ncy

        # 4. Catch?
        for cx, cy in self._cops:
            if cx == self._player_x and cy == self._player_y:
                self._message = "Caught by the cops!"
                reward += self._death_reward()
                self._on_life_lost()
                return reward, True, info

        info["progress"] = self._progress
        info["banks_left"] = len(self._banks)
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        for y in range(self._HEIGHT):
            for x in range(self._WIDTH):
                if self._is_wall(x, y):
                    grid[y][x] = "█"
        for bx, by in self._banks:
            if 0 <= bx < self._WIDTH and 0 <= by < self._HEIGHT:
                grid[by][bx] = "B"
        for cx, cy in self._cops:
            if 0 <= cx < self._WIDTH and 0 <= cy < self._HEIGHT:
                grid[cy][cx] = "c"
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            pch = self._DIR_CHARS.get(self._player_dir, "@")
            grid[self._player_y][self._player_x] = pch

        symbols = {
            " ": "street",
            "█": "wall",
            "B": "bank",
            "c": "cop",
        }
        pch = self._DIR_CHARS.get(self._player_dir, "@")
        symbols[pch] = f"your car (facing {self._DIR_NAMES.get(self._player_dir, 'right')})"

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Robbed: {self._progress}/{self._WIN_TARGET}    "
            f"Score: {self._score:.3f}    "
            f"Banks: {len(self._banks)}    "
            f"Cops: {len(self._cops)}"
        )
        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Bank Heist on a 14x10 walled city. Drive your car (arrow "
            "shows facing) onto each of 3 banks (B) to rob them. 2 cops "
            "(c) spawn at corners and step 1 cell every 3 ticks toward you "
            "on the dominant axis. LEFT/RIGHT/UP/DOWN moves you 1 cell. "
            "Reward: +1/3 per bank robbed. Being caught by a cop is -1 "
            "terminal."
        )
