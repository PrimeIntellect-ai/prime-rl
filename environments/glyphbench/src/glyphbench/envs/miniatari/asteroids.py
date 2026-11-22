"""miniatari Asteroids.

Identity: Top-down ship in a small asteroid field; shoot rocks to win.
Win condition: destroy 5 asteroids.
Reward: Pattern A, +1/5 per asteroid destroyed.
Loss: collide with an asteroid (no -1 per Pattern A; just terminates).

Gym ID: glyphbench/miniatari-asteroids-v0

Random baseline (seed=0..29): success_rate=3%, mean_length=52, mean_return=+0.200
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniAsteroidsEnv(MiniatariBase):
    """Mini Asteroids: 14x12 toroidal field, 5 drifting rocks.

    Ship sits at center, can move in 4 directions or fire in its current
    facing direction. Asteroids drift in straight lines and wrap. A bullet
    travels at +2 cells/tick in the facing direction; resolves the same
    tick. Hitting any asteroid with the ship hull ends the run; clearing
    all 5 wins.
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP", "DOWN", "LEFT", "RIGHT", "FIRE"),
        descriptions=(
            "do nothing",
            "thrust up one cell and face up",
            "thrust down one cell and face down",
            "thrust left one cell and face left",
            "thrust right one cell and face right",
            "fire a bullet in your current facing direction",
        ),
    )

    default_max_turns = 200

    _WIDTH = 14
    _HEIGHT = 12
    _WIN_TARGET = 5
    _BULLET_RANGE = 6
    _N_ASTEROIDS = 5

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._asteroids: list[tuple[int, int, int, int]] = []  # (x, y, dx, dy)
        self._progress: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-asteroids-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._progress = 0
        self._player_x = self._WIDTH // 2
        self._player_y = self._HEIGHT // 2
        self._player_dir = (0, -1)  # default: facing up

        # Place asteroids on the perimeter, drifting toward the interior.
        rng = self.rng
        self._asteroids = []
        for _ in range(self._N_ASTEROIDS):
            for _attempt in range(20):
                # spawn near edge
                if rng.random() < 0.5:
                    x = int(rng.integers(0, self._WIDTH))
                    y = 0 if rng.random() < 0.5 else self._HEIGHT - 1
                    dy = 1 if y == 0 else -1
                    dx = int(rng.integers(-1, 2))
                else:
                    y = int(rng.integers(0, self._HEIGHT))
                    x = 0 if rng.random() < 0.5 else self._WIDTH - 1
                    dx = 1 if x == 0 else -1
                    dy = int(rng.integers(-1, 2))
                # avoid spawning on player
                if abs(x - self._player_x) <= 1 and abs(y - self._player_y) <= 1:
                    continue
                # avoid spawning on existing asteroid
                if any(ax == x and ay == y for ax, ay, _, _ in self._asteroids):
                    continue
                self._asteroids.append((x, y, dx, dy))
                break

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        # 1. Player action
        nx, ny = self._player_x, self._player_y
        if action_name == "UP":
            ny = (ny - 1) % self._HEIGHT
            self._player_dir = (0, -1)
        elif action_name == "DOWN":
            ny = (ny + 1) % self._HEIGHT
            self._player_dir = (0, 1)
        elif action_name == "LEFT":
            nx = (nx - 1) % self._WIDTH
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT":
            nx = (nx + 1) % self._WIDTH
            self._player_dir = (1, 0)
        self._player_x, self._player_y = nx, ny

        # 2. Fire bullet (resolves same tick)
        destroyed: list[int] = []
        if action_name == "FIRE":
            bdx, bdy = self._player_dir
            if bdx == 0 and bdy == 0:
                bdy = -1  # default fire direction
            bx, by = self._player_x, self._player_y
            for _ in range(self._BULLET_RANGE):
                bx = (bx + bdx) % self._WIDTH
                by = (by + bdy) % self._HEIGHT
                hit = False
                for i, (ax, ay, _adx, _ady) in enumerate(self._asteroids):
                    if i in destroyed:
                        continue
                    if ax == bx and ay == by:
                        destroyed.append(i)
                        hit = True
                        break
                if hit:
                    break
        if destroyed:
            self._asteroids = [a for i, a in enumerate(self._asteroids) if i not in destroyed]
            for _ in destroyed:
                reward += self._progress_reward(self._WIN_TARGET)
                self._progress += 1
                if self._progress >= self._WIN_TARGET:
                    self._on_won()
                    self._message = f"Wave clear! Destroyed {self._progress} rocks."
                    return reward, self._game_over, info
            self._message = f"Hit! ({self._progress}/{self._WIN_TARGET})"

        # 3. Move asteroids
        new_asteroids: list[tuple[int, int, int, int]] = []
        for ax, ay, adx, ady in self._asteroids:
            ax = (ax + adx) % self._WIDTH
            ay = (ay + ady) % self._HEIGHT
            new_asteroids.append((ax, ay, adx, ady))
        self._asteroids = new_asteroids

        # 4. Collision check (after asteroids move)
        for ax, ay, _adx, _ady in self._asteroids:
            if ax == self._player_x and ay == self._player_y:
                self._message = "Asteroid hit your ship!"
                self._game_over = True
                self._won = False
                return reward, True, info

        info["progress"] = self._progress
        info["asteroids_left"] = len(self._asteroids)
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        # Asteroids
        for ax, ay, _adx, _ady in self._asteroids:
            if 0 <= ax < self._WIDTH and 0 <= ay < self._HEIGHT:
                grid[ay][ax] = "O"
        # Player ship
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            pch = self._DIR_CHARS.get(self._player_dir, "@")
            grid[self._player_y][self._player_x] = pch

        symbols = {
            " ": "empty space",
            "O": "asteroid (drifting)",
        }
        pch = self._DIR_CHARS.get(self._player_dir, "@")
        symbols[pch] = f"your ship (facing {self._DIR_NAMES.get(self._player_dir, 'up')})"

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Destroyed: {self._progress}/{self._WIN_TARGET}    "
            f"Score: {self._score:.3f}    "
            f"Rocks: {len(self._asteroids)}"
        )

        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Asteroids on a 14x12 toroidal field. You pilot a ship "
            "(arrow shows facing). Movement actions translate the ship one "
            "cell AND set the facing direction. The screen wraps. FIRE "
            "shoots a bullet up to 6 cells in your facing direction; the "
            "first asteroid in line is destroyed that same tick. There "
            "are 5 asteroids (O), each drifting in a fixed direction. "
            "Destroy all 5 to win. Hitting an asteroid with your hull "
            "ends the run. Reward: +1/5 per asteroid destroyed."
        )
