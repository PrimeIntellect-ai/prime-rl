"""miniatari BattleZone.

Identity: Top-down tank combat; enemies fire back.
Win condition: destroy 3 enemy tanks.
Reward: Pattern D, +1/3 per tank destroyed; -1 on death (enemy shell hits you).

Gym ID: glyphbench/miniatari-battlezone-v0

Random baseline (seed=0..29): success_rate=0%, mean_length=16, mean_return=-0.978
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


def _sign(a: int, b: int) -> int:
    return 1 if a > b else (-1 if a < b else 0)


class MiniBattleZoneEnv(MiniatariBase):
    """Mini BattleZone: 14x12 grid, 3 enemy tanks that pursue and fire.

    Player tank (Y, arrow facing) starts at center. 3 enemies (E) start at
    distinct edge positions and step 1 cell every 3 ticks toward the player.
    Each enemy fires a shell every 6 ticks: a shell (s) along the player's
    last-known direction (axis with greater distance), traveling 1 cell/tick
    until it leaves the field or hits the player. FIRE shoots in your facing
    direction up to 5 cells; first enemy in line is destroyed. Pattern D:
    +1/3 per kill; -1 on enemy shell hit (terminal).
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

    _WIDTH = 14
    _HEIGHT = 12
    _WIN_TARGET = 3
    _BULLET_RANGE = 5
    _ENEMY_MOVE_EVERY = 3
    _ENEMY_FIRE_EVERY = 6

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        # enemies: [x, y]
        self._enemies: list[list[int]] = []
        # shells: [x, y, dx, dy]
        self._shells: list[list[int]] = []
        self._tick_count: int = 0
        self._progress: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-battlezone-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._progress = 0
        self._tick_count = 0
        self._shells = []
        self._player_x = self._WIDTH // 2
        self._player_y = self._HEIGHT // 2
        self._player_dir = (0, -1)
        # 3 enemies at scattered edge positions
        rng = self.rng
        candidates = [
            (1, 1), (self._WIDTH - 2, 1),
            (1, self._HEIGHT - 2), (self._WIDTH - 2, self._HEIGHT - 2),
        ]
        rng.shuffle(candidates)
        self._enemies = [list(p) for p in candidates[:3]]

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

        # 2. Fire (resolves same tick)
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
                self._message = f"Enemy tank down! ({self._progress}/{self._WIN_TARGET})"
                if self._progress >= self._WIN_TARGET:
                    self._on_won()
                    return reward, self._game_over, info

        # 3. Enemies move (every K ticks toward player on dominant axis)
        if self._tick_count % self._ENEMY_MOVE_EVERY == 0:
            for e in self._enemies:
                dx = _sign(self._player_x, e[0])
                dy = _sign(self._player_y, e[1])
                if abs(self._player_x - e[0]) >= abs(self._player_y - e[1]):
                    e[0] += dx
                else:
                    e[1] += dy

        # 4. Enemies fire (every K ticks, all enemies fire at once toward player)
        if self._tick_count % self._ENEMY_FIRE_EVERY == 0:
            for ex, ey in self._enemies:
                dx = _sign(self._player_x, ex)
                dy = _sign(self._player_y, ey)
                if abs(self._player_x - ex) >= abs(self._player_y - ey):
                    sdx, sdy = dx, 0
                else:
                    sdx, sdy = 0, dy
                if sdx == 0 and sdy == 0:
                    continue
                # Spawn the shell at the enemy's cell; the same-tick
                # movement loop below carries it 1 cell toward the
                # player. This avoids the shell skipping the enemy's
                # cell on its first tick of flight.
                self._shells.append([ex, ey, sdx, sdy])

        # 5. Move shells; check hit
        new_shells: list[list[int]] = []
        for sx, sy, sdx, sdy in self._shells:
            if 0 <= sx < self._WIDTH and 0 <= sy < self._HEIGHT:
                if sx == self._player_x and sy == self._player_y:
                    self._message = "Hit by enemy shell!"
                    reward += self._death_reward()
                    self._on_life_lost()
                    return reward, True, info
                new_shells.append([sx + sdx, sy + sdy, sdx, sdy])
        self._shells = new_shells

        # 6. Collision (an enemy stepped onto player)
        for ex, ey in self._enemies:
            if ex == self._player_x and ey == self._player_y:
                self._message = "Enemy tank rammed you!"
                reward += self._death_reward()
                self._on_life_lost()
                return reward, True, info

        info["progress"] = self._progress
        info["enemies_left"] = len(self._enemies)
        info["shells"] = len(self._shells)
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        for ex, ey in self._enemies:
            if 0 <= ex < self._WIDTH and 0 <= ey < self._HEIGHT:
                grid[ey][ex] = "E"
        for sx, sy, _sdx, _sdy in self._shells:
            if 0 <= sx < self._WIDTH and 0 <= sy < self._HEIGHT:
                grid[sy][sx] = "s"
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            pch = self._DIR_CHARS.get(self._player_dir, "@")
            grid[self._player_y][self._player_x] = pch

        symbols = {
            " ": "ground",
            "E": "enemy tank",
            "s": "enemy shell",
        }
        pch = self._DIR_CHARS.get(self._player_dir, "@")
        symbols[pch] = f"your tank (facing {self._DIR_NAMES.get(self._player_dir, 'up')})"

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Killed: {self._progress}/{self._WIN_TARGET}    "
            f"Score: {self._score:.3f}    "
            f"Enemies: {len(self._enemies)}    "
            f"Shells: {len(self._shells)}"
        )
        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini BattleZone on a 14x12 battlefield. You command a tank "
            "(arrow shows facing) at the center. 3 enemy tanks (E) start "
            "at distinct corners and step 1 cell every 3 ticks toward you "
            "along the dominant axis. Every 6 ticks each enemy fires a "
            "shell (s) toward you, traveling 1 cell/tick. LEFT/RIGHT/UP/"
            "DOWN moves you and sets your facing. FIRE shoots up to 5 "
            "cells in your facing direction; first enemy in line is "
            "destroyed. Reward: +1/3 per kill. Getting hit by a shell or "
            "rammed by an enemy is -1 terminal."
        )
