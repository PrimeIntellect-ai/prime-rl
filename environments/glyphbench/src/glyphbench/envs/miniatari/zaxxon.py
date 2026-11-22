"""miniatari Zaxxon.

Identity: Fly through a fortress shooting turrets that fire back.
Win condition: destroy all 6 turrets.
Reward: Pattern D, +1/6 per turret destroyed; -1 if hit by turret fire.

Gym ID: glyphbench/miniatari-zaxxon-v0

Random baseline (seed=0..29): success_rate=0%, mean_length=27, mean_return=-0.767
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniZaxxonEnv(MiniatariBase):
    """Mini Zaxxon: 16x10 corridor with 6 stationary turrets.

    Player ship (Y, arrow facing right) flies through a 16x10 corridor.
    6 turrets (T) sit at scattered cells in the middle 12 columns. Each
    turret fires a bullet (b) every 5 ticks straight left along its row;
    bullets travel 1 cell/tick. FIRE shoots a torpedo straight right
    from the player; first turret in line on the same row dies. Pattern
    D: +1/6 per kill, -1 on bullet hit.
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP", "DOWN", "FIRE"),
        descriptions=(
            "do nothing",
            "rise one row",
            "dive one row",
            "fire a torpedo straight right along your row",
        ),
    )

    default_max_turns = 300

    _WIDTH = 16
    _HEIGHT = 10
    _N_TURRETS = 6
    _WIN_TARGET = _N_TURRETS
    _PLAYER_X = 1
    _TURRET_FIRE_EVERY = 5

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._turrets: list[list[int]] = []  # [x, y]
        self._bullets: list[list[int]] = []  # [x, y]
        self._tick_count: int = 0
        self._progress: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-zaxxon-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._tick_count = 0
        self._progress = 0
        self._bullets = []
        rng = self.rng
        used: set[tuple[int, int]] = set()
        self._turrets = []
        # Turrets between cols 4 and 14, rows 1..H-2
        while len(self._turrets) < self._N_TURRETS:
            x = int(rng.integers(4, self._WIDTH - 1))
            y = int(rng.integers(1, self._HEIGHT - 1))
            if (x, y) in used:
                continue
            used.add((x, y))
            self._turrets.append([x, y])
        self._player_x = self._PLAYER_X
        self._player_y = self._HEIGHT // 2
        self._player_dir = (1, 0)

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._tick_count += 1

        # 1. Player move
        if action_name == "UP" and self._player_y > 0:
            self._player_y -= 1
        elif action_name == "DOWN" and self._player_y < self._HEIGHT - 1:
            self._player_y += 1

        # 2. Fire torpedo
        if action_name == "FIRE":
            target: int | None = None
            target_x: int = self._WIDTH
            for i, (tx, ty) in enumerate(self._turrets):
                if ty == self._player_y and tx > self._player_x and tx < target_x:
                    target = i
                    target_x = tx
            if target is not None:
                self._turrets.pop(target)
                reward += self._progress_reward(self._WIN_TARGET)
                self._progress += 1
                self._message = f"Turret down! ({self._progress}/{self._WIN_TARGET})"
                if self._progress >= self._WIN_TARGET:
                    self._on_won()
                    return reward, self._game_over, info

        # 3. Turrets fire (every K ticks all turrets fire left)
        if self._tick_count % self._TURRET_FIRE_EVERY == 0:
            for tx, ty in self._turrets:
                self._bullets.append([tx - 1, ty])

        # 4. Move bullets left
        new_bullets: list[list[int]] = []
        for bx, by in self._bullets:
            if bx == self._player_x and by == self._player_y:
                self._message = "Hit by turret fire!"
                reward += self._death_reward()
                self._on_life_lost()
                return reward, True, info
            nbx = bx - 1
            if nbx < 0:
                continue
            new_bullets.append([nbx, by])
        self._bullets = new_bullets

        info["progress"] = self._progress
        info["turrets_left"] = len(self._turrets)
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        for tx, ty in self._turrets:
            if 0 <= tx < self._WIDTH and 0 <= ty < self._HEIGHT:
                grid[ty][tx] = "T"
        for bx, by in self._bullets:
            if 0 <= bx < self._WIDTH and 0 <= by < self._HEIGHT:
                grid[by][bx] = "b"
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            grid[self._player_y][self._player_x] = "Y"

        symbols = {
            " ": "open sky",
            "T": "turret",
            "b": "turret bullet",
            "Y": "your ship",
        }

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Killed: {self._progress}/{self._WIN_TARGET}    "
            f"Score: {self._score:.3f}    "
            f"Turrets: {len(self._turrets)}"
        )
        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Zaxxon on a 16x10 corridor. Your ship (Y) sits at column "
            "1 and can rise/dive (UP/DOWN). 6 turrets (T) sit at scattered "
            "interior cells (cols 4-14). FIRE shoots a torpedo straight "
            "right along your row, destroying the nearest turret on that "
            "row. Every 5 ticks all turrets fire bullets (b) left along "
            "their row, traveling 1 cell/tick. Reward: +1/6 per turret. "
            "Being hit by a bullet is -1 terminal."
        )
