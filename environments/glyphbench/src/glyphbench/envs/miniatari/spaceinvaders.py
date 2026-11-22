"""miniatari Space Invaders.

Identity: Bottom cannon defends against descending invaders.
Win condition: clear all 8 invaders in the wave.
Reward: Pattern D, +1/8 per kill; -1 if invaders reach the cannon row.
Loss: invaders reach bottom (terminal -1).

Gym ID: glyphbench/miniatari-spaceinvaders-v0

Random baseline (seed=0..29): success_rate=0%, mean_length=51, mean_return=-0.546
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniSpaceInvadersEnv(MiniatariBase):
    """Mini Space Invaders: 14x12 grid, single 8-invader wave.

    Cannon at the bottom row (y=11). Invaders form a 2x4 block
    starting at top-left, sweeping right→left→down. FIRE launches a
    travelling bullet (|) from the cannon row up the cannon's column.
    The bullet rises one row per tick and explodes on the first
    invader it reaches; only one bullet is in flight at a time. The
    wave advances one row every _ADVANCE_EVERY ticks AND drops one
    row each time it bounces off a wall (every _SWEEP_EVERY ticks).
    Game ends when invaders reach the cannon row (-1) or all are
    destroyed (+1).
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "FIRE"),
        descriptions=(
            "do nothing",
            "move cannon left one cell",
            "move cannon right one cell",
            "fire a bullet straight up your column",
        ),
    )

    default_max_turns = 200

    _WIDTH = 14
    _HEIGHT = 12
    _WIN_TARGET = 8
    _CANNON_Y = 11
    _INVADER_ROWS = 2
    _INVADER_COLS = 4
    # Slowed down so a real travelling bullet has time to traverse the
    # column between FIRE and impact (each kill takes up to ~9 ticks of
    # bullet flight). With the previous 8/2 the wave reached the cannon
    # at tick 48 — faster than 8 sequential bullets could clear. Tuned
    # so under pure NOOP the wave breaks the cannon row at ~140 ticks,
    # giving an active player time to clear all 8 invaders.
    _ADVANCE_EVERY = 24  # ticks per invader-row drop
    _SWEEP_EVERY = 5     # ticks per invader-column shift (also drives bounce-induced drops)

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._invaders: set[tuple[int, int]] = set()
        self._sweep_dir: int = 1
        self._wave_x: int = 0
        self._wave_y: int = 0
        self._tick_count: int = 0
        self._progress: int = 0
        # In-flight bullet, (x, y) or None. At most one bullet is in
        # flight; FIRE is ignored while a bullet is travelling.
        self._bullet: tuple[int, int] | None = None

    def env_id(self) -> str:
        return "glyphbench/miniatari-spaceinvaders-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._progress = 0
        self._tick_count = 0
        self._sweep_dir = 1
        self._wave_x = 1
        self._wave_y = 1
        self._bullet = None
        # 2x4 grid of invaders, spaced 2 cells apart horizontally
        self._invaders = set()
        for r in range(self._INVADER_ROWS):
            for c in range(self._INVADER_COLS):
                self._invaders.add((c * 2, r))
        self._player_x = self._WIDTH // 2
        self._player_y = self._CANNON_Y

    def _abs_invader_cells(self) -> set[tuple[int, int]]:
        return {(self._wave_x + ix, self._wave_y + iy) for ix, iy in self._invaders}

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._tick_count += 1

        # 1. Cannon move
        if action_name == "LEFT" and self._player_x > 0:
            self._player_x -= 1
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT" and self._player_x < self._WIDTH - 1:
            self._player_x += 1
            self._player_dir = (1, 0)

        # 2. FIRE: launch a bullet if no bullet is in flight. The bullet
        # spawns one row above the cannon and travels up at 1 row/tick.
        if action_name == "FIRE" and self._bullet is None:
            self._bullet = (self._player_x, self._CANNON_Y - 1)

        # 3. Bullet flight and impact. Resolve at the bullet's current
        # cell first (so a bullet that lands on an invader cell this
        # tick scores a hit), then advance.
        if self._bullet is not None:
            bx, by = self._bullet
            invader_cells = self._abs_invader_cells()
            if (bx, by) in invader_cells:
                # Find the invader at (bx, by) and remove it.
                for ix, iy in list(self._invaders):
                    if self._wave_x + ix == bx and self._wave_y + iy == by:
                        self._invaders.discard((ix, iy))
                        break
                self._bullet = None
                reward += self._progress_reward(self._WIN_TARGET)
                self._progress += 1
                self._message = f"Hit! ({self._progress}/{self._WIN_TARGET})"
                if self._progress >= self._WIN_TARGET:
                    self._on_won()
                    return reward, self._game_over, info
            else:
                # Advance bullet one row up.
                ny = by - 1
                if ny < 0:
                    self._bullet = None
                else:
                    self._bullet = (bx, ny)

        # 4. Wave sweeps every _SWEEP_EVERY ticks
        if self._tick_count % self._SWEEP_EVERY == 0 and self._invaders:
            xs = [self._wave_x + ix for ix, _iy in self._invaders]
            min_x, max_x = min(xs), max(xs)
            new_wave_x = self._wave_x + self._sweep_dir
            new_min = min_x + self._sweep_dir
            new_max = max_x + self._sweep_dir
            if new_min < 0 or new_max >= self._WIDTH:
                # Reverse and drop
                self._sweep_dir *= -1
                self._wave_y += 1
            else:
                self._wave_x = new_wave_x

        # 5. Wave drops every _ADVANCE_EVERY ticks (regardless of bounce)
        if self._tick_count % self._ADVANCE_EVERY == 0 and self._invaders:
            self._wave_y += 1

        # 6. Reach cannon row?
        if self._invaders:
            ys = [self._wave_y + iy for _ix, iy in self._invaders]
            max_y = max(ys)
            if max_y >= self._CANNON_Y:
                self._message = "Invaders reached the cannon row!"
                # Use += so any kill bonus earned earlier this tick is
                # preserved alongside the death penalty.
                reward += self._death_reward()
                self._on_life_lost()
                return reward, True, info

        info["progress"] = self._progress
        info["invaders_left"] = len(self._invaders)
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        # Cannon row line
        for x in range(self._WIDTH):
            grid[self._CANNON_Y][x] = "_"
        # Invaders
        for ix, iy in self._invaders:
            ax, ay = self._wave_x + ix, self._wave_y + iy
            if 0 <= ax < self._WIDTH and 0 <= ay < self._HEIGHT:
                grid[ay][ax] = "M"
        # Bullet (drawn before cannon so cannon never gets overdrawn)
        if self._bullet is not None:
            bx, by = self._bullet
            if 0 <= bx < self._WIDTH and 0 <= by < self._HEIGHT:
                grid[by][bx] = "|"
        # Cannon
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            grid[self._player_y][self._player_x] = "Y"

        symbols = {
            " ": "empty space",
            "_": "cannon row (defense line)",
            "M": "invader",
            "|": "your bullet",
            "Y": "your cannon",
        }

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Killed: {self._progress}/{self._WIN_TARGET}    "
            f"Score: {self._score:.3f}    "
            f"Sweep dir: {self._sweep_dir:+d}"
        )

        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Space Invaders on a 14x12 grid. A 2x4 wave of 8 invaders "
            "(M) sweeps left/right, dropping one row when it reaches an "
            f"edge (every {self._SWEEP_EVERY} ticks) AND every "
            f"{self._ADVANCE_EVERY} ticks regardless. Your cannon (Y) "
            "sits on the bottom row. LEFT/RIGHT slides the cannon. FIRE "
            "launches a bullet (|) from the cannon's column; the bullet "
            "rises one row per tick and explodes on the first invader "
            "it touches. Only one bullet may be in flight at a time — "
            "FIRE is ignored while a bullet is travelling. Clear all 8 "
            "invaders to win. If any invader reaches the cannon row you "
            "take a -1 terminal penalty. Reward: +1/8 per kill, -1 if "
            "the wave breaks through."
        )
