"""miniatari Breakout.

Identity: Paddle-and-ball brick-breaker on a tight 16x10 grid.
Win condition: clear all bricks in the single brick row.
Reward: Pattern A, +1/N per brick broken (N = full brick-row width).
Loss: ball falls off bottom (single life).

Gym ID: glyphbench/miniatari-breakout-v0

Random baseline (seed=0..29): success_rate=0%, mean_length=24, mean_return=+0.000
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniBreakoutEnv(MiniatariBase):
    """Mini Breakout: 16x10 court, single full-width brick row.

    Paddle at row 8. Bricks fill row 2 across the entire court interior
    (cols 1..14, 14 bricks placed). Ball auto-serves on reset. Single
    life: missing the ball ends the episode (no -1 penalty per Pattern A).
    Win after breaking 11 bricks (the parity-stable subset).

    Geometry note: with the original 6-brick block at cols 5..10 and a
    parity-1 ball trajectory from serve (8, 7) the ball never visited any
    brick column. Widening the row to the full court interior makes
    bricks reachable from the natural serve trajectory; the win threshold
    is 11/14 because the residual {3} or {7, 8} parity cells are mutually
    exclusive across serve directions and never reachable in the same
    episode under paddle-tracking.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT"),
        descriptions=(
            "do nothing this tick",
            "move paddle left one cell",
            "move paddle right one cell",
        ),
    )

    default_max_turns = 200

    _WIDTH = 16
    _HEIGHT = 10
    _PADDLE_Y = 8
    _PADDLE_W = 3
    _BRICK_Y = 2
    _BRICK_X_LO = 1
    _BRICK_X_HI = 14  # inclusive — full court interior (14 bricks placed)
    # Win after the agent breaks 11 bricks. Placing 14 keeps the wave
    # visually full while the lower target absorbs the parity dead-cell
    # cluster (cols 3 / 7-8 are mutually exclusive across serve directions
    # and never reachable in the same episode).
    _WIN_TARGET = 11

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._paddle_x: int = 0
        self._ball_x: int = 0
        self._ball_y: int = 0
        self._ball_dx: int = 0
        self._ball_dy: int = 0
        self._bricks: set[int] = set()
        self._progress: int = 0
        self._serving: bool = True

    def env_id(self) -> str:
        return "glyphbench/miniatari-breakout-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._bricks = set(range(self._BRICK_X_LO, self._BRICK_X_HI + 1))
        self._progress = 0
        self._paddle_x = self._WIDTH // 2 - self._PADDLE_W // 2
        self._ball_x = self._paddle_x + self._PADDLE_W // 2
        self._ball_y = self._PADDLE_Y - 1
        # Random initial ball direction (toward bricks)
        self._ball_dx = -1 if self.rng.random() < 0.5 else 1
        self._ball_dy = -1
        self._serving = False  # auto-served (no FIRE button needed)

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        # 1. Move paddle
        if action_name == "LEFT" and self._paddle_x > 0:
            self._paddle_x -= 1
        elif action_name == "RIGHT" and self._paddle_x + self._PADDLE_W < self._WIDTH:
            self._paddle_x += 1

        # 2. Move ball
        new_x = self._ball_x + self._ball_dx
        new_y = self._ball_y + self._ball_dy

        # Side wall bounce
        if new_x < 0:
            new_x = 0
            self._ball_dx = -self._ball_dx
        elif new_x >= self._WIDTH:
            new_x = self._WIDTH - 1
            self._ball_dx = -self._ball_dx

        # Top wall bounce
        if new_y < 0:
            new_y = 0
            self._ball_dy = -self._ball_dy

        # Brick collision (only at row _BRICK_Y)
        if new_y == self._BRICK_Y and new_x in self._bricks:
            self._bricks.discard(new_x)
            self._ball_dy = -self._ball_dy
            new_y = self._BRICK_Y + 1
            reward += self._progress_reward(self._WIN_TARGET)
            self._progress += 1
            self._message = f"Brick broken! ({self._WIN_TARGET - self._progress} left)"
            if self._progress >= self._WIN_TARGET:
                self._on_won()
                self._ball_x, self._ball_y = new_x, new_y
                return reward, self._game_over, info

        # Paddle collision (ball hits paddle row going down)
        if new_y == self._PADDLE_Y and self._ball_dy > 0:
            paddle_end = self._paddle_x + self._PADDLE_W
            if self._paddle_x <= new_x < paddle_end:
                self._ball_dy = -1
                hit_pos = new_x - self._paddle_x
                if hit_pos == 0:
                    self._ball_dx = -1
                elif hit_pos == self._PADDLE_W - 1:
                    self._ball_dx = 1
                new_y = self._PADDLE_Y - 1

        # Ball past paddle row -> game over (no -1 per Pattern A)
        if new_y > self._PADDLE_Y:
            self._ball_x, self._ball_y = new_x, new_y
            self._message = "Ball lost!"
            self._game_over = True
            self._won = False
            return reward, True, info

        self._ball_x, self._ball_y = new_x, new_y
        info["bricks_left"] = len(self._bricks)
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        # Bricks
        for bx in self._bricks:
            grid[self._BRICK_Y][bx] = "█"
        # Paddle
        for dx in range(self._PADDLE_W):
            px = self._paddle_x + dx
            if 0 <= px < self._WIDTH:
                grid[self._PADDLE_Y][px] = "="
        # Ball
        if 0 <= self._ball_x < self._WIDTH and 0 <= self._ball_y < self._HEIGHT:
            grid[self._ball_y][self._ball_x] = "*"

        symbols = {
            " ": "empty",
            "█": "brick",
            "=": "your paddle",
            "*": "ball",
        }

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Score: {self._score:.3f}    "
            f"Broken: {self._progress}/{self._WIN_TARGET}    "
            f"Bricks left: {len(self._bricks)}    "
            f"Ball vel=({self._ball_dx:+d},{self._ball_dy:+d})"
        )

        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Breakout on a 16x10 field. A full-width row of 14 bricks "
            "(█) fills row 2 (cols 1..14); your 3-cell paddle (=) is at "
            "row 8. The ball (*) is auto-served and travels 1 cell per "
            "tick. Move LEFT/RIGHT to intercept and bounce it off the "
            "paddle into the bricks. The ball reflects off the top, left, "
            "and right walls; off bricks (removing the brick); and off "
            "your paddle (with hit-position spin: hitting with the "
            "leftmost cell sends the ball left, rightmost cell sends it "
            "right, middle preserves dx). Break 11 of the 14 bricks to "
            "win (a couple of cells are unreachable from a single serve "
            "due to ball-trajectory parity). If the ball drops below the "
            "paddle row, the episode ends with no further reward. The HUD "
            "shows Ball vel=(dx,dy): dx=-1 moves left, dx=+1 moves right; "
            "dy=-1 moves up, dy=+1 moves down. "
            "Reward: +1/11 per brick destroyed."
        )
