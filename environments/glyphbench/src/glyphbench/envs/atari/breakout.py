"""Atari Breakout environment.

Paddle and ball brick-breaking game.

Gym ID: glyphbench/atari-breakout-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase


class BreakoutEnv(AtariBase):
    """Breakout: paddle-and-ball brick-breaker.

    20x20 grid. Paddle at bottom, ball bouncing, wall of bricks at top.
    Break all bricks to clear the level.

    Actions: NOOP, FIRE, LEFT, RIGHT
    Pattern A: +1/_WIN_TARGET per brick broken (full-scope = 60 bricks
    in the full wall). No failure penalty.
    """

    action_spec = ActionSpec(
        names=("NOOP", "FIRE", "LEFT", "RIGHT"),
        descriptions=(
            "do nothing",
            "serve the ball",
            "move paddle left",
            "move paddle right",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 20
    _PADDLE_Y = 18
    _PADDLE_WIDTH = 3
    _BRICK_ROWS = 4
    _BRICK_START_Y = 3
    _BRICK_START_X = 2
    _BRICK_END_X = 17  # exclusive

    # Pattern A full-scope target: 60 bricks (4 rows x 15 columns).
    _WIN_TARGET: int = 60

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._paddle_x: int = 0  # left edge of paddle
        self._ball_x: int = 0
        self._ball_y: int = 0
        self._ball_dx: int = 0
        self._ball_dy: int = 0
        self._serving: bool = True
        self._bricks: set[tuple[int, int]] = set()
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-breakout-v0"

    def _reset(self, seed: int) -> GridObservation:
        self._progress_count = 0
        return super()._reset(seed)

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._serving = True

        # Borders
        for x in range(self._WIDTH):
            self._set_cell(x, 0, "─")
            self._set_cell(x, self._HEIGHT - 1, "─")
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "│")
            self._set_cell(self._WIDTH - 1, y, "│")
        self._set_cell(0, 0, "┼")
        self._set_cell(self._WIDTH - 1, 0, "┼")
        self._set_cell(0, self._HEIGHT - 1, "┼")
        self._set_cell(self._WIDTH - 1, self._HEIGHT - 1, "┼")

        # Bricks
        self._bricks = set()
        for row in range(self._BRICK_ROWS):
            y = self._BRICK_START_Y + row
            for x in range(self._BRICK_START_X, self._BRICK_END_X):
                self._bricks.add((x, y))
                self._set_cell(x, y, "█")

        # Paddle in center
        self._paddle_x = self._WIDTH // 2 - self._PADDLE_WIDTH // 2
        self._redraw_paddle()

        # Track player position on the paddle for base renderer
        self._player_x = self._paddle_x + self._PADDLE_WIDTH // 2
        self._player_y = self._PADDLE_Y

        # Ball on paddle
        self._ball_x = self._paddle_x + self._PADDLE_WIDTH // 2
        self._ball_y = self._PADDLE_Y - 1
        self._ball_dx = 0
        self._ball_dy = 0
        self._set_cell(self._ball_x, self._ball_y, "*")

    def _redraw_paddle(self) -> None:
        """Draw paddle at current position."""
        # Clear paddle row (interior only)
        for x in range(1, self._WIDTH - 1):
            if self._grid_at(x, self._PADDLE_Y) in ("=", "_"):
                self._set_cell(x, self._PADDLE_Y, " ")
        for x in range(self._PADDLE_WIDTH):
            px = self._paddle_x + x
            if 1 <= px < self._WIDTH - 1:
                self._set_cell(px, self._PADDLE_Y, "=")

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        # Move paddle
        if action_name == "LEFT" and self._paddle_x > 1:
            self._paddle_x -= 1
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT" and self._paddle_x + self._PADDLE_WIDTH < self._WIDTH - 1:
            self._paddle_x += 1
            self._player_dir = (1, 0)

        # Keep player position in sync with paddle
        self._player_x = self._paddle_x + self._PADDLE_WIDTH // 2
        self._player_y = self._PADDLE_Y

        # Serve
        if self._serving:
            self._ball_x = self._paddle_x + self._PADDLE_WIDTH // 2
            self._ball_y = self._PADDLE_Y - 1
            if action_name == "FIRE":
                self._serving = False
                self._ball_dx = 1 if self.rng.random() < 0.5 else -1
                self._ball_dy = -1
        else:
            # Clear old ball position
            if (self._ball_x, self._ball_y) not in self._bricks:
                self._set_cell(self._ball_x, self._ball_y, " ")

            # Move ball
            new_x = self._ball_x + self._ball_dx
            new_y = self._ball_y + self._ball_dy

            # Wall bounces (left/right)
            if new_x <= 0:
                new_x = 1
                self._ball_dx = -self._ball_dx
            elif new_x >= self._WIDTH - 1:
                new_x = self._WIDTH - 2
                self._ball_dx = -self._ball_dx

            # Top wall bounce
            if new_y <= 0:
                new_y = 1
                self._ball_dy = -self._ball_dy

            # Paddle collision
            paddle_end = self._paddle_x + self._PADDLE_WIDTH
            if new_y == self._PADDLE_Y and self._paddle_x <= new_x < paddle_end:
                new_y = self._PADDLE_Y - 1
                self._ball_dy = -1
                # Adjust dx based on hit position
                hit_pos = new_x - self._paddle_x
                if hit_pos == 0:
                    self._ball_dx = -1
                elif hit_pos == self._PADDLE_WIDTH - 1:
                    self._ball_dx = 1
                # else keep dx

            # Ball past paddle (lose life)
            on_paddle = (
                new_y == self._PADDLE_Y
                and self._paddle_x <= new_x < paddle_end
            )
            if new_y >= self._PADDLE_Y and not on_paddle:
                    self._on_life_lost()
                    self._serving = True
                    self._ball_x = self._paddle_x + self._PADDLE_WIDTH // 2
                    self._ball_y = self._PADDLE_Y - 1
                    self._ball_dx = 0
                    self._ball_dy = 0
                    self._message = "Ball lost! Lost a life."
                    self._redraw_field()
                    info["bricks_left"] = len(self._bricks)
                    return reward, self._game_over, info

            # Brick collision
            if (new_x, new_y) in self._bricks:
                self._bricks.discard((new_x, new_y))
                self._ball_dy = -self._ball_dy
                self._on_point_scored(1)
                if self._progress_count < self._WIN_TARGET:
                    reward += 1.0 / self._WIN_TARGET
                    self._progress_count += 1
                self._message = f"Brick! ({len(self._bricks)} left)"
                # Don't move into the brick cell, bounce back
                new_y = self._ball_y
            # Also check horizontal brick hit
            elif (new_x, self._ball_y) in self._bricks:
                self._bricks.discard((new_x, self._ball_y))
                self._ball_dx = -self._ball_dx
                self._on_point_scored(1)
                if self._progress_count < self._WIN_TARGET:
                    reward += 1.0 / self._WIN_TARGET
                    self._progress_count += 1
                new_x = self._ball_x

            self._ball_x = new_x
            self._ball_y = new_y

            # Win check
            if self._progress_count >= self._WIN_TARGET:
                self._game_over = True
                info["won"] = True
                self._message = "Wall cleared!"

            # Check level clear (regenerate; no reward bonus)
            if len(self._bricks) == 0 and not self._game_over:
                self._message = "Level clear!"
                self._level += 1
                self._generate_level(self._level)

        self._redraw_field()
        info["bricks_left"] = len(self._bricks)
        return reward, self._game_over, info

    def _redraw_field(self) -> None:
        """Redraw bricks, paddle, ball."""
        # Clear interior
        for y in range(1, self._HEIGHT - 1):
            for x in range(1, self._WIDTH - 1):
                self._set_cell(x, y, " ")

        # Draw bricks
        for bx, by in self._bricks:
            self._set_cell(bx, by, "█")

        # Draw paddle
        for x in range(self._PADDLE_WIDTH):
            px = self._paddle_x + x
            if 1 <= px < self._WIDTH - 1:
                self._set_cell(px, self._PADDLE_Y, "=")

        # Draw ball
        if 0 < self._ball_x < self._WIDTH - 1 and 0 < self._ball_y < self._HEIGHT - 1:
            self._set_cell(self._ball_x, self._ball_y, "*")

    def _advance_entities(self) -> None:
        # No entities to advance; ball is handled directly
        pass

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "─": "wall (top/bottom)",
            "│": "wall (side)",
            "┼": "corner",
            "█": "brick",
            "=": "paddle",
            "*": "ball",
            " ": "empty",
        }.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        dx, dy = self._ball_dx, self._ball_dy
        if dx > 0:
            dirname = "right" if dy == 0 else (
                "up-right" if dy < 0 else "down-right"
            )
        elif dx < 0:
            dirname = "left" if dy == 0 else (
                "up-left" if dy < 0 else "down-left"
            )
        else:
            dirname = "up" if dy < 0 else (
                "down" if dy > 0 else "stopped"
            )
        total = self._BRICK_ROWS * (
            self._BRICK_END_X - self._BRICK_START_X
        )
        remaining = len(self._bricks)
        ball = f"Ball vel=({dx},{dy}) dir={dirname}"
        bricks = f"Bricks: {remaining}/{total}"
        extra = f"{ball}    {bricks}"
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Break all bricks by bouncing the ball off your paddle. "
            "Press FIRE to serve. Move LEFT/RIGHT to position the paddle. "
            "Don't let the ball get past you."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Breakout.\n\n"
            "TASK\n"
            "Destroy every brick in the wall by deflecting a ball with "
            "your paddle. Clear the wall to advance to a new level.\n\n"
            "BOARD\n"
            "20x20 field with wall borders ('-', '|', '+' corners). "
            "A wall of bricks '#' fills the upper area. Your paddle "
            "'=' is 3 cells wide along the bottom. Ball is '*'.\n\n"
            "MECHANICS\n"
            "LEFT / RIGHT move the paddle 1 cell (clamped inside the "
            "walls). Until you FIRE, the ball sits on the paddle and "
            "follows it. FIRE serves with dy=-1 and dx=+/-1 (random). "
            "Once live, the ball moves 1 cell per step, bouncing off "
            "the left, right, and top walls (flip the relevant "
            "velocity). Hitting a brick removes it and flips dy. "
            "Hitting the paddle flips dy and sets dx based on hit "
            "position (left cell => dx=-1, right cell => dx=+1, middle "
            "keeps dx).\n\n"
            "SCORING\n"
            "+1 reward per brick broken. +5 bonus when the last brick "
            "is cleared (level-up bonus). Missing the ball ends the "
            "episode (single-life).\n\n"
            "TERMINATION\n"
            "Single-life: if the ball goes past the paddle row "
            "the episode ends. The episode also ends after "
            "max_turns.\n\n"
            "HUD\n"
            "Shows score, level, ball position and velocity, "
            "paddle x, and bricks remaining vs total.\n\n"
            + self.action_spec.render_for_prompt()
        )
