"""Atari Pong environment.

A 32x16 court (34x18 with border). Two paddles (3 cells tall), one ball.
Ball has integer velocity. First to 21 wins.

Gym ID: glyphbench/atari-pong-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation


class PongEnv(BaseGlyphEnv):
    """Atari Pong: paddle vs AI opponent. First to 21 wins.

    Court: 34 cols x 18 rows (32x16 interior + border).
    Left paddle (opponent) at col 1. Right paddle (agent) at col 32.
    Ball starts at center on serve.

    Actions: NOOP, FIRE, UP, DOWN, UP_FIRE, DOWN_FIRE
    Pattern C (adversarial first-to-W): ±1/_WIN_TARGET per point.
    First side to _WIN_TARGET wins (full-scope = 21).
    """

    # Pattern C full-scope target: first to 21 points.
    _WIN_TARGET: int = 21

    action_spec = ActionSpec(
        names=("NOOP", "FIRE", "UP", "DOWN", "UP_FIRE", "DOWN_FIRE"),
        descriptions=(
            "do nothing this step",
            "serve the ball (only meaningful at serve time)",
            "move your paddle up one cell",
            "move your paddle down one cell",
            "move paddle up and serve",
            "move paddle down and serve",
        ),
    )
    noop_action_name = "NOOP"

    # Court geometry (including border)
    _TOTAL_W = 34  # cols
    _TOTAL_H = 18  # rows
    _COURT_LEFT = 1  # first interior col
    _COURT_RIGHT = 32  # last interior col
    _COURT_TOP = 1  # first interior row
    _COURT_BOTTOM = 16  # last interior row

    # Paddle geometry
    _PADDLE_SIZE = 3
    _LEFT_PADDLE_COL = 1  # opponent paddle column
    _RIGHT_PADDLE_COL = 32  # agent paddle column

    # Ball speed cap
    _MAX_VX = 2
    _SPEED_UP_INTERVAL = 6  # every N hits, |vx| increases

    # Opponent AI
    _OPPONENT_TRACK_PROB = 0.7  # probability of tracking ball

    # Winning score (alias for backward-compat HUD).
    _WIN_SCORE = _WIN_TARGET

    def __init__(self, max_turns: int = 5000) -> None:
        super().__init__(max_turns=max_turns)
        self._ball_x: int = 0
        self._ball_y: int = 0
        self._ball_vx: int = 0
        self._ball_vy: int = 0
        self._paddle_left_y: int = 0  # center row of left paddle
        self._paddle_right_y: int = 0  # center row of right paddle
        self._score_left: int = 0  # opponent
        self._score_right: int = 0  # agent
        self._rally_hits: int = 0
        self._ball_speed_level: int = 1
        self._serving: bool = True
        self._serve_side: str = "agent"  # who serves next
        self._message: str = ""
        self._agent_progress: int = 0
        self._opp_progress: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-pong-v0"

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Pong.\n\n"
            "TASK\n"
            "Score 21 points before your opponent. You control the right paddle. "
            "The opponent controls the left paddle with AI.\n\n"
            "COURT\n"
            "34 columns x 18 rows (32x16 interior + border). You control the "
            "right paddle (▐); the opponent controls the left paddle (▌). "
            "Each paddle is 3 cells tall. The ball is O.\n\n"
            "PHYSICS\n"
            "The ball moves with integer velocity (vx, vy). Starting speed is "
            "|vx|=1. Every 6 paddle hits, |vx| increases by 1 (up to 2). The ball "
            "bounces off the top and bottom walls (vy flips sign). When the ball "
            "hits a paddle, vx flips sign and vy may change based on where the "
            "ball hits the paddle.\n\n"
            "SCORING\n"
            "Ball passes the left edge -> you score +1 (reward +1/21). "
            "Ball passes the right edge -> opponent scores +1 "
            "(reward -1/21). After each point, the scoring side "
            "serves. First to 21 wins (cumulative reward bound: "
            "[-1, +1]).\n\n"
            "SERVE\n"
            "At game start and after each point, the ball is at center. Press FIRE "
            "(or UP_FIRE/DOWN_FIRE) to serve. Until you serve, the ball stays put. "
            "UP and DOWN move your paddle without serving.\n\n"
            "HUD\n"
            "The HUD shows ball position, velocity, paddle positions, score, and "
            "rally hits. Use the velocity to predict where the ball will go.\n\n"
            + self.action_spec.render_for_prompt()
        )

    def _reset(self, seed: int) -> GridObservation:
        self._score_left = 0
        self._score_right = 0
        self._rally_hits = 0
        self._ball_speed_level = 1
        self._paddle_left_y = (self._COURT_TOP + self._COURT_BOTTOM) // 2
        self._paddle_right_y = (self._COURT_TOP + self._COURT_BOTTOM) // 2
        self._serving = True
        self._serve_side = "agent"
        self._message = ""
        self._agent_progress = 0
        self._opp_progress = 0
        self._reset_ball_to_center()
        return self._render_current_observation()

    def _reset_ball_to_center(self) -> None:
        """Place ball at court center, zero velocity."""
        self._ball_x = (self._COURT_LEFT + self._COURT_RIGHT) // 2
        self._ball_y = (self._COURT_TOP + self._COURT_BOTTOM) // 2
        self._ball_vx = 0
        self._ball_vy = 0

    def _serve_ball(self) -> None:
        """Launch ball from center with initial velocity."""
        if self._serve_side == "agent":
            # Ball goes toward opponent (left)
            self._ball_vx = -self._ball_speed_level
        else:
            # Ball goes toward agent (right)
            self._ball_vx = self._ball_speed_level
        vy_choices = [-1, 0, 1]
        self._ball_vy = int(self.rng.choice(vy_choices))
        self._serving = False

    def _on_paddle_hit(self) -> None:
        """Handle rally hit count and speed increase."""
        self._rally_hits += 1
        if (
            self._rally_hits % self._SPEED_UP_INTERVAL == 0
            and self._ball_speed_level < self._MAX_VX
        ):
                self._ball_speed_level += 1
                # Update ball speed (maintain direction)
                if self._ball_vx > 0:
                    self._ball_vx = self._ball_speed_level
                else:
                    self._ball_vx = -self._ball_speed_level

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        name = self.action_spec.names[action]
        self._message = ""
        reward = 0.0
        terminated = False

        # --- 1. Handle agent paddle movement ---
        wants_up = name in ("UP", "UP_FIRE")
        wants_down = name in ("DOWN", "DOWN_FIRE")
        wants_fire = name in ("FIRE", "UP_FIRE", "DOWN_FIRE")

        if wants_up:
            new_y = self._paddle_right_y - 1
            if new_y - self._PADDLE_SIZE // 2 >= self._COURT_TOP:
                self._paddle_right_y = new_y
        elif wants_down:
            new_y = self._paddle_right_y + 1
            if new_y + self._PADDLE_SIZE // 2 <= self._COURT_BOTTOM:
                self._paddle_right_y = new_y

        # --- 2. Handle serve ---
        if self._serving and wants_fire:
            self._serve_ball()

        # --- 3. Move ball (if not serving) ---
        if not self._serving:
            # Move ball step by step for each unit of |vx|
            scored = False
            for _ in range(abs(self._ball_vx)):
                dx = 1 if self._ball_vx > 0 else -1
                new_x = self._ball_x + dx

                # Check paddle collisions before moving
                # Right paddle (agent) collision
                if new_x >= self._RIGHT_PADDLE_COL and dx > 0:
                    paddle_top = self._paddle_right_y - self._PADDLE_SIZE // 2
                    paddle_bottom = self._paddle_right_y + self._PADDLE_SIZE // 2
                    if paddle_top <= self._ball_y <= paddle_bottom:
                        # Bounce off agent paddle
                        self._ball_vx = -abs(self._ball_vx)
                        # Adjust vy based on hit position
                        offset = self._ball_y - self._paddle_right_y
                        self._ball_vy = offset  # -1, 0, or 1
                        if self._ball_vy > 1:
                            self._ball_vy = 1
                        elif self._ball_vy < -1:
                            self._ball_vy = -1
                        self._on_paddle_hit()
                        break
                    else:
                        # Missed paddle -- opponent scores
                        self._score_left += 1
                        if self._opp_progress < self._WIN_TARGET:
                            reward = -1.0 / self._WIN_TARGET
                            self._opp_progress += 1
                        self._message = "Point for opponent."
                        scored = True
                        break

                # Left paddle (opponent) collision
                if new_x <= self._LEFT_PADDLE_COL and dx < 0:
                    paddle_top = self._paddle_left_y - self._PADDLE_SIZE // 2
                    paddle_bottom = self._paddle_left_y + self._PADDLE_SIZE // 2
                    if paddle_top <= self._ball_y <= paddle_bottom:
                        # Bounce off opponent paddle
                        self._ball_vx = abs(self._ball_vx)
                        offset = self._ball_y - self._paddle_left_y
                        self._ball_vy = offset
                        if self._ball_vy > 1:
                            self._ball_vy = 1
                        elif self._ball_vy < -1:
                            self._ball_vy = -1
                        self._on_paddle_hit()
                        break
                    else:
                        # Missed paddle -- agent scores
                        self._score_right += 1
                        if self._agent_progress < self._WIN_TARGET:
                            reward = 1.0 / self._WIN_TARGET
                            self._agent_progress += 1
                        self._message = "Point for you."
                        scored = True
                        break

                self._ball_x = new_x

            if not scored and self._ball_vy != 0:
                # Move vertically
                    new_y = self._ball_y + self._ball_vy
                    # Wall bounce
                    if new_y < self._COURT_TOP:
                        new_y = self._COURT_TOP + (self._COURT_TOP - new_y)
                        self._ball_vy = -self._ball_vy
                    elif new_y > self._COURT_BOTTOM:
                        new_y = self._COURT_BOTTOM - (new_y - self._COURT_BOTTOM)
                        self._ball_vy = -self._ball_vy
                    self._ball_y = new_y

            # After scoring, reset ball
            if scored:
                self._rally_hits = 0
                self._ball_speed_level = 1
                self._reset_ball_to_center()
                self._serving = True
                if reward > 0:
                    self._serve_side = "agent"
                else:
                    self._serve_side = "opponent"

                # Check for game over
                if (
                    self._agent_progress >= self._WIN_TARGET
                    or self._opp_progress >= self._WIN_TARGET
                ):
                    terminated = True

                # If opponent serves, auto-serve next step
                if self._serve_side == "opponent" and not terminated:
                    self._serve_ball()

        # --- 4. Move opponent AI ---
        if not self._serving and self.rng.random() < self._OPPONENT_TRACK_PROB:
                # Track ball
                if self._ball_y < self._paddle_left_y:
                    new_y = self._paddle_left_y - 1
                    if new_y - self._PADDLE_SIZE // 2 >= self._COURT_TOP:
                        self._paddle_left_y = new_y
                elif self._ball_y > self._paddle_left_y:
                    new_y = self._paddle_left_y + 1
                    if new_y + self._PADDLE_SIZE // 2 <= self._COURT_BOTTOM:
                        self._paddle_left_y = new_y
            # else: stay still (30% chance)

        # --- 5. Build info ---
        info: dict[str, Any] = {
            "ball_pos": (self._ball_x, self._ball_y),
            "ball_vel": (self._ball_vx, self._ball_vy),
            "paddle_left_y": self._paddle_left_y,
            "paddle_right_y": self._paddle_right_y,
            "score_left": self._score_left,
            "score_right": self._score_right,
            "rally_hits": self._rally_hits,
            "ball_speed_level": self._ball_speed_level,
            "serving": self._serve_side if self._serving else None,
            "game_over": terminated,
        }

        return self._render_current_observation(), reward, terminated, False, info

    def _render_current_observation(self) -> GridObservation:
        # Build 34x18 grid
        grid: list[list[str]] = []
        for y in range(self._TOTAL_H):
            row: list[str] = []
            for x in range(self._TOTAL_W):
                # Border
                if y == 0 or y == self._TOTAL_H - 1:
                    if x == 0 or x == self._TOTAL_W - 1:
                        row.append("┼")
                    else:
                        row.append("─")
                elif x == 0 or x == self._TOTAL_W - 1:
                    row.append("│")
                else:
                    row.append(" ")
            grid.append(row)

        # Draw left paddle (opponent)
        for dy in range(-self._PADDLE_SIZE // 2, self._PADDLE_SIZE // 2 + 1):
            py = self._paddle_left_y + dy
            if self._COURT_TOP <= py <= self._COURT_BOTTOM:
                grid[py][self._LEFT_PADDLE_COL] = "▌"

        # Draw right paddle (agent)
        for dy in range(-self._PADDLE_SIZE // 2, self._PADDLE_SIZE // 2 + 1):
            py = self._paddle_right_y + dy
            if self._COURT_TOP <= py <= self._COURT_BOTTOM:
                grid[py][self._RIGHT_PADDLE_COL] = "▐"

        # Draw ball
        if (self._COURT_LEFT <= self._ball_x <= self._COURT_RIGHT
                and self._COURT_TOP <= self._ball_y <= self._COURT_BOTTOM):
            grid[self._ball_y][self._ball_x] = "O"

        # Direction description
        dir_map = {
            (1, -1): "NE", (1, 0): "E", (1, 1): "SE",
            (-1, -1): "NW", (-1, 0): "W", (-1, 1): "SW",
            (0, -1): "N", (0, 0): "stationary", (0, 1): "S",
        }
        vx_sign = 1 if self._ball_vx > 0 else (-1 if self._ball_vx < 0 else 0)
        vy_sign = 1 if self._ball_vy > 0 else (-1 if self._ball_vy < 0 else 0)
        direction = dir_map.get((vx_sign, vy_sign), "unknown")

        serving_str = self._serve_side if self._serving else "none"
        vx_str = f"+{self._ball_vx}" if self._ball_vx >= 0 else str(self._ball_vx)
        vy_str = f"+{self._ball_vy}" if self._ball_vy >= 0 else str(self._ball_vy)

        hud = (
            f"Step: {self._turn} / {self.max_turns}\n"
            f"Score: Agent {self._score_right} -- Opponent {self._score_left}    "
            f"Serving: {serving_str}    "
            f"Ball vel=({vx_str},{vy_str}) direction={direction}\n"
            f"Rally hits: {self._rally_hits}    "
            f"Ball speed level: {self._ball_speed_level} (|vx|={self._ball_speed_level})    "
            f"Game to: {self._WIN_SCORE}"
        )

        legend = build_legend({
            "O": "ball",
            "▌": "left paddle (opponent)",
            "▐": "right paddle (you)",
            "│": "court side wall",
            "┼": "court corner",
            "─": "court wall (top/bottom)",
        })

        return GridObservation(
            grid=grid_to_string(grid),
            legend=legend,
            hud=hud,
            message=self._message,
        )
