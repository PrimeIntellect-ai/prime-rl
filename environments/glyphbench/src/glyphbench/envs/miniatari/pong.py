"""miniatari Pong.

Identity: Two paddles, one ball, ricochet rallies. First to 3 wins.
Win condition: agent reaches 3 points.
Reward: Pattern C, +1/3 per agent point, -1/3 per opponent point.
Loss: opponent reaches 3 points.

Gym ID: glyphbench/miniatari-pong-v0

Random baseline (seed=0..29): success_rate=3%, mean_length=93, mean_return=-0.567
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniPongEnv(MiniatariBase):
    """Mini Pong: 16x10 court, first-to-3 wins.

    Agent paddle on right (col 14). Opponent paddle on left (col 1).
    Ball moves with integer velocity per tick. Each LLM action = 1 tick.
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP", "DOWN"),
        descriptions=(
            "do nothing this step",
            "move your paddle up one cell",
            "move your paddle down one cell",
        ),
    )

    default_max_turns = 200

    _WIDTH = 16
    _HEIGHT = 10
    _COURT_T = 1
    _COURT_B = 8
    _LEFT_PADDLE_X = 1
    _RIGHT_PADDLE_X = 14
    _PADDLE_SIZE = 3
    _WIN_TARGET = 3
    _OPP_TRACK_PROB = 0.75  # opponent track probability

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._ball_x: int = 0
        self._ball_y: int = 0
        self._ball_vx: int = 0
        self._ball_vy: int = 0
        self._paddle_left_y: int = 0
        self._paddle_right_y: int = 0
        self._agent_score: int = 0
        self._opp_score: int = 0
        self._serve_side: str = "agent"
        self._serving: bool = True

    def env_id(self) -> str:
        return "glyphbench/miniatari-pong-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._agent_score = 0
        self._opp_score = 0
        self._paddle_left_y = (self._COURT_T + self._COURT_B) // 2
        self._paddle_right_y = (self._COURT_T + self._COURT_B) // 2
        self._serve_side = "agent"
        self._serving = True
        self._reset_ball()

    def _reset_ball(self) -> None:
        self._ball_x = self._WIDTH // 2
        self._ball_y = (self._COURT_T + self._COURT_B) // 2
        # Auto-serve every point so the agent can't stall by refusing to FIRE.
        if self._serve_side == "agent":
            self._ball_vx = -1
        else:
            self._ball_vx = 1
        self._ball_vy = int(self.rng.integers(-1, 2))
        self._serving = False

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        # 1. Agent paddle movement
        if action_name == "UP":
            new_y = self._paddle_right_y - 1
            if new_y - self._PADDLE_SIZE // 2 >= self._COURT_T:
                self._paddle_right_y = new_y
        elif action_name == "DOWN":
            new_y = self._paddle_right_y + 1
            if new_y + self._PADDLE_SIZE // 2 <= self._COURT_B:
                self._paddle_right_y = new_y

        # 2. Move ball
        scored_for: str | None = None
        new_x = self._ball_x + self._ball_vx
        new_y = self._ball_y + self._ball_vy

        # Wall bounce (top/bottom)
        if new_y < self._COURT_T:
            new_y = self._COURT_T + (self._COURT_T - new_y)
            self._ball_vy = -self._ball_vy
        elif new_y > self._COURT_B:
            new_y = self._COURT_B - (new_y - self._COURT_B)
            self._ball_vy = -self._ball_vy

        # Right paddle (agent) collision
        if new_x >= self._RIGHT_PADDLE_X and self._ball_vx > 0:
            paddle_top = self._paddle_right_y - self._PADDLE_SIZE // 2
            paddle_bot = self._paddle_right_y + self._PADDLE_SIZE // 2
            if paddle_top <= new_y <= paddle_bot:
                self._ball_vx = -1
                offset = new_y - self._paddle_right_y
                self._ball_vy = max(-1, min(1, offset))
                new_x = self._RIGHT_PADDLE_X - 1
            else:
                scored_for = "opp"
        # Left paddle (opponent) collision
        elif new_x <= self._LEFT_PADDLE_X and self._ball_vx < 0:
            paddle_top = self._paddle_left_y - self._PADDLE_SIZE // 2
            paddle_bot = self._paddle_left_y + self._PADDLE_SIZE // 2
            if paddle_top <= new_y <= paddle_bot:
                self._ball_vx = 1
                offset = new_y - self._paddle_left_y
                self._ball_vy = max(-1, min(1, offset))
                new_x = self._LEFT_PADDLE_X + 1
            else:
                scored_for = "agent"

        self._ball_x = new_x
        self._ball_y = new_y

        # 3. Score / serve reset
        if scored_for == "agent":
            reward += self._agent_score_reward(self._WIN_TARGET)
            self._agent_score += 1
            self._message = "Point for you."
            self._serve_side = "opp"
            if self._agent_score >= self._WIN_TARGET:
                self._on_won()
            else:
                self._reset_ball()
        elif scored_for == "opp":
            reward += self._opp_score_reward(self._WIN_TARGET)
            self._opp_score += 1
            self._message = "Point for opponent."
            self._serve_side = "agent"
            if self._opp_score >= self._WIN_TARGET:
                self._on_life_lost()
            else:
                self._reset_ball()

        # 4. Opponent AI: track ball with high probability
        if not self._game_over and self.rng.random() < self._OPP_TRACK_PROB:
            if self._ball_y < self._paddle_left_y:
                new_yp = self._paddle_left_y - 1
                if new_yp - self._PADDLE_SIZE // 2 >= self._COURT_T:
                    self._paddle_left_y = new_yp
            elif self._ball_y > self._paddle_left_y:
                new_yp = self._paddle_left_y + 1
                if new_yp + self._PADDLE_SIZE // 2 <= self._COURT_B:
                    self._paddle_left_y = new_yp

        info["agent_score"] = self._agent_score
        info["opp_score"] = self._opp_score
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        # Build the grid with border, paddles, ball
        grid: list[list[str]] = []
        for y in range(self._HEIGHT):
            row: list[str] = []
            for x in range(self._WIDTH):
                if y == 0 or y == self._HEIGHT - 1:
                    row.append("─")
                elif x == 0 or x == self._WIDTH - 1:
                    row.append("│")
                else:
                    row.append(" ")
            grid.append(row)

        # Left paddle (3 cells: paddle_y - 1 .. paddle_y + 1)
        for dy in range(-(self._PADDLE_SIZE // 2), self._PADDLE_SIZE // 2 + 1):
            py = self._paddle_left_y + dy
            if self._COURT_T <= py <= self._COURT_B:
                grid[py][self._LEFT_PADDLE_X] = "P"

        # Right paddle (3 cells: paddle_y - 1 .. paddle_y + 1)
        for dy in range(-(self._PADDLE_SIZE // 2), self._PADDLE_SIZE // 2 + 1):
            py = self._paddle_right_y + dy
            if self._COURT_T <= py <= self._COURT_B:
                grid[py][self._RIGHT_PADDLE_X] = "Y"

        # Ball
        if 0 <= self._ball_x < self._WIDTH and 0 <= self._ball_y < self._HEIGHT:
            grid[self._ball_y][self._ball_x] = "O"

        legend = build_legend({
            "─": "court wall",
            "│": "court wall",
            "P": "opponent paddle",
            "Y": "your paddle",
            "O": "ball",
            " ": "court",
        })

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"You {self._agent_score} - {self._opp_score} Opp    "
            f"First to {self._WIN_TARGET}\n"
            f"Ball vel=({self._ball_vx:+d},{self._ball_vy:+d})"
        )

        return GridObservation(
            grid=grid_to_string(grid),
            legend=legend,
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Play Pong on a 16x10 court. You control the right paddle (Y); a "
            "scripted opponent (P) controls the left. The ball (O) bounces "
            "between you. Move UP/DOWN to intercept. The ball is auto-served "
            "after each point. Score by getting the ball past the opponent's "
            f"paddle. First player to {self._WIN_TARGET} points wins. "
            "The HUD shows Ball vel=(vx,vy): vx=-1 moves left, vx=+1 moves right; "
            "vy=-1 moves up, vy=+1 moves down. "
            f"Reward: +1/{self._WIN_TARGET} per point you score, "
            f"-1/{self._WIN_TARGET} per point conceded."
        )
