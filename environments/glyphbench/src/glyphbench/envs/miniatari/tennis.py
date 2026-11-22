"""miniatari Tennis.

Identity: Top-down tennis on a small court; first to 3 games wins.
Win condition: agent wins 3 games (each game: 2 points, win-by-1).
Reward: Pattern C, +1/3 per game won, -1/3 per game lost.
Loss: opponent wins 3 games.

Gym ID: glyphbench/miniatari-tennis-v0

Random baseline (seed=0..29): success_rate=3%, mean_length=118, mean_return=-0.678
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniTennisEnv(MiniatariBase):
    """Mini Tennis: 18x10 top-down court, simplified scoring (best-of games).

    Agent plays the bottom half (rows 6..8); opponent plays the top half
    (rows 1..3). Net runs across row 5. Each "game" is decided by 2 points
    win-by-1 (so the agent must score 2 points before opp scores 2).
    Match: first to 3 games.
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP", "DOWN", "LEFT", "RIGHT"),
        descriptions=(
            "do nothing",
            "move up one cell",
            "move down one cell",
            "move left one cell",
            "move right one cell",
        ),
    )

    default_max_turns = 300

    _WIDTH = 18
    _HEIGHT = 10
    _COURT_L = 1
    _COURT_R = 16
    _COURT_T = 1
    _COURT_B = 8
    _NET_Y = 5
    _WIN_TARGET = 3      # games to win match
    _GAME_POINTS = 2     # points to win one game
    _OPP_TRACK_PROB = 0.7

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._opp_x: int = 0
        self._opp_y: int = 0
        self._ball_x: int = 0
        self._ball_y: int = 0
        self._ball_dx: int = 0
        self._ball_dy: int = 0
        self._ball_in_play: bool = False
        self._agent_points: int = 0
        self._opp_points: int = 0
        self._agent_score: int = 0  # games won
        self._opp_score: int = 0
        self._serve_side: str = "agent"

    def env_id(self) -> str:
        return "glyphbench/miniatari-tennis-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._agent_score = 0
        self._opp_score = 0
        self._agent_points = 0
        self._opp_points = 0
        self._serve_side = "agent"
        self._setup_point()

    def _setup_point(self) -> None:
        # Agent at bottom center, opponent at top center
        self._player_x = self._WIDTH // 2
        self._player_y = self._COURT_B - 1
        self._opp_x = self._WIDTH // 2
        self._opp_y = self._COURT_T + 1
        # Ball auto-served from server, traveling toward receiver
        if self._serve_side == "agent":
            self._ball_x = self._player_x
            self._ball_y = self._player_y - 1
            self._ball_dx = 0
            self._ball_dy = -1
        else:
            self._ball_x = self._opp_x
            self._ball_y = self._opp_y + 1
            self._ball_dx = 0
            self._ball_dy = 1
        self._ball_in_play = True

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        # Player movement (constrained to bottom half)
        nx, ny = self._player_x, self._player_y
        if action_name == "UP":
            ny -= 1
            self._player_dir = (0, -1)
        elif action_name == "DOWN":
            ny += 1
            self._player_dir = (0, 1)
        elif action_name == "LEFT":
            nx -= 1
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT":
            nx += 1
            self._player_dir = (1, 0)
        if self._COURT_L < nx < self._COURT_R and self._NET_Y < ny <= self._COURT_B:
            self._player_x, self._player_y = nx, ny

        # Move ball
        if self._ball_in_play:
            new_bx = self._ball_x + self._ball_dx
            new_by = self._ball_y + self._ball_dy

            # Side bounce
            if new_bx <= self._COURT_L:
                new_bx = self._COURT_L + 1
                self._ball_dx = abs(self._ball_dx)
            elif new_bx >= self._COURT_R:
                new_bx = self._COURT_R - 1
                self._ball_dx = -abs(self._ball_dx)

            self._ball_x, self._ball_y = new_bx, new_by

            # Player hit (ball moving down toward player)
            if (
                self._ball_dy > 0
                and abs(self._ball_x - self._player_x) <= 1
                and abs(self._ball_y - self._player_y) <= 1
            ):
                self._ball_dy = -1
                self._ball_dx = max(-1, min(1, self._ball_x - self._player_x))

            # Opponent hit (ball moving up toward opp)
            elif (
                self._ball_dy < 0
                and abs(self._ball_x - self._opp_x) <= 1
                and abs(self._ball_y - self._opp_y) <= 1
            ):
                self._ball_dy = 1
                self._ball_dx = max(-1, min(1, self._ball_x - self._opp_x))

            # Ball out top (agent scores point)
            if self._ball_y < self._COURT_T:
                self._agent_points += 1
                self._message = "Point won!"
                self._ball_in_play = False
                reward += self._check_game_award()
                if not self._game_over:
                    self._setup_point()
            # Ball out bottom (opp scores point)
            elif self._ball_y > self._COURT_B:
                self._opp_points += 1
                self._message = "Point lost!"
                self._ball_in_play = False
                reward += self._check_game_award()
                if not self._game_over:
                    self._setup_point()

        # Opponent AI: track ball horizontally
        if self._ball_in_play and self.rng.random() < self._OPP_TRACK_PROB:
            if self._opp_x < self._ball_x:
                if self._COURT_L < self._opp_x + 1 < self._COURT_R:
                    self._opp_x += 1
            elif self._opp_x > self._ball_x and self._COURT_L < self._opp_x - 1 < self._COURT_R:
                self._opp_x -= 1

        info["agent_games"] = self._agent_score
        info["opp_games"] = self._opp_score
        info["agent_pts"] = self._agent_points
        info["opp_pts"] = self._opp_points
        return reward, self._game_over, info

    def _check_game_award(self) -> float:
        """If a game has been decided, award reward and reset point counters."""
        reward = 0.0
        if self._agent_points >= self._GAME_POINTS and self._agent_points - self._opp_points >= 1:
            self._agent_score += 1
            self._agent_points = 0
            self._opp_points = 0
            reward += self._agent_score_reward(self._WIN_TARGET)
            self._serve_side = "opp"
            self._message = "Game! You win this game."
            if self._agent_score >= self._WIN_TARGET:
                self._on_won()
        elif self._opp_points >= self._GAME_POINTS and self._opp_points - self._agent_points >= 1:
            self._opp_score += 1
            self._agent_points = 0
            self._opp_points = 0
            reward += self._opp_score_reward(self._WIN_TARGET)
            self._serve_side = "agent"
            self._message = "Game! Opponent wins this game."
            if self._opp_score >= self._WIN_TARGET:
                self._on_life_lost()
        return reward

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = []
        for y in range(self._HEIGHT):
            row: list[str] = []
            for x in range(self._WIDTH):
                if y == 0 or y == self._HEIGHT - 1:
                    row.append("─")
                elif x == 0 or x == self._WIDTH - 1:
                    row.append("│")
                elif y == self._NET_Y:
                    row.append("=")
                else:
                    row.append(" ")
            grid.append(row)

        # Ball
        if 0 <= self._ball_x < self._WIDTH and 0 <= self._ball_y < self._HEIGHT:
            grid[self._ball_y][self._ball_x] = "O"

        # Opponent
        if 0 <= self._opp_x < self._WIDTH and 0 <= self._opp_y < self._HEIGHT:
            grid[self._opp_y][self._opp_x] = "P"

        # Agent (player) — drawn last
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            pch = self._DIR_CHARS.get(self._player_dir, "@")
            grid[self._player_y][self._player_x] = pch

        symbols = {
            "─": "court line",
            "│": "court line",
            "=": "net",
            "O": "ball",
            "P": "opponent",
            " ": "court",
        }
        pch = self._DIR_CHARS.get(self._player_dir, "@")
        symbols[pch] = f"you (facing {self._DIR_NAMES.get(self._player_dir, 'none')})"

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Games: You {self._agent_score} - {self._opp_score} Opp    "
            f"First to {self._WIN_TARGET}\n"
            f"Points this game: You {self._agent_points} - {self._opp_points} Opp    "
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
            "Top-down tennis on a small 18x10 court. You control your "
            "racket on the bottom half; the opponent (P) plays the top "
            "half. The ball (O) bounces between you. Move UP/DOWN/LEFT/"
            "RIGHT (within your half) to intercept it. The ball auto-"
            "serves each point. A game is won by being first to 2 points "
            "(win by 1). Match is first to 3 games. Reward: +1/3 per "
            "game you win, -1/3 per game you lose."
        )
