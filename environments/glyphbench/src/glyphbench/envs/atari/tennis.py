"""Atari Tennis environment.

Top-down tennis court with ball physics and AI opponent.

Gym ID: glyphbench/atari-tennis-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec

from .base import AtariBase

_SCORE_NAMES = ["0", "15", "30", "40"]

class TennisEnv(AtariBase):
    """Tennis: top-down tennis with standard scoring.

    20x24 grid. Ball bounces, AI opponent returns.
    Standard scoring within games (15-30-40-game), but the
    match terminates when either side reaches _WIN_TARGET total
    points scored (Pattern C).

    Actions: NOOP, UP, DOWN, LEFT, RIGHT
    Pattern C (adversarial first-to-W): ±1/_WIN_TARGET per
    point won/lost.
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP", "DOWN", "LEFT", "RIGHT"),
        descriptions=(
            "do nothing",
            "move up",
            "move down",
            "move left",
            "move right",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 24
    _COURT_L = 1
    _COURT_R = 18
    _COURT_T = 1
    _COURT_B = 22
    _NET_Y = 12

    # Pattern C full-scope: first side to 18 total points wins
    # the match. Each point is worth ±1/18 reward.
    _WIN_TARGET: int = 18

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._opp_x: int = 0
        self._opp_y: int = 0
        self._ball_x: float = 0.0
        self._ball_y: float = 0.0
        self._ball_dx: float = 0.0
        self._ball_dy: float = 0.0
        self._player_points: int = 0
        self._opp_points: int = 0
        self._player_games: int = 0
        self._opp_games: int = 0
        self._serving: bool = True
        self._serve_side: str = "player"
        self._rally_active: bool = False
        # Pattern C running totals (separate from per-game points
        # which reset every game).
        self._agent_progress: int = 0
        self._opp_progress: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-tennis-v0"

    def _reset(self, seed: int):
        self._agent_progress = 0
        self._opp_progress = 0
        return super()._reset(seed)

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._player_points = 0
        self._opp_points = 0
        self._player_games = 0
        self._opp_games = 0
        self._serve_side = "player"
        self._lives = 1  # not used traditionally
        self._setup_point()
        self._redraw()

    def _setup_point(self) -> None:
        # Player at bottom
        self._player_x = self._WIDTH // 2
        self._player_y = self._COURT_B - 2
        # Opponent at top
        self._opp_x = self._WIDTH // 2
        self._opp_y = self._COURT_T + 2
        # Ball on server
        if self._serve_side == "player":
            self._ball_x = float(self._player_x)
            self._ball_y = float(self._player_y - 1)
            self._ball_dx = 0.0
            self._ball_dy = -1.5
        else:
            self._ball_x = float(self._opp_x)
            self._ball_y = float(self._opp_y + 1)
            self._ball_dx = 0.0
            self._ball_dy = 1.5
        self._rally_active = True
        self._serving = True

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        # Player movement
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
        if (
            self._COURT_L < nx < self._COURT_R
            and self._NET_Y + 1 < ny < self._COURT_B
        ):
            self._player_x, self._player_y = nx, ny

        # Move ball
        if self._rally_active:
            self._ball_x += self._ball_dx
            self._ball_y += self._ball_dy
            bx = int(round(self._ball_x))
            by = int(round(self._ball_y))
            # Side wall bounce
            if bx <= self._COURT_L or bx >= self._COURT_R:
                self._ball_dx = -self._ball_dx
                self._ball_x += self._ball_dx

            # Player hit
            if (
                by >= self._player_y - 1
                and self._ball_dy > 0
                and abs(bx - self._player_x) <= 2
            ):
                self._ball_dy = -abs(self._ball_dy)
                self._ball_dx = float(bx - self._player_x) * 0.5
                self._serving = False
            # Opponent hit
            if (
                by <= self._opp_y + 1
                and self._ball_dy < 0
                and abs(bx - self._opp_x) <= 2
            ):
                self._ball_dy = abs(self._ball_dy)
                self._ball_dx = float(bx - self._opp_x) * 0.5

            # Ball out bottom = opponent scores
            if by > self._COURT_B:
                self._rally_active = False
                self._opp_points += 1
                if self._opp_progress < self._WIN_TARGET:
                    reward = -1.0 / self._WIN_TARGET
                    self._opp_progress += 1
                self._message = "Out! Opponent scores."
                self._check_game()
                self._setup_point()
            # Ball out top = player scores
            elif by < self._COURT_T:
                self._rally_active = False
                self._player_points += 1
                if self._agent_progress < self._WIN_TARGET:
                    reward = 1.0 / self._WIN_TARGET
                    self._agent_progress += 1
                self._on_point_scored(1)
                self._message = "Point! You score."
                self._check_game()
                self._setup_point()

        # Opponent AI
        self._move_opponent()
        self._redraw()

        info["p_pts"] = self._player_points
        info["o_pts"] = self._opp_points
        info["p_games"] = self._player_games
        info["o_games"] = self._opp_games
        info["agent_progress"] = self._agent_progress
        info["opp_progress"] = self._opp_progress
        terminated = (
            self._agent_progress >= self._WIN_TARGET
            or self._opp_progress >= self._WIN_TARGET
        )
        return reward, terminated, info

    def _check_game(self) -> None:
        pp, op = self._player_points, self._opp_points
        won = lost = False
        if pp >= 3 and op >= 3:
            won, lost = pp - op >= 2, op - pp >= 2
        else:
            won, lost = pp >= 4, op >= 4
        if won or lost:
            if won:
                self._player_games += 1
                self._message = "Game! You win."
            else:
                self._opp_games += 1
                self._message = "Game! Opponent wins."
            self._player_points = 0
            self._opp_points = 0
            self._serve_side = (
                "opponent"
                if self._serve_side == "player"
                else "player"
            )

    def _move_opponent(self) -> None:
        bx = int(round(self._ball_x))
        rng = self.rng
        if rng.random() < 0.7:
            if self._opp_x < bx:
                nx = self._opp_x + 1
            elif self._opp_x > bx:
                nx = self._opp_x - 1
            else:
                nx = self._opp_x
            if self._COURT_L < nx < self._COURT_R:
                self._opp_x = nx

    def _redraw(self) -> None:
        for y in range(self._HEIGHT):
            for x in range(self._WIDTH):
                self._set_cell(x, y, " ")

        # Court borders
        for x in range(self._WIDTH):
            self._set_cell(x, self._COURT_T, "─")
            self._set_cell(x, self._COURT_B, "─")
        for y in range(self._COURT_T, self._COURT_B + 1):
            self._set_cell(self._COURT_L, y, "│")
            self._set_cell(self._COURT_R, y, "│")

        # Net
        for x in range(self._COURT_L + 1, self._COURT_R):
            self._set_cell(x, self._NET_Y, "█")

        # Ball
        bx = int(round(self._ball_x))
        by = int(round(self._ball_y))
        if (
            self._COURT_L < bx < self._COURT_R
            and self._COURT_T < by < self._COURT_B
        ):
            self._set_cell(bx, by, "o")

        # Opponent
        self._set_cell(self._opp_x, self._opp_y, "V")

    def _advance_entities(self) -> None:
        pass

    def _render_current_observation(self, **kw: Any):  # type: ignore[override]
        """Append tennis-specific HUD on top of base HUD.

        Base HUD provides ``Step: T / N`` (X4 cross-cutting).
        """
        from glyphbench.core.observation import GridObservation

        obs = super()._render_current_observation()
        pp = self._fmt_pts(self._player_points)
        op = self._fmt_pts(self._opp_points)
        bdx = round(self._ball_dx, 1)
        bdy = round(self._ball_dy, 1)
        ball = f"Ball vel=({bdx},{bdy})"
        serve = ""
        if self._serving:
            serve = f"  Serving: {self._serve_side}"
        extra = (
            f"You {pp} - {op} Opp | "
            f"Games: {self._player_games}-{self._opp_games}"
            f"\n{ball}{serve}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid,
            legend=obs.legend,
            hud=new_hud,
            message=obs.message,
        )

    @staticmethod
    def _fmt_pts(pts: int) -> str:
        if pts < 4:
            return _SCORE_NAMES[pts]
        return str(pts * 10)

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "─": "baseline",
            "│": "sideline",
            "█": "net",
            "o": "ball",
            "V": "opponent",
            " ": "court",
        }.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "Play tennis. Move to intercept the ball. "
            "Standard scoring: 15-30-40-game. "
            "First to 6 games wins the set."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Tennis.\n\n"
            "TASK\n"
            "Top-down tennis with classic scoring: 15/30/40/game, "
            "win-by-2 after deuce, first to 6 games wins the set.\n\n"
            "BOARD\n"
            "20x24 court. Sidelines '|', baselines '-'. Net '#' "
            "divides the court; the opponent 'V' plays one half and "
            "you play the other (the [Grid] shows your half). Ball "
            "'o' bounces back and forth.\n\n"
            "MECHANICS\n"
            "UP/DOWN/LEFT/RIGHT move 1 cell inside your court "
            "half. The ball moves per step with float velocity, "
            "bouncing off sidelines. When the ball comes within 2 "
            "columns of your position while traveling toward you, "
            "you hit it: dy flips sign; dx = (ball.x - you.x)*0.5. "
            "Opponent AI returns similarly. Ball leaving top = you "
            "score; leaving bottom = opponent scores.\n\n"
            "SCORING\n"
            "Pattern C (first-to-18): +1/18 reward per point you "
            "win (ball leaves opponent's baseline). -1/18 reward "
            "per point lost. The classic 15/30/40/game scoring is "
            "kept for HUD only -- the match terminates the moment "
            "either side reaches 18 total points scored. "
            "Cumulative reward bound: [-1, +1].\n\n"
            "TERMINATION\n"
            "Episode ends when either player accumulates 18 "
            "points. No life system. Max_turns also ends it.\n\n"
            "HUD\n"
            "Shows your point/game score, opponent's, ball "
            "position and velocity, and serving side.\n\n"
            + self.action_spec.render_for_prompt()
        )
