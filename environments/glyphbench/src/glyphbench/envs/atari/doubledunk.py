"""Atari Double Dunk (basketball) environment.

Basketball game with 2-point and 3-point shots.

Gym ID: glyphbench/atari-doubledunk-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec

from .base import AtariBase


class DoubleDunkEnv(AtariBase):
    """Double Dunk: basketball game.

    20x16 court. Score baskets (2 or 3 points).
    AI opponent defends and attacks.

    Actions: NOOP, UP, DOWN, LEFT, RIGHT, SHOOT
    Pattern C (adversarial first-to-W): +1/_WIN_TARGET per agent point,
    -1/_WIN_TARGET per opponent point. First side to _WIN_TARGET points
    wins. Cumulative is bounded in [-1, +1].
    """

    action_spec = ActionSpec(
        names=(
            "NOOP", "UP", "DOWN", "LEFT",
            "RIGHT", "SHOOT",
        ),
        descriptions=(
            "do nothing",
            "move up",
            "move down",
            "move left",
            "move right",
            "shoot the ball",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 16
    _COURT_L = 1
    _COURT_R = 18
    _COURT_T = 1
    _COURT_B = 14
    _HOOP_Y = 2
    _OPP_HOOP_Y = 13
    _THREE_LINE = 5

    # Pattern C target: first side to 24 points wins.
    _WIN_TARGET: int = 24

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._opp_x: int = 0
        self._opp_y: int = 0
        self._opp_score: int = 0
        self._has_ball: bool = True
        self._ball_x: float = 0.0
        self._ball_y: float = 0.0
        self._ball_flying: bool = False
        self._ball_dx: float = 0.0
        self._ball_dy: float = 0.0
        self._shot_clock: int = 0
        self._quarter: int = 1
        self._quarter_timer: int = 0
        self._quarter_len: int = 300
        self._agent_progress: int = 0
        self._opp_progress: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-doubledunk-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._opp_score = 0
        self._quarter = 1
        self._quarter_timer = 0
        self._lives = 1
        self._agent_progress = 0
        self._opp_progress = 0
        self._reset_possession(player_has=True)
        self._redraw()

    def _reset_possession(self, player_has: bool = True) -> None:
        cx = self._WIDTH // 2
        self._player_x, self._player_y = cx, self._COURT_B - 3
        self._opp_x, self._opp_y = cx, self._COURT_T + 3
        self._has_ball, self._ball_flying = player_has, False
        self._shot_clock = 0

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._quarter_timer += 1

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
        if self._on_court(nx, ny):
            self._player_x, self._player_y = nx, ny

        # Ball follows player if held
        if self._has_ball and not self._ball_flying:
            self._ball_x = float(self._player_x)
            self._ball_y = float(self._player_y - 1)

        # Shoot
        hx = self._WIDTH // 2
        if action_name == "SHOOT" and self._has_ball and not self._ball_flying:
            self._ball_flying, self._has_ball = True, False
            self._ball_dx = (hx - self._player_x) * 0.3
            self._ball_dy = -1.5
        # Move flying ball
        if self._ball_flying:
            self._ball_x += self._ball_dx
            self._ball_y += self._ball_dy
            bx, by = int(round(self._ball_x)), int(round(self._ball_y))
            if by <= self._HOOP_Y + 1:
                if abs(bx - hx) <= 2:
                    dist = abs(self._player_y - self._HOOP_Y)
                    pts = 3 if dist >= self._THREE_LINE else 2
                    if self.rng.random() < 0.6:
                        self._message = (
                            "Three pointer!" if pts == 3 else "Basket!"
                        )
                        self._on_point_scored(pts)
                        # Pattern C: cap progress at _WIN_TARGET; only
                        # the units below the cap accrue reward.
                        granted = min(
                            pts,
                            self._WIN_TARGET - self._agent_progress,
                        )
                        if granted > 0:
                            reward += granted / self._WIN_TARGET
                            self._agent_progress += granted
                    else:
                        self._message = "Missed shot!"
                    self._ball_flying = False
                    self._reset_possession(player_has=False)
                elif by < self._COURT_T:
                    self._ball_flying = False
                    self._message = "Air ball!"
                    self._reset_possession(player_has=False)

        # Opponent AI
        self._move_opponent()

        # Steal check
        terminated = False
        if self._has_ball:
            dist = abs(self._player_x - self._opp_x) + abs(self._player_y - self._opp_y)
            if dist <= 1 and self.rng.random() < 0.15:
                self._has_ball = False
                self._message = "Stolen!"
                opp_delta, opp_won = self._opp_attack()
                reward += opp_delta
                if opp_won:
                    terminated = True
                    self._message = "Opponent wins!"

        # First-to-_WIN_TARGET termination
        if not terminated and self._agent_progress >= self._WIN_TARGET:
            terminated = True
            info["won"] = True
            self._message = "You win!"

        # Quarter transitions
        if not terminated and self._quarter_timer >= self._quarter_len:
            self._quarter += 1
            self._quarter_timer = 0
            if self._quarter > 4:
                terminated = True
                if self._score > self._opp_score:
                    self._message = "You win!"
                elif self._opp_score > self._score:
                    self._message = "Opponent wins!"
                else:
                    self._message = "Tie game!"
            else:
                self._reset_possession(
                    player_has=self._quarter % 2 == 1
                )
                self._message = (
                    f"Quarter {self._quarter}"
                )

        info["opp_score"] = self._opp_score
        info["quarter"] = min(self._quarter, 4)
        self._redraw()
        return reward, terminated, info

    def _on_court(self, x: int, y: int) -> bool:
        return (
            self._COURT_L < x < self._COURT_R
            and self._COURT_T < y < self._COURT_B
        )

    def _move_opponent(self) -> None:
        rng = self.rng
        def _dir(a: int, b: int) -> int:
            return 0 if a == b else (1 if a < b else -1)
        if self._has_ball or self._ball_flying:
            dx = _dir(self._opp_x, self._player_x)
            dy = _dir(self._opp_y, self._player_y)
        else:
            dx = _dir(self._opp_x, self._WIDTH // 2)
            dy = 1
        if rng.random() < 0.5:
            nx, ny = self._opp_x + dx, self._opp_y + dy
            if self._on_court(nx, ny):
                self._opp_x, self._opp_y = nx, ny

    def _opp_attack(self) -> tuple[float, bool]:
        """Return (delta_reward, opp_won) from this attack."""
        rng = self.rng
        delta = 0.0
        opp_won = False
        dist = abs(self._opp_y - self._OPP_HOOP_Y)
        if dist < 6 and rng.random() < 0.2:
            pts = 3 if dist >= self._THREE_LINE else 2
            if rng.random() < 0.4:
                self._opp_score += pts
                granted = min(
                    pts, self._WIN_TARGET - self._opp_progress
                )
                if granted > 0:
                    delta -= granted / self._WIN_TARGET
                    self._opp_progress += granted
                self._message = f"Opponent scores {pts}!"
                if self._opp_progress >= self._WIN_TARGET:
                    opp_won = True
            else:
                self._message = "Opponent missed!"
            self._reset_possession(player_has=True)
        return delta, opp_won

    def _redraw(self) -> None:
        for y in range(self._HEIGHT):
            for x in range(self._WIDTH):
                self._set_cell(x, y, " ")

        # Court borders
        for x in range(self._COURT_L, self._COURT_R + 1):
            self._set_cell(x, self._COURT_T, "─")
            self._set_cell(x, self._COURT_B, "─")
        for y in range(self._COURT_T, self._COURT_B + 1):
            self._set_cell(self._COURT_L, y, "│")
            self._set_cell(self._COURT_R, y, "│")

        # Hoops and half court
        hx, mid = self._WIDTH // 2, self._HEIGHT // 2
        self._set_cell(hx, self._HOOP_Y, "H")
        self._set_cell(hx, self._OPP_HOOP_Y, "H")
        for x in range(self._COURT_L + 1, self._COURT_R):
            self._set_cell(x, mid, "·")
        # Three-point arcs
        for x in range(hx - 4, hx + 5):
            if self._COURT_L < x < self._COURT_R:
                self._set_cell(x, self._HOOP_Y + self._THREE_LINE, "~")
                self._set_cell(x, self._OPP_HOOP_Y - self._THREE_LINE, "~")
        # Ball and opponent
        if self._ball_flying:
            bx, by = int(round(self._ball_x)), int(round(self._ball_y))
            if self._on_court(bx, by):
                self._set_cell(bx, by, "o")
        self._set_cell(self._opp_x, self._opp_y, "V")

    def _advance_entities(self) -> None:
        pass

    def _render_current_observation(self, **kw: Any):  # type: ignore[override]
        """Append doubledunk-specific HUD info on top of base HUD.

        Base HUD provides ``Step: T / N`` (X4 cross-cutting).
        """
        from glyphbench.core.observation import GridObservation

        obs = super()._render_current_observation()
        q = min(self._quarter, 4)
        if self._has_ball:
            ball_state = "held by you"
        elif self._ball_flying:
            bx = int(round(self._ball_x))
            by = int(round(self._ball_y))
            bdx = round(self._ball_dx, 1)
            bdy = round(self._ball_dy, 1)
            ball_state = (
                f"in flight pos=({bx},{by})"
                f" vel=({bdx},{bdy})"
            )
        else:
            ball_state = "held by opponent"
        extra = (
            f"You {self._score} - {self._opp_score} Opp"
            f" | Q{q}\nBall: {ball_state}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "─": "court line",
            "│": "court line",
            "H": "hoop",
            "·": "half court",
            "~": "three-point line",
            "o": "ball",
            "V": "opponent",
            " ": "court",
        }.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "Play basketball. Move and SHOOT to score. "
            "Shots beyond the ~ line are worth 3 points, "
            "closer shots worth 2. "
            "4 quarters. Most points wins."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Double Dunk.\n\n"
            "TASK\n"
            "Play a 4-quarter game of basketball versus an AI opponent. "
            "Outscore them by making baskets; each basket is 2 or 3 "
            "points depending on shot distance.\n\n"
            "BOARD\n"
            "20x16 court, bounded by '-' and '|'. Half-court row is "
            "row 8 marked by '.'. Hoops 'H' sit at opposite ends of "
            "the court; three-point arcs '~' encircle each hoop. Ball "
            "is 'o' in flight; opponent player 'V'. You are an arrow "
            "glyph.\n\n"
            "MECHANICS\n"
            "UP/DOWN/LEFT/RIGHT move you 1 cell within the court. "
            "SHOOT launches the ball (only if you hold it); velocity "
            "aims the ball toward the top hoop. In flight, ball "
            "position updates each step; when it reaches within 2 "
            "columns of the hoop at the top row it scores with 60 "
            "percent probability. Opponent chases you when you hold "
            "the ball and has a 15 percent chance to steal when "
            "adjacent. Each quarter lasts 300 ticks.\n\n"
            "SCORING\n"
            "+2 reward for a made basket from inside the three-point "
            "line; +3 reward for a shot from row >= 5 rows from hoop. "
            "Reward 0 for misses. Opponent baskets do NOT give "
            "negative reward directly; they just raise opponent "
            "score. A game ends with -1 reward only if opponent has "
            "a higher score at termination? No: the env yields no "
            "terminal bonus; total return = sum of your baskets.\n\n"
            "TERMINATION\n"
            "Episode ends after quarter 4 expires (1200 ticks). No "
            "lives system. Final message shows win / loss / tie.\n\n"
            "HUD\n"
            "Shows your score, opponent score, quarter, and ball "
            "state (held / in flight / opp).\n\n"
            + self.action_spec.render_for_prompt()
        )
