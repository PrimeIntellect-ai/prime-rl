"""Atari Ice Hockey environment.

Top-down hockey rink. Score goals against AI opponent.

Gym ID: glyphbench/atari-icehockey-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec

from .base import AtariBase


class IceHockeyEnv(AtariBase):
    """Ice Hockey: top-down hockey rink.

    20x24 grid. Score goals past the opponent goalie.
    AI opponent chases puck and shoots.

    Actions: NOOP, UP, DOWN, LEFT, RIGHT, SHOOT
    Pattern C (adversarial first-to-W): ±1/_WIN_TARGET per goal.
    First to _WIN_TARGET wins (full-scope = 21).
    """

    action_spec = ActionSpec(
        names=(
            "NOOP", "UP", "DOWN", "LEFT",
            "RIGHT", "SHOOT",
        ),
        descriptions=(
            "do nothing",
            "move up (skate)",
            "move down (skate)",
            "move left (skate)",
            "move right (skate)",
            "shoot the puck",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 24
    _RINK_L = 1
    _RINK_R = 18
    _RINK_T = 1
    _RINK_B = 22
    _GOAL_L = 7
    _GOAL_R = 12
    _PERIOD_LEN = 500

    # Pattern C full-scope target: first to 21 goals.
    _WIN_TARGET: int = 21

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._opp_x: int = 0
        self._opp_y: int = 0
        self._puck_x: float = 0.0
        self._puck_y: float = 0.0
        self._puck_dx: float = 0.0
        self._puck_dy: float = 0.0
        self._puck_free: bool = True
        self._puck_holder: str = ""  # "player"/"opp"/""
        self._player_goals: int = 0
        self._opp_goals: int = 0
        self._period: int = 1
        self._period_timer: int = 0
        self._agent_progress: int = 0
        self._opp_progress: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-icehockey-v0"

    def _reset(self, seed: int):
        self._agent_progress = 0
        self._opp_progress = 0
        return super()._reset(seed)

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._player_goals = 0
        self._opp_goals = 0
        self._period = 1
        self._period_timer = 0
        self._lives = 1
        self._reset_positions()
        self._redraw()

    def _reset_positions(self) -> None:
        cx = self._WIDTH // 2
        self._player_x, self._player_y = cx, self._RINK_B - 4
        self._opp_x, self._opp_y = cx, self._RINK_T + 4
        self._puck_x, self._puck_y = float(cx), float(self._HEIGHT // 2)
        self._puck_dx = self._puck_dy = 0.0
        self._puck_free, self._puck_holder = True, ""

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._period_timer += 1

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
        if self._in_rink(nx, ny):
            self._player_x, self._player_y = nx, ny

        # Pick up puck if nearby
        px = int(round(self._puck_x))
        py = int(round(self._puck_y))
        if (
            self._puck_free
            and abs(self._player_x - px) <= 1
            and abs(self._player_y - py) <= 1
        ):
            self._puck_holder = "player"
            self._puck_free = False

        # Carry puck
        if self._puck_holder == "player":
            self._puck_x = float(self._player_x)
            self._puck_y = float(self._player_y - 1)

        # Shoot
        if action_name == "SHOOT" and self._puck_holder == "player":
            self._puck_holder = ""
            self._puck_free = True
            self._puck_dy = -2.0
            self._puck_dx = float(self.rng.integers(-1, 2)) * 0.5

        # Move puck if free and moving
        if self._puck_free and (self._puck_dx or self._puck_dy):
            self._puck_x += self._puck_dx
            self._puck_y += self._puck_dy
            px = int(round(self._puck_x))
            py = int(round(self._puck_y))
            # Wall bounce
            if px <= self._RINK_L or px >= self._RINK_R:
                self._puck_dx = -self._puck_dx
            # Check goals
            if py <= self._RINK_T:
                if self._GOAL_L <= px <= self._GOAL_R:
                    self._player_goals += 1
                    self._on_point_scored(1)
                    if self._agent_progress < self._WIN_TARGET:
                        reward = 1.0 / self._WIN_TARGET
                        self._agent_progress += 1
                    self._message = "GOAL! You score!"
                    self._reset_positions()
                else:
                    self._puck_dy = -self._puck_dy
            elif py >= self._RINK_B:
                if self._GOAL_L <= px <= self._GOAL_R:
                    self._opp_goals += 1
                    if self._opp_progress < self._WIN_TARGET:
                        reward = -1.0 / self._WIN_TARGET
                        self._opp_progress += 1
                    self._message = "Opponent scores!"
                    self._reset_positions()
                else:
                    self._puck_dy = -self._puck_dy

            # Friction
            self._puck_dx *= 0.95
            self._puck_dy *= 0.95
            if abs(self._puck_dx) < 0.1 and abs(self._puck_dy) < 0.1:
                self._puck_dx = self._puck_dy = 0.0
        # Opponent AI
        self._move_opponent()
        # First-to-N termination
        terminated = False
        if self._agent_progress >= self._WIN_TARGET:
            terminated = True
            self._message = "You win!"
            info["won"] = True
        elif self._opp_progress >= self._WIN_TARGET:
            terminated = True
            self._message = "Opponent wins!"
        elif self._period_timer >= self._PERIOD_LEN:
            self._period += 1
            self._period_timer = 0
            if self._period > 3:
                terminated = True
                if self._player_goals > self._opp_goals:
                    self._message = "You win!"
                elif self._opp_goals > self._player_goals:
                    self._message = "Opponent wins!"
                else:
                    self._message = "Draw!"
            else:
                self._reset_positions()

        info["player_goals"] = self._player_goals
        info["opp_goals"] = self._opp_goals
        info["period"] = min(self._period, 3)
        self._redraw()
        return reward, terminated, info

    def _in_rink(self, x: int, y: int) -> bool:
        return (
            self._RINK_L < x < self._RINK_R
            and self._RINK_T < y < self._RINK_B
        )

    def _move_opponent(self) -> None:
        rng = self.rng
        px = int(round(self._puck_x))
        py = int(round(self._puck_y))

        # Pick up puck
        if (
            self._puck_free
            and abs(self._opp_x - px) <= 1
            and abs(self._opp_y - py) <= 1
        ):
            self._puck_holder = "opp"
            self._puck_free = False

        if self._puck_holder == "opp":
            self._puck_x = float(self._opp_x)
            self._puck_y = float(self._opp_y + 1)
            # Shoot toward goal
            if (
                self._opp_y > self._RINK_B - 8
                and rng.random() < 0.3
            ):
                self._puck_holder = ""
                self._puck_free = True
                self._puck_dy = 2.0
                self._puck_dx = float(rng.integers(-1, 2)) * 0.5
        # Chase puck or drive to goal
        tx = px if self._puck_holder != "opp" else self._WIDTH // 2
        ty = py if self._puck_holder != "opp" else self._RINK_B
        if rng.random() < 0.6:
            dx = 0 if self._opp_x == tx else (1 if self._opp_x < tx else -1)
            dy = 0 if self._opp_y == ty else (1 if self._opp_y < ty else -1)
            nx, ny = self._opp_x + dx, self._opp_y + dy
            if self._in_rink(nx, ny):
                self._opp_x, self._opp_y = nx, ny

    def _redraw(self) -> None:
        for y in range(self._HEIGHT):
            for x in range(self._WIDTH):
                self._set_cell(x, y, " ")

        # Rink borders
        for x in range(self._RINK_L, self._RINK_R + 1):
            self._set_cell(x, self._RINK_T, "─")
            self._set_cell(x, self._RINK_B, "─")
        for y in range(self._RINK_T, self._RINK_B + 1):
            self._set_cell(self._RINK_L, y, "│")
            self._set_cell(self._RINK_R, y, "│")

        # Goals (gaps in border)
        for x in range(self._GOAL_L, self._GOAL_R + 1):
            self._set_cell(x, self._RINK_T, "G")
            self._set_cell(x, self._RINK_B, "G")

        # Center line
        for x in range(self._RINK_L + 1, self._RINK_R):
            self._set_cell(x, self._HEIGHT // 2, "·")

        # Puck
        px, py = int(round(self._puck_x)), int(round(self._puck_y))
        if self._in_rink(px, py):
            self._set_cell(px, py, "*")
        self._set_cell(self._opp_x, self._opp_y, "V")

    def _advance_entities(self) -> None:
        pass

    def _render_current_observation(self, **kw: Any):  # type: ignore[override]
        """Append icehockey-specific HUD on top of base HUD.

        Base HUD provides ``Step: T / N`` (X4 cross-cutting).
        """
        from glyphbench.core.observation import GridObservation

        obs = super()._render_current_observation()
        p = min(self._period, 3)
        pdx = round(self._puck_dx, 1)
        pdy = round(self._puck_dy, 1)
        holder = self._puck_holder or "loose"
        puck = f"Puck vel=({pdx},{pdy}) holder={holder}"
        extra = (
            f"You {self._player_goals} - "
            f"{self._opp_goals} Opp | Period {p}/3"
            f"\n{puck}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "─": "boards",
            "│": "boards",
            "G": "goal",
            "·": "center line",
            "*": "puck",
            "V": "opponent",
            " ": "ice",
        }.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "Play ice hockey. Skate to pick up the puck, "
            "then SHOOT to score. "
            "Defend your goal from the opponent. "
            "3 periods, most goals wins."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Ice Hockey.\n\n"
            "TASK\n"
            "Play 3 periods of 1-on-1 hockey. Carry the puck to the "
            "opponent goal and SHOOT to score; defend your goal from "
            "their shots.\n\n"
            "BOARD\n"
            "20x24 rink. Boards '-' and '|' enclose the rink. Goal "
            "zones 'G' sit at the two ends of the rink; you and the "
            "opponent start in opposite halves separated by the "
            "center line '.'. Puck '*'. Opponent 'V'. You appear as "
            "an arrow. Goal at row 1 is yours to attack; the row 22 "
            "goal is theirs.\n\n"
            "MECHANICS\n"
            "UP / DOWN / LEFT / RIGHT skate 1 cell inside the rink. "
            "If the puck is loose and within Manhattan distance 1 of "
            "you, you pick it up; it then travels with you 1 cell "
            "above your head. SHOOT with the puck launches it upward "
            "(dy=-2) with dx in {-0.5, 0, 0.5}. Loose puck bounces "
            "off side boards; entering a goal zone scores. Friction "
            "reduces puck velocity by 5 percent per step.\n\n"
            "SCORING\n"
            "+1/21 reward per goal you score (Pattern C). -1/21 "
            "reward per goal conceded. Bound: cumulative reward "
            "stays in [-1, +1].\n\n"
            "TERMINATION\n"
            "First side to 21 goals wins (terminates). Otherwise "
            "three periods of 500 ticks each (1500 total). No lives. "
            "At end the higher score wins (message only).\n\n"
            "HUD\n"
            "Shows your goals, opponent goals, current period, and "
            "puck pos / vel / holder.\n\n"
            + self.action_spec.render_for_prompt()
        )
