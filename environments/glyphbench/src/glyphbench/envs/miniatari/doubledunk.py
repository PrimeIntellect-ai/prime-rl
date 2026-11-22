"""miniatari Double Dunk.

Identity: Mini-court basketball; race to score 4 baskets.
Win condition: agent makes 4 baskets first.
Reward: Pattern C, +1/4 per agent basket, -1/4 per opp basket.
Loss: opponent makes 4 baskets first.

Gym ID: glyphbench/miniatari-doubledunk-v0

Random baseline (seed=0..29): success_rate=0%, mean_length=86, mean_return=-0.675
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniDoubleDunkEnv(MiniatariBase):
    """Mini Double Dunk: 14x10 court, first-to-4 baskets.

    Agent at the bottom must move toward the top hoop and SHOOT. The
    farther the shot, the lower the make-probability. Opponent attacks
    when in possession. Steals possible by adjacency.
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP", "DOWN", "LEFT", "RIGHT", "SHOOT"),
        descriptions=(
            "do nothing",
            "move up",
            "move down",
            "move left",
            "move right",
            "shoot the ball (only with possession)",
        ),
    )

    default_max_turns = 250

    _WIDTH = 14
    _HEIGHT = 10
    _COURT_L = 1
    _COURT_R = 12
    _COURT_T = 1
    _COURT_B = 8
    _AGENT_HOOP_Y = 1   # top hoop, the one agent attacks
    _OPP_HOOP_Y = 8     # bottom hoop, the one opponent attacks
    _WIN_TARGET = 4

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._opp_x: int = 0
        self._opp_y: int = 0
        self._possession: str = "agent"  # "agent" or "opp"
        self._agent_score: int = 0
        self._opp_score: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-doubledunk-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._agent_score = 0
        self._opp_score = 0
        self._reset_positions(possession="agent")

    def _reset_positions(self, possession: str) -> None:
        cx = self._WIDTH // 2
        self._player_x = cx
        self._player_y = self._COURT_B - 1
        self._opp_x = cx
        self._opp_y = self._COURT_T + 1
        self._possession = possession

    def _on_court(self, x: int, y: int) -> bool:
        return self._COURT_L < x < self._COURT_R and self._COURT_T < y < self._COURT_B

    def _shot_probability(self, shooter_y: int, hoop_y: int) -> float:
        """Make probability decreases with distance from hoop."""
        dist = abs(shooter_y - hoop_y)
        if dist <= 2:
            return 0.7
        elif dist <= 4:
            return 0.45
        else:
            return 0.2

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        agent_scored_this_tick = False

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
        if self._on_court(nx, ny) and not (nx == self._opp_x and ny == self._opp_y):
            self._player_x, self._player_y = nx, ny

        # Player shoots (must hold ball)
        if action_name == "SHOOT" and self._possession == "agent":
            p = self._shot_probability(self._player_y, self._AGENT_HOOP_Y)
            if self.rng.random() < p:
                reward += self._agent_score_reward(self._WIN_TARGET)
                self._agent_score += 1
                agent_scored_this_tick = True
                self._message = "Basket! +1/4"
                if self._agent_score >= self._WIN_TARGET:
                    self._on_won()
                else:
                    self._reset_positions(possession="opp")
            else:
                self._message = "Missed shot."
                self._reset_positions(possession="opp")

        # Opponent steal attempt when adjacent and agent has ball
        if (
            not self._game_over
            and self._possession == "agent"
            and abs(self._player_x - self._opp_x) <= 1
            and abs(self._player_y - self._opp_y) <= 1
            and self.rng.random() < 0.20
        ):
            self._possession = "opp"
            self._message = "Stolen!"

        # Opponent AI: when in possession, drive toward bottom hoop and shoot
        # Skip opp's shot block if agent just scored this tick (one-tick gap on
        # possession reset prevents same-tick agent-make + opp-make).
        if not self._game_over and not agent_scored_this_tick:
            if self._possession == "opp":
                # Move toward bottom hoop
                if self.rng.random() < 0.6:
                    target_x = self._WIDTH // 2
                    if self._opp_x < target_x:
                        nx_o = self._opp_x + 1
                    elif self._opp_x > target_x:
                        nx_o = self._opp_x - 1
                    else:
                        nx_o = self._opp_x
                    ny_o = min(self._OPP_HOOP_Y - 1, self._opp_y + 1)
                    if self._on_court(nx_o, ny_o) and not (nx_o == self._player_x and ny_o == self._player_y):
                        self._opp_x, self._opp_y = nx_o, ny_o
                # Try to shoot when close to opp hoop
                if abs(self._opp_y - self._OPP_HOOP_Y) <= 3 and self.rng.random() < 0.3:
                    p = self._shot_probability(self._opp_y, self._OPP_HOOP_Y)
                    if self.rng.random() < p:
                        reward += self._opp_score_reward(self._WIN_TARGET)
                        self._opp_score += 1
                        self._message = "Opponent scores!"
                        if self._opp_score >= self._WIN_TARGET:
                            self._on_life_lost()
                        else:
                            self._reset_positions(possession="agent")
                    else:
                        self._message = "Opponent missed."
                        self._reset_positions(possession="agent")
            else:
                # Defender: move toward player
                if self.rng.random() < 0.5:
                    if self._opp_x < self._player_x:
                        nx_o = self._opp_x + 1
                    elif self._opp_x > self._player_x:
                        nx_o = self._opp_x - 1
                    else:
                        nx_o = self._opp_x
                    if self._opp_y < self._player_y:
                        ny_o = self._opp_y + 1
                    elif self._opp_y > self._player_y:
                        ny_o = self._opp_y - 1
                    else:
                        ny_o = self._opp_y
                    if self._on_court(nx_o, ny_o) and not (nx_o == self._player_x and ny_o == self._player_y):
                        self._opp_x, self._opp_y = nx_o, ny_o

        info["agent_score"] = self._agent_score
        info["opp_score"] = self._opp_score
        info["possession"] = self._possession
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
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

        # Hoops at top and bottom
        hoop_x = self._WIDTH // 2
        grid[self._AGENT_HOOP_Y][hoop_x] = "H"
        grid[self._OPP_HOOP_Y][hoop_x] = "H"

        # Half-court line
        for x in range(self._COURT_L + 1, self._COURT_R):
            if grid[self._HEIGHT // 2][x] == " ":
                grid[self._HEIGHT // 2][x] = "·"

        # Opponent
        if 0 <= self._opp_x < self._WIDTH and 0 <= self._opp_y < self._HEIGHT:
            grid[self._opp_y][self._opp_x] = "P"

        # Agent
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            pch = self._DIR_CHARS.get(self._player_dir, "@")
            grid[self._player_y][self._player_x] = pch

        symbols = {
            "─": "court line", "│": "court line", "H": "hoop",
            "·": "half-court", "P": "opponent", " ": "court",
        }
        pch = self._DIR_CHARS.get(self._player_dir, "@")
        symbols[pch] = f"you (facing {self._DIR_NAMES.get(self._player_dir, 'none')})"

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Baskets: You {self._agent_score} - {self._opp_score} Opp    "
            f"First to {self._WIN_TARGET}    "
            f"Possession: {self._possession}"
        )
        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini basketball on a 14x10 court. You attack the top hoop (H, "
            "row 1); the opponent (P) attacks the bottom hoop. Move with "
            "UP/DOWN/LEFT/RIGHT. SHOOT (with possession) to attempt a "
            "basket - closer shots are higher percentage. After a make or "
            "a miss, possession changes. The opponent has a 20% chance to "
            "steal when adjacent. First to 4 baskets wins. Reward: +1/4 "
            "per basket scored, -1/4 per basket conceded."
        )
