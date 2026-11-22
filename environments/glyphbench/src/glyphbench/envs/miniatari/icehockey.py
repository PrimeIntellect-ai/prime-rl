"""miniatari Ice Hockey.

Identity: Top-down 1v1 hockey, score goals past the opponent.
Win condition: agent scores 3 goals first.
Reward: Pattern C, +1/3 per agent goal, -1/3 per opp goal.
Loss: opponent scores 3 goals first.

Gym ID: glyphbench/miniatari-icehockey-v0

Random baseline (seed=0..29): success_rate=3%, mean_length=80, mean_return=-0.200
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniIceHockeyEnv(MiniatariBase):
    """Mini Ice Hockey: 16x10 rink, first-to-3 goals.

    Agent starts at the bottom; opponent at the top. Both can pick up a
    free puck by being adjacent to it. Holding the puck, the agent skates
    upward and SHOOTs to launch the puck toward the opponent's goal at
    row 0. Agent's goal sits at the bottom (row 9). Goals are 4 cells wide.
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP", "DOWN", "LEFT", "RIGHT", "SHOOT"),
        descriptions=(
            "do nothing",
            "skate up",
            "skate down",
            "skate left",
            "skate right",
            "shoot the puck (only if you hold it)",
        ),
    )

    default_max_turns = 300

    _WIDTH = 16
    _HEIGHT = 10
    _RINK_L = 1
    _RINK_R = 14
    _RINK_T = 1
    _RINK_B = 8
    _GOAL_L = 6
    _GOAL_R = 9
    _WIN_TARGET = 3

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._opp_x: int = 0
        self._opp_y: int = 0
        self._puck_x: int = 0
        self._puck_y: int = 0
        self._puck_dx: int = 0
        self._puck_dy: int = 0
        self._puck_holder: str = "free"  # "free", "agent", "opp"
        self._agent_score: int = 0
        self._opp_score: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-icehockey-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._agent_score = 0
        self._opp_score = 0
        self._reset_positions()

    def _reset_positions(self) -> None:
        cx = self._WIDTH // 2
        self._player_x = cx
        self._player_y = self._RINK_B - 1
        self._opp_x = cx
        self._opp_y = self._RINK_T + 1
        self._puck_x = cx
        self._puck_y = self._HEIGHT // 2
        self._puck_dx = 0
        self._puck_dy = 0
        self._puck_holder = "free"

    def _in_rink(self, x: int, y: int) -> bool:
        return self._RINK_L < x < self._RINK_R and self._RINK_T < y < self._RINK_B

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
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
        if self._in_rink(nx, ny) and not (nx == self._opp_x and ny == self._opp_y):
            self._player_x, self._player_y = nx, ny

        # Pick up free puck if adjacent
        if (
            self._puck_holder == "free"
            and abs(self._player_x - self._puck_x) <= 1
            and abs(self._player_y - self._puck_y) <= 1
        ):
            self._puck_holder = "agent"

        # If agent holds puck, attach to head
        if self._puck_holder == "agent":
            self._puck_x = self._player_x
            self._puck_y = max(self._RINK_T + 1, self._player_y - 1)
            self._puck_dx = 0
            self._puck_dy = 0

        # Shoot. The puck is launched with dy=-2 toward the opponent's goal.
        # We stash a "just-shot" flag so this tick skips opponent and agent
        # pickup checks: the puck physically leaves the agent's vicinity on
        # the NEXT tick, otherwise the agent or the opponent (spawning at
        # row 2) would immediately re-grab the shot.
        just_shot = False
        if action_name == "SHOOT" and self._puck_holder == "agent":
            self._puck_holder = "free"
            self._puck_dx = 0
            self._puck_dy = -2
            # Place the puck two rows ahead of player to give it room.
            self._puck_x = self._player_x
            self._puck_y = max(self._RINK_T + 1, self._player_y - 3)
            self._message = "Shot!"
            just_shot = True

        # Move free puck (skipped on the tick of the shot, since the SHOOT
        # block above already placed the puck ahead of the agent).
        scored: str | None = None
        if not just_shot and self._puck_holder == "free" and (self._puck_dx or self._puck_dy):
            new_x = self._puck_x + self._puck_dx
            new_y = self._puck_y + self._puck_dy

            # Side bounce
            if new_x <= self._RINK_L:
                new_x = self._RINK_L + 1
                self._puck_dx = abs(self._puck_dx)
            elif new_x >= self._RINK_R:
                new_x = self._RINK_R - 1
                self._puck_dx = -abs(self._puck_dx)

            # Goal at top (agent scores)
            if new_y <= self._RINK_T:
                if self._GOAL_L <= new_x <= self._GOAL_R:
                    scored = "agent"
                else:
                    new_y = self._RINK_T + 1
                    self._puck_dy = abs(self._puck_dy)
            # Goal at bottom (opp scores)
            elif new_y >= self._RINK_B:
                if self._GOAL_L <= new_x <= self._GOAL_R:
                    scored = "opp"
                else:
                    new_y = self._RINK_B - 1
                    self._puck_dy = -abs(self._puck_dy)

            self._puck_x = new_x
            self._puck_y = new_y

        # Opponent AI
        if self._puck_holder == "opp":
            # Opp carries puck downward and shoots when low
            self._puck_x = self._opp_x
            self._puck_y = min(self._RINK_B - 1, self._opp_y + 1)

        if self._puck_holder == "free":
            # Chase puck
            if self.rng.random() < 0.6:
                tx, ty = self._puck_x, self._puck_y
                dx = 0 if self._opp_x == tx else (1 if self._opp_x < tx else -1)
                dy = 0 if self._opp_y == ty else (1 if self._opp_y < ty else -1)
                nx_o, ny_o = self._opp_x + dx, self._opp_y + dy
                if self._in_rink(nx_o, ny_o) and not (nx_o == self._player_x and ny_o == self._player_y):
                    self._opp_x, self._opp_y = nx_o, ny_o
            # pick up puck if adjacent (skipped on the shot tick so the puck
            # actually leaves the agent before the opponent can steal it)
            if not just_shot and abs(self._opp_x - self._puck_x) <= 1 and abs(self._opp_y - self._puck_y) <= 1:
                self._puck_holder = "opp"
        elif self._puck_holder == "opp":
            # Move toward bottom and shoot when in range
            if self.rng.random() < 0.5 and self._opp_y < self._RINK_B - 2:
                nx_o, ny_o = self._opp_x, self._opp_y + 1
                if self._in_rink(nx_o, ny_o) and not (nx_o == self._player_x and ny_o == self._player_y):
                    self._opp_x, self._opp_y = nx_o, ny_o
            if self._opp_y >= self._RINK_B - 3 and self.rng.random() < 0.4:
                # Shoot
                self._puck_holder = "free"
                self._puck_dx = max(-1, min(1, (self._WIDTH // 2) - self._opp_x))
                self._puck_dy = 2
                self._message = "Opponent shoots!"

        # Apply scoring
        if scored == "agent":
            reward += self._agent_score_reward(self._WIN_TARGET)
            self._agent_score += 1
            self._message = "GOAL! +1/3"
            if self._agent_score >= self._WIN_TARGET:
                self._on_won()
            else:
                self._reset_positions()
        elif scored == "opp":
            reward += self._opp_score_reward(self._WIN_TARGET)
            self._opp_score += 1
            self._message = "Opponent scores!"
            if self._opp_score >= self._WIN_TARGET:
                self._on_life_lost()
            else:
                self._reset_positions()

        info["agent_score"] = self._agent_score
        info["opp_score"] = self._opp_score
        info["puck_holder"] = self._puck_holder
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = []
        for y in range(self._HEIGHT):
            row: list[str] = []
            for x in range(self._WIDTH):
                if y == self._RINK_T - 1 or y == self._RINK_B + 1:
                    if self._GOAL_L <= x <= self._GOAL_R:
                        row.append("G")
                    else:
                        row.append("─")
                elif x == self._RINK_L - 1 or x == self._RINK_R + 1:
                    row.append("│")
                else:
                    row.append(" ")
            grid.append(row)

        # Center line
        for x in range(self._RINK_L + 1, self._RINK_R):
            grid[self._HEIGHT // 2][x] = "·"

        # Puck
        if 0 <= self._puck_x < self._WIDTH and 0 <= self._puck_y < self._HEIGHT:
            grid[self._puck_y][self._puck_x] = "*"

        # Opponent
        if 0 <= self._opp_x < self._WIDTH and 0 <= self._opp_y < self._HEIGHT:
            grid[self._opp_y][self._opp_x] = "P"

        # Agent
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            pch = self._DIR_CHARS.get(self._player_dir, "@")
            grid[self._player_y][self._player_x] = pch

        symbols = {
            "─": "boards", "│": "boards", "G": "goal mouth",
            "·": "center line", "*": "puck", "P": "opponent",
            " ": "ice",
        }
        pch = self._DIR_CHARS.get(self._player_dir, "@")
        symbols[pch] = f"you (facing {self._DIR_NAMES.get(self._player_dir, 'none')})"

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Goals: You {self._agent_score} - {self._opp_score} Opp    "
            f"First to {self._WIN_TARGET}\n"
            f"Puck vel=({self._puck_dx:+d},{self._puck_dy:+d}) "
            f"holder={self._puck_holder}"
        )
        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "1-on-1 ice hockey on a 16x10 rink. Score by getting the puck "
            "(*) into the goal mouth (G) at the top of the rink, defended "
            "by the opponent (P). Skate adjacent to a free puck to pick it "
            "up; SHOOT to launch it upward toward the goal. Defend your own "
            "goal at the bottom. First to 3 goals wins. Reward: +1/3 per "
            "goal scored, -1/3 per goal conceded."
        )
