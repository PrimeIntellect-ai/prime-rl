"""miniatari Boxing.

Identity: Top-down 1v1 boxing in a small ring; first to 5 punches wins.
Win condition: agent lands 5 punches.
Reward: Pattern C, +1/5 per punch landed, -1/5 per punch absorbed.
Loss: opponent lands 5 punches first.

Gym ID: glyphbench/miniatari-boxing-v0

Random baseline (seed=0..29): success_rate=0%, mean_length=33, mean_return=-0.427
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniBoxingEnv(MiniatariBase):
    """Mini Boxing: 12x8 ring, first-to-5 punches.

    Both fighters move in a square ring. PUNCH lands when the opponent
    is within Manhattan distance 2. Punches have a cooldown so spamming
    is not rewarded.
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP", "DOWN", "LEFT", "RIGHT", "PUNCH"),
        descriptions=(
            "do nothing",
            "move up",
            "move down",
            "move left",
            "move right",
            "throw a punch (lands if opponent within 2 cells)",
        ),
    )

    default_max_turns = 200

    _WIDTH = 12
    _HEIGHT = 8
    _RING_L = 1
    _RING_R = 10
    _RING_T = 1
    _RING_B = 6
    _WIN_TARGET = 5
    _PUNCH_COOLDOWN = 2

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._opp_x: int = 0
        self._opp_y: int = 0
        self._agent_score: int = 0
        self._opp_score: int = 0
        self._punch_cd: int = 0
        self._opp_cd: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-boxing-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._agent_score = 0
        self._opp_score = 0
        self._punch_cd = 0
        self._opp_cd = 0
        # Player at bottom-left, opponent at top-right
        self._player_x = self._RING_L + 2
        self._player_y = self._RING_B - 1
        self._opp_x = self._RING_R - 2
        self._opp_y = self._RING_T + 1

    def _in_ring(self, x: int, y: int) -> bool:
        return self._RING_L < x < self._RING_R and self._RING_T < y < self._RING_B

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        if self._punch_cd > 0:
            self._punch_cd -= 1
        if self._opp_cd > 0:
            self._opp_cd -= 1

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
        if self._in_ring(nx, ny) and not (nx == self._opp_x and ny == self._opp_y):
            self._player_x, self._player_y = nx, ny

        # Player punch
        if action_name == "PUNCH" and self._punch_cd == 0:
            self._punch_cd = self._PUNCH_COOLDOWN
            dist = abs(self._player_x - self._opp_x) + abs(self._player_y - self._opp_y)
            if dist <= 2:
                reward += self._agent_score_reward(self._WIN_TARGET)
                self._agent_score += 1
                self._message = "Punch landed! +1/5"
                if self._agent_score >= self._WIN_TARGET:
                    self._on_won()

        # Opponent move and punch
        if not self._game_over:
            self._move_opponent()

            if self._opp_cd == 0:
                dist = abs(self._player_x - self._opp_x) + abs(self._player_y - self._opp_y)
                if dist <= 2 and self.rng.random() < 0.3:
                    self._opp_cd = self._PUNCH_COOLDOWN
                    reward += self._opp_score_reward(self._WIN_TARGET)
                    self._opp_score += 1
                    self._message = "Opponent punched you!"
                    if self._opp_score >= self._WIN_TARGET:
                        self._on_life_lost()

        info["agent_score"] = self._agent_score
        info["opp_score"] = self._opp_score
        return reward, self._game_over, info

    def _move_opponent(self) -> None:
        rng = self.rng
        # 60% chase, 40% randomly move
        if rng.random() < 0.6:
            dx = 0
            dy = 0
            if self._opp_x < self._player_x:
                dx = 1
            elif self._opp_x > self._player_x:
                dx = -1
            if self._opp_y < self._player_y:
                dy = 1
            elif self._opp_y > self._player_y:
                dy = -1
            # Pick one axis to move on
            if rng.random() < 0.5:
                ddx, ddy = dx, 0
            else:
                ddx, ddy = 0, dy
        else:
            ddx = int(rng.integers(-1, 2))
            ddy = int(rng.integers(-1, 2))

        nx, ny = self._opp_x + ddx, self._opp_y + ddy
        if self._in_ring(nx, ny) and not (nx == self._player_x and ny == self._player_y):
            self._opp_x, self._opp_y = nx, ny

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = []
        for y in range(self._HEIGHT):
            row: list[str] = []
            for x in range(self._WIDTH):
                if y == self._RING_T or y == self._RING_B:
                    if x == self._RING_L or x == self._RING_R:
                        row.append("┼")
                    else:
                        row.append("=")
                elif x == self._RING_L or x == self._RING_R:
                    row.append("│")
                else:
                    row.append(" ")
            grid.append(row)

        # Opponent
        if 0 <= self._opp_x < self._WIDTH and 0 <= self._opp_y < self._HEIGHT:
            grid[self._opp_y][self._opp_x] = "B"

        # Agent
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            pch = self._DIR_CHARS.get(self._player_dir, "@")
            grid[self._player_y][self._player_x] = pch

        symbols = {
            "=": "ring rope", "│": "ring rope", "┼": "corner post",
            "B": "opponent boxer", " ": "ring floor",
        }
        pch = self._DIR_CHARS.get(self._player_dir, "@")
        symbols[pch] = f"you (facing {self._DIR_NAMES.get(self._player_dir, 'none')})"

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Hits: You {self._agent_score} - {self._opp_score} Opp    "
            f"First to {self._WIN_TARGET}\n"
            f"Punch CD: {self._punch_cd}    Opp CD: {self._opp_cd}    "
            f"Manhattan dist: "
            f"{abs(self._player_x - self._opp_x) + abs(self._player_y - self._opp_y)}"
        )
        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Boxing in a 12x8 ring vs an AI opponent (B). Move with UP/DOWN/"
            "LEFT/RIGHT. PUNCH lands a hit if the opponent is within "
            "Manhattan distance 2; punching has a 2-tick cooldown. The "
            "opponent chases you and punches when adjacent (30% per tick "
            "while in range). First to 5 hits wins. Reward: +1/5 per punch "
            "landed, -1/5 per punch absorbed."
        )
