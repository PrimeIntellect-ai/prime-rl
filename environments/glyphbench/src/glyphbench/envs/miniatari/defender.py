"""miniatari Defender.

Identity: Spaceship rescues humans from descending landers on a planet surface.
Win condition: 3 humans reach safety (player escorts each off the top edge).
Reward: Pattern A, +1/3 per human rescued.
Loss: a lander reaches a human and abducts them off the bottom (terminal).

Gym ID: glyphbench/miniatari-defender-v0

Random baseline (seed=0..29): success_rate=0%, mean_length=151, mean_return=+0.044
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniDefenderEnv(MiniatariBase):
    """Mini Defender: 14x10 grid, 3 humans on the ground row, 3 landers above.

    Player ship (Y) starts at the top center. Humans (h) sit on the ground
    row 8. Landers (L) descend from row 1 toward the humans, dropping 1
    cell every 3 ticks. To rescue a human you must (1) be on the ground
    row in the human's column to pick them up (h becomes a tracked
    'carried' status on you), then (2) reach row 0 with a carried human,
    counting as one rescue. Picking up a human consumes that human (it
    disappears from the ground). FIRE shoots a bullet up to 4 cells
    horizontally in your last facing direction; the first lander hit is
    destroyed. If a lander reaches a human's row before you've picked
    them up, that human is abducted and the run ends. Rescue all 3
    humans to win.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "UP", "DOWN", "FIRE"),
        descriptions=(
            "do nothing",
            "move left one cell and face left",
            "move right one cell and face right",
            "move up one cell",
            "move down one cell",
            "fire a bullet horizontally in your facing direction",
        ),
    )

    default_max_turns = 300

    _WIDTH = 14
    _HEIGHT = 10
    _WIN_TARGET = 3
    _GROUND_Y = 8
    _BULLET_RANGE = 4
    _LANDER_MOVE_EVERY = 3

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._humans: list[int] = []  # x positions of humans still on ground
        self._landers: list[list[int]] = []  # [x, y]
        self._carrying: bool = False
        self._tick_count: int = 0
        self._progress: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-defender-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._progress = 0
        self._tick_count = 0
        self._carrying = False
        self._player_x = self._WIDTH // 2
        self._player_y = 0
        self._player_dir = (1, 0)
        rng = self.rng
        # 3 humans evenly spaced
        slots = [2, 6, 11]
        rng.shuffle(slots)
        self._humans = sorted(slots[:3])
        # 3 landers descending toward each human's column (slight jitter)
        self._landers = []
        for hx in self._humans:
            jitter = int(rng.integers(-1, 2))
            lx = max(1, min(self._WIDTH - 2, hx + jitter))
            self._landers.append([lx, 1])

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._tick_count += 1

        # 1. Player move
        if action_name == "LEFT" and self._player_x > 0:
            self._player_x -= 1
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT" and self._player_x < self._WIDTH - 1:
            self._player_x += 1
            self._player_dir = (1, 0)
        elif action_name == "UP" and self._player_y > 0:
            self._player_y -= 1
        elif action_name == "DOWN" and self._player_y < self._GROUND_Y:
            self._player_y += 1

        # 2. Fire (horizontal)
        if action_name == "FIRE":
            bdx = self._player_dir[0] if self._player_dir[0] != 0 else 1
            bx = self._player_x
            target: int | None = None
            for _ in range(self._BULLET_RANGE):
                bx += bdx
                if bx < 0 or bx >= self._WIDTH:
                    break
                hit = False
                for i, (lx, ly) in enumerate(self._landers):
                    if lx == bx and ly == self._player_y:
                        target = i
                        hit = True
                        break
                if hit:
                    break
            if target is not None:
                self._landers.pop(target)
                self._message = "Lander destroyed!"

        # 3. Pick up human if standing on ground row in their column
        if not self._carrying and self._player_y == self._GROUND_Y:
            for i, hx in enumerate(self._humans):
                if hx == self._player_x:
                    self._humans.pop(i)
                    self._carrying = True
                    self._message = "Picked up human!"
                    break

        # 4. Deliver to safety: reach top row while carrying
        if self._carrying and self._player_y == 0:
            self._carrying = False
            self._progress += 1
            reward += self._progress_reward(self._WIN_TARGET)
            self._message = f"Rescued! ({self._progress}/{self._WIN_TARGET})"
            if self._progress >= self._WIN_TARGET:
                self._on_won()
                return reward, self._game_over, info

        # 5. Move landers (every K ticks)
        if self._tick_count % self._LANDER_MOVE_EVERY == 0:
            new_landers: list[list[int]] = []
            for lx, ly in self._landers:
                ly += 1
                if ly > self._GROUND_Y:
                    continue  # missed; despawn
                new_landers.append([lx, ly])
            self._landers = new_landers
            # Check abduction: a lander on ground row in a human's column
            for lx, ly in self._landers:
                if ly == self._GROUND_Y and lx in self._humans:
                    self._message = "A human was abducted!"
                    self._game_over = True
                    self._won = False
                    return reward, True, info

        info["progress"] = self._progress
        info["humans_left"] = len(self._humans)
        info["landers_left"] = len(self._landers)
        info["carrying"] = self._carrying
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        # Ground
        for x in range(self._WIDTH):
            grid[self._GROUND_Y + 1][x] = "="
        # Humans
        for hx in self._humans:
            if 0 <= hx < self._WIDTH:
                grid[self._GROUND_Y][hx] = "h"
        # Landers
        for lx, ly in self._landers:
            if 0 <= lx < self._WIDTH and 0 <= ly < self._HEIGHT:
                grid[ly][lx] = "L"
        # Player
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            pch = "C" if self._carrying else "Y"
            grid[self._player_y][self._player_x] = pch

        symbols = {
            " ": "sky",
            "=": "ground",
            "h": "human (rescue them)",
            "L": "lander (abducts humans)",
            "Y": "your ship",
            "C": "your ship (carrying a human)",
        }
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Rescued: {self._progress}/{self._WIN_TARGET}    "
            f"Score: {self._score:.3f}    "
            f"Carrying: {self._carrying}    "
            f"Humans: {len(self._humans)}    "
            f"Landers: {len(self._landers)}"
        )
        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Defender on a 14x10 grid. 3 humans (h) stand on the ground "
            "row (y=8). 3 landers (L) descend from row 1 every 3 ticks toward "
            "the humans. To rescue a human: descend to row 8 in their column "
            "(picks them up automatically) and then return to row 0 to deliver. "
            "FIRE shoots horizontally up to 4 cells in your facing direction, "
            "destroying the first lander in the same row. If a lander reaches "
            "a human's column on the ground row, that human is abducted and "
            "the run ends. Rescue all 3 humans to win. Reward: +1/3 per rescue."
        )
