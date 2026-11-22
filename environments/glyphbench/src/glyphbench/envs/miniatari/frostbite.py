"""miniatari Frostbite.

Identity: Hop across drifting ice floes to build an igloo.
Win condition: step on 4 distinct ice floes.
Reward: Pattern A, +1/4 per fresh floe touched.
Loss: fall into the water (no -1 per Pattern A; just terminates the run).

Gym ID: glyphbench/miniatari-frostbite-v0

Random baseline (seed=0..29): success_rate=0%, mean_length=6, mean_return=+0.000
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniFrostbiteEnv(MiniatariBase):
    """Mini Frostbite: 16x10 grid, 4 ice floes drifting horizontally.

    Player starts on solid shore at the top (y=0). Below are 4 rows of
    drifting floes (y=2,4,6,8). Each floe is a single cell; floes in
    different rows drift in alternating directions and wrap around.
    Stepping onto a fresh floe (one not yet visited) awards +1/4. After
    visiting 4 distinct floes the igloo is built and the agent wins.

    Falling into water (any non-floe cell at floe rows) ends the run.
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP", "DOWN", "LEFT", "RIGHT"),
        descriptions=(
            "do nothing",
            "hop one cell up",
            "hop one cell down",
            "hop one cell left",
            "hop one cell right",
        ),
    )

    default_max_turns = 300

    _WIDTH = 16
    _HEIGHT = 10
    _FLOE_ROWS = (2, 4, 6, 8)
    _FLOE_LEN = 3  # each floe is 3 cells wide
    _WIN_TARGET = 4
    _SURVIVAL_TURNS = 24

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._floe_x: list[int] = []  # leftmost x of each floe
        self._floe_dx: list[int] = []
        self._visited_floes: set[int] = set()
        self._progress: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-frostbite-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._progress = 0
        self._visited_floes = set()
        self._player_x = self._WIDTH // 2
        self._player_y = 0
        # Each floe row: random initial position, alternating dx
        self._floe_x = [
            int(self.rng.integers(0, self._WIDTH))
            for _ in self._FLOE_ROWS
        ]
        self._floe_dx = [1 if i % 2 == 0 else -1 for i in range(len(self._FLOE_ROWS))]

    def _floe_cells(self, li: int) -> set[int]:
        """Return set of x-positions occupied by floe li (with wrap)."""
        x0 = self._floe_x[li]
        return {(x0 + dx) % self._WIDTH for dx in range(self._FLOE_LEN)}

    def _on_floe(self, x: int, y: int) -> int | None:
        """Return floe index if (x,y) is on a floe, else None."""
        for li, row in enumerate(self._FLOE_ROWS):
            if y == row and x in self._floe_cells(li):
                return li
        return None

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        if self._turn > self._SURVIVAL_TURNS and not self._game_over:
            self._message = "Froze out!"
            self._game_over = True
            self._won = False
            return reward, True, info

        # 1. Move floes.
        for li in range(len(self._FLOE_ROWS)):
            self._floe_x[li] = (self._floe_x[li] + self._floe_dx[li]) % self._WIDTH

        # 2. If the player is still on the floe after it drifts, carry them
        # along; standing on the trailing edge can leave the player in water.
        prev_floe = self._on_floe(self._player_x, self._player_y) if self._player_y > 0 else None
        if prev_floe is not None:
            new_x = (self._player_x + self._floe_dx[prev_floe]) % self._WIDTH
            self._player_x = new_x

        # 3. Player action.
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
        if 0 <= nx < self._WIDTH and 0 <= ny < self._HEIGHT:
            self._player_x, self._player_y = nx, ny

        # 4. Check what player is on now.
        py = self._player_y
        if py == 0:
            # Solid shore at top — safe
            pass
        elif py in self._FLOE_ROWS:
            li = self._on_floe(self._player_x, self._player_y)
            if li is None:
                # Fell into water — game over (no penalty per Pattern A)
                self._message = "Fell into the water!"
                self._game_over = True
                self._won = False
                return reward, True, info
            else:
                if li not in self._visited_floes:
                    self._visited_floes.add(li)
                    reward += self._progress_reward(self._WIN_TARGET)
                    self._progress += 1
                    self._message = f"Floe {li} reached! ({self._progress}/{self._WIN_TARGET})"
                    if self._progress >= self._WIN_TARGET:
                        self._on_won()
                        return reward, self._game_over, info
        # Gap rows (1, 3, 5, 7, 9) are safe to stand on — no death, no reward.

        info["progress"] = self._progress
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = []
        for y in range(self._HEIGHT):
            row: list[str] = []
            for _x in range(self._WIDTH):
                if y == 0:
                    row.append("=")  # shore
                elif y in self._FLOE_ROWS:
                    row.append("~")  # water
                else:
                    row.append(" ")  # safe gap (cosmetic — agent can hop through)
            grid.append(row)

        # Floes
        for li, row in enumerate(self._FLOE_ROWS):
            cells = self._floe_cells(li)
            ch = "I" if li in self._visited_floes else "F"
            for x in cells:
                grid[row][x] = ch

        # Player
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            pch = self._DIR_CHARS.get(self._player_dir, "@")
            grid[self._player_y][self._player_x] = pch

        symbols = {
            " ": "open air",
            "=": "shore (safe)",
            "~": "water (deadly)",
            "F": "fresh ice floe (rewards on landing)",
            "I": "visited floe (igloo brick)",
        }
        pch = self._DIR_CHARS.get(self._player_dir, "@")
        symbols[pch] = f"you (facing {self._DIR_NAMES.get(self._player_dir, 'none')})"

        floe_dirs = " ".join("right" if dx == 1 else "left" for dx in self._floe_dx)
        freeze_in = max(0, self._SURVIVAL_TURNS - self._turn)
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Freeze in: {freeze_in} turns    "
            f"Floes visited: {self._progress}/{self._WIN_TARGET}    "
            f"Score: {self._score:.3f}    "
            f"Floe dirs rows 2/4/6/8: {floe_dirs}"
        )

        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Frostbite: hop across 4 drifting ice floes (F) on a 16x10 "
            "grid to build an igloo. Shore (=) at row 0 is safe; rows 2, 4, "
            "6, 8 are water (~) interrupted by 3-cell floes. Floes drift "
            "with alternating directions and wrap around the grid. When you "
            "stand on a floe after it drifts, you drift with it; standing on "
            "a trailing edge can leave you in water. Stepping on a "
            "floe you have not yet visited counts toward your igloo (its "
            "glyph turns I). Visit all 4 floes to win. Falling into water "
            "(any non-floe cell at a water row) ends the run. The shore "
            "row and the gap rows (1, 3, 5, 7, 9) are safe to stand on. "
            f"You freeze out after {self._SURVIVAL_TURNS} turns, so the "
            "HUD shows the remaining freeze timer. "
            "Reward: +1/4 per fresh floe."
        )
