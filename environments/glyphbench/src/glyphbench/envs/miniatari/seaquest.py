"""miniatari Seaquest.

Identity: Submarine rescues divers and resurfaces before air runs out.
Win condition: rescue 3 divers and surface (4 phases).
Reward: Pattern D, +1/4 per phase; -1 if oxygen runs out underwater.

Gym ID: glyphbench/miniatari-seaquest-v0

Random baseline (seed=0..29): success_rate=0%, mean_length=166, mean_return=-0.833
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniSeaquestEnv(MiniatariBase):
    """Mini Seaquest: 14x12 ocean; rescue 3 divers, then surface.

    The ocean is 14x12 with surface row at y=0. Sub (Y, arrow) starts
    at the surface. 3 divers (D) sit at distinct interior cells.
    Picking up a diver (sub on diver cell) gives +1/4. Sub has an
    oxygen counter that depletes by 1 when underwater (y > 0) each
    tick, capacity 80. Returning to surface row (y == 0) refills
    oxygen. After picking up all 3 divers, returning to surface gives
    the final +1/4 (terminal win). Pattern D: -1 if oxygen hits 0.
    No enemies in this miniaturized version (the spec says +1/4 per
    phase suffices to discriminate).
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "UP", "DOWN"),
        descriptions=(
            "do nothing",
            "thrust left and face left",
            "thrust right and face right",
            "rise up and face up",
            "dive down and face down",
        ),
    )

    default_max_turns = 400

    _WIDTH = 14
    _HEIGHT = 12
    _N_DIVERS = 3
    _WIN_TARGET = 4  # 3 divers + 1 surfacing
    _OXY_MAX = 80

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._divers: set[tuple[int, int]] = set()
        self._rescued: int = 0
        self._oxygen: int = 0
        self._progress: int = 0
        self._tick_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-seaquest-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._tick_count = 0
        self._progress = 0
        self._rescued = 0
        self._oxygen = self._OXY_MAX
        self._player_x = self._WIDTH // 2
        self._player_y = 0
        self._player_dir = (0, 1)
        rng = self.rng
        used: set[tuple[int, int]] = {(self._player_x, self._player_y)}
        self._divers = set()
        while len(self._divers) < self._N_DIVERS:
            x = int(rng.integers(0, self._WIDTH))
            y = int(rng.integers(2, self._HEIGHT))  # below surface
            if (x, y) in used:
                continue
            used.add((x, y))
            self._divers.add((x, y))

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._tick_count += 1

        # 1. Sub move
        nx, ny = self._player_x, self._player_y
        if action_name == "LEFT":
            nx = max(0, nx - 1)
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT":
            nx = min(self._WIDTH - 1, nx + 1)
            self._player_dir = (1, 0)
        elif action_name == "UP":
            ny = max(0, ny - 1)
            self._player_dir = (0, -1)
        elif action_name == "DOWN":
            ny = min(self._HEIGHT - 1, ny + 1)
            self._player_dir = (0, 1)
        self._player_x, self._player_y = nx, ny

        # 2. Pick up diver
        cell = (self._player_x, self._player_y)
        if cell in self._divers:
            self._divers.discard(cell)
            self._rescued += 1
            reward += self._progress_reward(self._WIN_TARGET)
            self._progress += 1
            self._message = f"Diver rescued! ({self._rescued}/{self._N_DIVERS})"

        # 3. Surface logic
        if self._player_y == 0:
            self._oxygen = self._OXY_MAX
            # Final win when all rescued and at surface
            if self._rescued >= self._N_DIVERS and self._progress < self._WIN_TARGET:
                reward += self._progress_reward(self._WIN_TARGET)
                self._progress += 1
                self._message = "Surfaced with all divers!"
                self._on_won()
                return reward, self._game_over, info
        else:
            self._oxygen -= 1
            if self._oxygen <= 0:
                self._message = "Out of oxygen!"
                reward += self._death_reward()
                self._on_life_lost()
                return reward, True, info

        info["progress"] = self._progress
        info["rescued"] = self._rescued
        info["oxygen"] = self._oxygen
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        # Surface row (above water indicator at y=0; sea below)
        for x in range(self._WIDTH):
            grid[0][x] = "≈"
        # Divers
        for dx, dy in self._divers:
            if 0 <= dx < self._WIDTH and 0 <= dy < self._HEIGHT:
                grid[dy][dx] = "D"
        # Sub
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            pch = self._DIR_CHARS.get(self._player_dir, "@")
            grid[self._player_y][self._player_x] = pch

        symbols = {
            " ": "water",
            "≈": "surface",
            "D": "diver",
        }
        pch = self._DIR_CHARS.get(self._player_dir, "@")
        symbols[pch] = f"your sub (facing {self._DIR_NAMES.get(self._player_dir, 'down')})"

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Rescued: {self._rescued}/{self._N_DIVERS}    "
            f"Phase: {self._progress}/{self._WIN_TARGET}    "
            f"O2: {self._oxygen}/{self._OXY_MAX}    "
            f"Score: {self._score:.3f}    "
            f"Divers: {len(self._divers)}"
        )
        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Seaquest in a 14x12 ocean. Surface (≈) is row 0; below "
            "is open water. Your sub (Y, arrow shows facing) starts at "
            "the surface with 80 oxygen. 3 divers (D) sit at scattered "
            "underwater cells. LEFT/RIGHT/UP/DOWN moves 1 cell. Picking "
            "up a diver (entering its cell) is +1/4. Returning to row 0 "
            "refills oxygen. After all 3 are aboard, returning to row 0 "
            "is the final +1/4 (win). Underwater oxygen depletes by 1 "
            "each tick; if it hits 0, -1 terminal."
        )
