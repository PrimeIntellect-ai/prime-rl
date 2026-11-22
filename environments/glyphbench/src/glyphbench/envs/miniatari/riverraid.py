"""miniatari River Raid.

Identity: Fly down a narrow river shooting or dodging obstacles.
Win condition: clear/pass 8 obstacles AND reach the river's end.
Reward: Pattern D, +1/9 per progress unit (8 obstacles + final goal);
-1 on collision with an unshot obstacle.

Gym ID: glyphbench/miniatari-riverraid-v0

Random baseline (seed=0..29): success_rate=0%, mean_length=43, mean_return=-0.044
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniRiverRaidEnv(MiniatariBase):
    """Mini River Raid: 12-wide river that scrolls upward; 8 obstacles.

    The world is 12 wide x 16 tall, with banks at columns 0-1 and 10-11
    (visible walls █). The player jet (Y) sits at row 14 and moves
    LEFT/RIGHT (1 cell each). Obstacles (T) are at scattered positions
    along the river; the world scrolls up 1 row every 2 ticks (player's
    position fixed; obstacles drift down toward the player). FIRE shoots
    a missile straight up the player's column, destroying the closest
    obstacle there if any. Each cleared (shot) or passed (drifted past
    player row) obstacle counts as +1/9 progress. After all 8 obstacles
    are dealt with AND 50 ticks have elapsed, the river ends — final
    +1/9 (win). Collision with an obstacle (it reaches the player's
    cell) is -1 terminal.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "FIRE"),
        descriptions=(
            "do nothing",
            "shift jet left one column",
            "shift jet right one column",
            "fire a missile straight up your column",
        ),
    )

    default_max_turns = 300

    _WIDTH = 12
    _HEIGHT = 16
    _N_OBSTACLES = 8
    _PLAYER_Y = 14
    _LEFT_BANK = 1   # cols 0..LEFT_BANK are bank
    _RIGHT_BANK = 10  # cols RIGHT_BANK..WIDTH-1 are bank
    _SCROLL_EVERY = 2
    _RIVER_END_TICK = 50
    _PROGRESS_TARGET = 9  # 8 obstacles + final goal

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._obstacles: list[list[int]] = []  # [col, y]; col in [LEFT_BANK+1..RIGHT_BANK-1]
        self._tick_count: int = 0
        self._cleared: int = 0
        self._progress: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-riverraid-v0"

    def _is_bank(self, x: int) -> bool:
        return x <= self._LEFT_BANK or x >= self._RIGHT_BANK

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._tick_count = 0
        self._cleared = 0
        self._progress = 0
        self._player_x = (self._LEFT_BANK + self._RIGHT_BANK) // 2
        self._player_y = self._PLAYER_Y
        self._player_dir = (0, -1)
        rng = self.rng
        # Place 8 obstacles at distinct rows above the player. Columns
        # are clamped to the player's reachable range [LEFT_BANK+2,
        # RIGHT_BANK-2] = [3, 8] so every obstacle is shootable.
        # Previously cols 2 and 9 were possible spawns, leaving some
        # seeds unwinnable since the player cannot move into those.
        min_col = self._LEFT_BANK + 2
        max_col = self._RIGHT_BANK - 1  # randint upper bound is exclusive
        used_rows: set[int] = set()
        self._obstacles = []
        while len(self._obstacles) < self._N_OBSTACLES:
            row = int(rng.integers(0, self._PLAYER_Y - 1))
            if row in used_rows:
                continue
            used_rows.add(row)
            col = int(rng.integers(min_col, max_col))
            self._obstacles.append([col, row])

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._tick_count += 1

        # 1. Player move (constrained between banks)
        if action_name == "LEFT" and self._player_x > self._LEFT_BANK + 1:
            self._player_x -= 1
        elif action_name == "RIGHT" and self._player_x < self._RIGHT_BANK - 1:
            self._player_x += 1

        # 2. Fire missile
        if action_name == "FIRE":
            target: int | None = None
            target_y: int = -1
            for i, (cx, cy) in enumerate(self._obstacles):
                if cx == self._player_x and cy < self._player_y and cy > target_y:
                    target = i
                    target_y = cy
            if target is not None:
                self._obstacles.pop(target)
                self._cleared += 1
                reward += self._progress_reward(self._PROGRESS_TARGET)
                self._progress += 1
                self._message = f"Obstacle down! ({self._progress}/{self._PROGRESS_TARGET})"

        # 3. Scroll: obstacles drift down 1 row every K ticks
        if self._tick_count % self._SCROLL_EVERY == 0:
            survivors: list[list[int]] = []
            for cx, cy in self._obstacles:
                ncy = cy + 1
                if ncy == self._player_y and cx == self._player_x:
                    self._message = "Crashed into an obstacle!"
                    reward += self._death_reward()
                    self._on_life_lost()
                    return reward, True, info
                if ncy > self._player_y:
                    # Passed below player without colliding: NO progress
                    # credit (must be shot to count). It still drifts off
                    # the field.
                    continue
                survivors.append([cx, ncy])
            self._obstacles = survivors

        # 4. River end: when tick budget reached, episode ends. Final
        # +1/9 only if all 8 obstacles were shot.
        if self._tick_count >= self._RIVER_END_TICK:
            if self._cleared >= self._N_OBSTACLES:
                reward += self._progress_reward(self._PROGRESS_TARGET)
                self._progress += 1
                self._message = "Reached the river's end!"
                self._on_won()
            else:
                self._message = (
                    f"River ends — only {self._cleared}/{self._N_OBSTACLES} "
                    f"obstacles cleared."
                )
                self._game_over = True
            return reward, True, info

        info["progress"] = self._progress
        info["cleared"] = self._cleared
        info["obstacles_in_play"] = len(self._obstacles)
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        for y in range(self._HEIGHT):
            for x in range(self._WIDTH):
                if self._is_bank(x):
                    grid[y][x] = "█"
        for cx, cy in self._obstacles:
            if 0 <= cx < self._WIDTH and 0 <= cy < self._HEIGHT:
                grid[cy][cx] = "T"
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            grid[self._player_y][self._player_x] = "Y"

        symbols = {
            " ": "river water",
            "█": "river bank",
            "T": "obstacle",
            "Y": "your jet",
        }

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Progress: {self._progress}/{self._PROGRESS_TARGET}    "
            f"Cleared: {self._cleared}/{self._N_OBSTACLES}    "
            f"Score: {self._score:.3f}    "
            f"Obstacles: {len(self._obstacles)}"
        )
        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini River Raid: a 12x16 vertical river with banks (█) at "
            "cols 0-1 and 10-11. Your jet (Y) sits at row 14, columns "
            "2-9. LEFT/RIGHT shifts you 1 column. FIRE shoots a missile "
            "straight up your column, destroying the lowest obstacle (T) "
            "in that column for +1/9 progress. The river scrolls down 1 "
            "row every 2 ticks, drifting obstacles toward you. Obstacles "
            "that drift past your row without being shot are gone but "
            "give NO progress credit. After 50 ticks the river ends: if "
            "all 8 obstacles were shot, you reach the end (+1/9 final, "
            "win); otherwise the episode ends with whatever partial "
            "progress you earned. Collision with an obstacle is -1 "
            "terminal."
        )
