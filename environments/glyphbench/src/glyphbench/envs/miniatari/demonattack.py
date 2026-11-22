"""miniatari Demon Attack.

Identity: Bottom cannon shoots demons that swoop and drop bombs.
Win condition: clear 1 wave of 5 demons.
Reward: Pattern D, +1/5 per kill, -1 if a bomb hits the player.
Loss: bomb hits player (terminal -1).

Gym ID: glyphbench/miniatari-demonattack-v0

Random baseline (seed=0..29): success_rate=20%, mean_length=31, mean_return=-0.367
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniDemonAttackEnv(MiniatariBase):
    """Mini Demon Attack: 14x10 grid, 5 demons + drifting bombs.

    Player cannon at row 9. Demons sit at rows 1..3, drifting horizontally.
    Each tick, with low probability, an active demon drops a bomb that
    falls 1 cell/tick. FIRE shoots a bullet up the player's column.
    Clearing all 5 demons wins; getting hit by a bomb kills (-1).
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "FIRE"),
        descriptions=(
            "do nothing",
            "move cannon left",
            "move cannon right",
            "fire a bullet up your column",
        ),
    )

    default_max_turns = 200

    _WIDTH = 14
    _HEIGHT = 10
    _WIN_TARGET = 5
    _CANNON_Y = 9
    _DEMON_ROWS = (1, 2, 3)
    _BOMB_DROP_PROB = 0.15

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._demons: list[list[int]] = []  # per-demon: [x, y, dx]
        self._bombs: list[list[int]] = []  # per-bomb: [x, y]
        self._progress: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-demonattack-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._progress = 0
        self._demons = []
        self._bombs = []
        rng = self.rng
        # Place exactly 5 demons without replacement. The arena has many more
        # slots than demons, so generation should never silently under-spawn.
        slots = [
            (x, y)
            for y in self._DEMON_ROWS
            for x in range(1, self._WIDTH - 1)
        ]
        order = self.rng.permutation(len(slots))[: self._WIN_TARGET]
        for idx in order:
            x, y = slots[int(idx)]
            dx = 1 if rng.random() < 0.5 else -1
            self._demons.append([x, y, dx])
        self._player_x = self._WIDTH // 2
        self._player_y = self._CANNON_Y

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        # 1. Cannon move
        if action_name == "LEFT" and self._player_x > 0:
            self._player_x -= 1
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT" and self._player_x < self._WIDTH - 1:
            self._player_x += 1
            self._player_dir = (1, 0)

        # 2. Fire (instantaneous column shot)
        if action_name == "FIRE":
            target: int | None = None
            target_y: int = -1
            for i, (dx_pos, dy_pos, _ddx) in enumerate(self._demons):
                if dx_pos == self._player_x and dy_pos > target_y:
                    target = i
                    target_y = dy_pos
            if target is not None:
                self._demons.pop(target)
                reward += self._progress_reward(self._WIN_TARGET)
                self._progress += 1
                self._message = f"Demon down! ({self._progress}/{self._WIN_TARGET})"
                if self._progress >= self._WIN_TARGET:
                    self._on_won()
                    return reward, self._game_over, info

        # 3. Move demons (drift and bounce)
        for d in self._demons:
            d[0] += d[2]
            if d[0] <= 0 or d[0] >= self._WIDTH - 1:
                d[2] *= -1
                d[0] = max(0, min(self._WIDTH - 1, d[0]))

        # 4. Drop bombs (each demon may drop)
        rng = self.rng
        for d in self._demons:
            if rng.random() < self._BOMB_DROP_PROB:
                self._bombs.append([d[0], d[1] + 1])

        # 5. Move bombs down
        survived: list[list[int]] = []
        for b in self._bombs:
            b[1] += 1
            if b[1] < self._HEIGHT:
                survived.append(b)
        self._bombs = survived

        # 6. Check bomb hits player
        for b in self._bombs:
            if b[0] == self._player_x and b[1] == self._player_y:
                self._message = "Bomb hit your cannon!"
                reward += self._death_reward()
                self._on_life_lost()
                return reward, True, info

        info["progress"] = self._progress
        info["demons_left"] = len(self._demons)
        info["bombs_in_air"] = len(self._bombs)
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        # Cannon row marker
        for x in range(self._WIDTH):
            grid[self._CANNON_Y][x] = "_"
        # Demons
        for dx_pos, dy_pos, _ddx in self._demons:
            if 0 <= dx_pos < self._WIDTH and 0 <= dy_pos < self._HEIGHT:
                grid[dy_pos][dx_pos] = "D"
        # Bombs
        for bx, by in self._bombs:
            if 0 <= bx < self._WIDTH and 0 <= by < self._HEIGHT:
                grid[by][bx] = "*"
        # Cannon
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            grid[self._player_y][self._player_x] = "Y"

        symbols = {
            " ": "empty space",
            "_": "cannon row",
            "D": "demon",
            "*": "falling bomb",
            "Y": "your cannon",
        }
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Killed: {self._progress}/{self._WIN_TARGET}    "
            f"Score: {self._score:.3f}    "
            f"Demons: {len(self._demons)}    "
            f"Bombs: {len(self._bombs)}"
        )

        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Demon Attack on a 14x10 grid. 5 demons (D) drift along "
            "rows 1-3, bouncing off side walls. Your cannon (Y) sits at "
            "row 9. LEFT/RIGHT slides it; FIRE instantly destroys the "
            "lowest demon in your current column. Each tick a demon may "
            "drop a bomb (*) that falls 1 cell/tick. Clearing all 5 "
            "demons wins (+1/5 per kill). A bomb hitting your cannon "
            "ends the run with -1. Reward: +1/5 per kill, -1 on bomb hit."
        )
