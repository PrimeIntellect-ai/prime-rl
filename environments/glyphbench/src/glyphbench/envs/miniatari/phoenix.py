"""miniatari Phoenix.

Identity: Bottom cannon shoots a wave of swooping phoenix birds.
Win condition: clear 1 wave of 6 phoenixes.
Reward: Pattern A, +1/6 per phoenix destroyed.
Loss: a phoenix reaches the cannon row (no -1; just terminates).

Tuning (Phase 3 hardening): dive probability bumped from 0.18 to 0.32
so the cannon row is regularly threatened, and phoenix spawn columns
are sampled so none start directly above the player's default column.
This eliminates the audit's "phoenixes fall into player column for
free" bias.

Gym ID: glyphbench/miniatari-phoenix-v0
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniPhoenixEnv(MiniatariBase):
    """Mini Phoenix: 14x10 grid, 6 phoenixes that swoop and dive.

    Cannon (Y) at row 9. 6 phoenixes (P) start at rows 1..3 spread across
    the field. Each tick a phoenix has a small chance to dive: drop one
    row and shift toward the cannon's column. FIRE shoots a bullet up the
    cannon's column instantly, destroying the lowest phoenix in that
    column. If any phoenix reaches the cannon row, the run ends.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "FIRE"),
        descriptions=(
            "do nothing",
            "move cannon left one cell",
            "move cannon right one cell",
            "fire a bullet straight up your column",
        ),
    )

    default_max_turns = 200

    _WIDTH = 14
    _HEIGHT = 10
    _WIN_TARGET = 6
    _CANNON_Y = 9
    _DIVE_PROB = 0.32
    _ADVANCE_EVERY = 6  # base downward drift

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._phoenixes: list[list[int]] = []  # [x, y]
        self._tick_count: int = 0
        self._progress: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-phoenix-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._progress = 0
        self._tick_count = 0
        self._phoenixes = []
        rng = self.rng
        # 6 phoenixes scattered on rows 1..3, but no phoenix may start
        # in the player's default column (cannon spawn). This forces
        # the agent to actively pan the cannon to engage; previously
        # alignment-by-default leaked free credit.
        cannon_col = self._WIDTH // 2
        positions: set[tuple[int, int]] = set()
        for _ in range(self._WIN_TARGET):
            for _attempt in range(40):
                x = int(rng.integers(1, self._WIDTH - 1))
                if x == cannon_col:
                    continue
                y = int(rng.integers(1, 4))
                if (x, y) not in positions:
                    positions.add((x, y))
                    self._phoenixes.append([x, y])
                    break
        self._player_x = cannon_col
        self._player_y = self._CANNON_Y

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._tick_count += 1

        # 1. Cannon move
        if action_name == "LEFT" and self._player_x > 0:
            self._player_x -= 1
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT" and self._player_x < self._WIDTH - 1:
            self._player_x += 1
            self._player_dir = (1, 0)

        # 2. Fire (instantaneous column shot, lowest in column)
        if action_name == "FIRE":
            target: int | None = None
            target_y: int = -1
            for i, (px, py) in enumerate(self._phoenixes):
                if px == self._player_x and py < self._CANNON_Y and py > target_y:
                    target = i
                    target_y = py
            if target is not None:
                self._phoenixes.pop(target)
                reward += self._progress_reward(self._WIN_TARGET)
                self._progress += 1
                self._message = f"Phoenix down! ({self._progress}/{self._WIN_TARGET})"
                if self._progress >= self._WIN_TARGET:
                    self._on_won()
                    return reward, self._game_over, info

        # 3. Phoenixes dive randomly
        rng = self.rng
        for p in self._phoenixes:
            if rng.random() < self._DIVE_PROB:
                p[1] += 1
                if p[0] < self._player_x:
                    p[0] = min(self._WIDTH - 1, p[0] + 1)
                elif p[0] > self._player_x:
                    p[0] = max(0, p[0] - 1)

        # 4. Periodic baseline drift (every K ticks all advance one)
        if self._tick_count % self._ADVANCE_EVERY == 0:
            for p in self._phoenixes:
                p[1] += 1

        # 5. Reach cannon row?
        for p in self._phoenixes:
            if p[1] >= self._CANNON_Y:
                self._message = "A phoenix reached the cannon row!"
                self._game_over = True
                self._won = False
                return reward, True, info

        info["progress"] = self._progress
        info["phoenixes_left"] = len(self._phoenixes)
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        # Cannon row line
        for x in range(self._WIDTH):
            grid[self._CANNON_Y][x] = "_"
        # Phoenixes
        for px, py in self._phoenixes:
            if 0 <= px < self._WIDTH and 0 <= py < self._HEIGHT:
                grid[py][px] = "P"
        # Cannon
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            grid[self._player_y][self._player_x] = "Y"

        symbols = {
            " ": "sky",
            "_": "cannon row (defense line)",
            "P": "phoenix",
            "Y": "your cannon",
        }
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Killed: {self._progress}/{self._WIN_TARGET}    "
            f"Score: {self._score:.3f}    "
            f"Phoenixes: {len(self._phoenixes)}"
        )
        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Phoenix on a 14x10 grid. 6 phoenixes (P) scatter on "
            "rows 1..3, never directly above your starting cannon "
            "column. Each tick each phoenix has a 32% chance to dive: "
            "drop one row and shift one column toward your cannon's "
            "column. Every 6 ticks all phoenixes drift one row down "
            "regardless. Your cannon (Y) sits at row 9. LEFT/RIGHT "
            "slides it; FIRE instantly destroys the lowest phoenix in "
            "your column. Clearing all 6 wins. If any phoenix reaches "
            "the cannon row, the run ends. Reward: +1/6 per kill."
        )
