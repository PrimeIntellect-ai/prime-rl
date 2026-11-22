"""miniatari Assault.

Identity: Bottom turret defends against descending enemy ships in formation.
Win condition: clear 5 enemy ships.
Reward: Pattern A, +1/5 per ship destroyed.
Loss: a ship reaches the turret row (no -1; just terminates).

Tuning (Phase 3 hardening): the turret now drifts in the wind. On any
NOOP / FIRE tick the turret automatically slides one cell in its
current wind direction (and bounces off the side walls). The wind
direction is exposed via the HUD so the agent can plan ahead.
LEFT/RIGHT actions still nudge the turret directly and consume that
tick's drift, so the agent has full control while acting. This breaks
the audit's "stationary FIRE wins" bias by punishing inaction with
position drift.

Gym ID: glyphbench/miniatari-assault-v0
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniAssaultEnv(MiniatariBase):
    """Mini Assault: 14x12 grid, 5 enemy ships in a tight V formation.

    Turret (Y) at row 11. Enemies (E) start in a V at rows 1..3 and
    descend 1 row every 4 ticks. Each enemy has a fixed horizontal
    drift bouncing off side walls every 2 ticks. FIRE shoots straight
    up the turret column instantly, destroying the lowest enemy in that
    column. Reaching turret row ends the run. Clearing all 5 wins.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "FIRE"),
        descriptions=(
            "do nothing",
            "move turret left one cell",
            "move turret right one cell",
            "fire a bullet straight up your column",
        ),
    )

    default_max_turns = 200

    _WIDTH = 14
    _HEIGHT = 12
    _WIN_TARGET = 5
    _TURRET_Y = 11
    _DRIFT_EVERY = 2
    _ADVANCE_EVERY = 4

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        # enemies: [x, y, dx]
        self._enemies: list[list[int]] = []
        self._tick_count: int = 0
        self._progress: int = 0
        self._wind_dx: int = 1  # +1 / -1; bounces off walls

    def env_id(self) -> str:
        return "glyphbench/miniatari-assault-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._progress = 0
        self._tick_count = 0
        rng = self.rng
        # Off-centre V formation so the turret cannot win by sitting at
        # the default column. The leader's column is sampled in the
        # left or right third.
        cx = int(rng.integers(2, self._WIDTH - 2))
        formation = [
            (cx, 1),
            (cx - 1, 2),
            (cx + 1, 2),
            (cx - 2, 3),
            (cx + 2, 3),
        ]
        self._enemies = []
        for fx, fy in formation:
            dx = 1 if rng.random() < 0.5 else -1
            self._enemies.append([fx, fy, dx])
        self._player_x = self._WIDTH // 2
        self._player_y = self._TURRET_Y
        # Wind direction sampled per episode.
        self._wind_dx = 1 if rng.random() < 0.5 else -1

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._tick_count += 1

        # 1. Turret move. LEFT/RIGHT take direct control and consume
        # the wind-drift for this tick. NOOP / FIRE let the wind drift
        # the turret one cell, bouncing off the walls.
        if action_name == "LEFT":
            self._player_dir = (-1, 0)
            if self._player_x > 0:
                self._player_x -= 1
        elif action_name == "RIGHT":
            self._player_dir = (1, 0)
            if self._player_x < self._WIDTH - 1:
                self._player_x += 1
        else:
            # Wind drift.
            new_x = self._player_x + self._wind_dx
            if new_x < 0 or new_x >= self._WIDTH:
                self._wind_dx = -self._wind_dx
                new_x = self._player_x + self._wind_dx
            self._player_x = max(0, min(self._WIDTH - 1, new_x))

        # 2. Fire (instantaneous column shot)
        if action_name == "FIRE":
            target: int | None = None
            target_y: int = -1
            for i, (ex, ey, _edx) in enumerate(self._enemies):
                if ex == self._player_x and ey < self._TURRET_Y and ey > target_y:
                    target = i
                    target_y = ey
            if target is not None:
                self._enemies.pop(target)
                reward += self._progress_reward(self._WIN_TARGET)
                self._progress += 1
                self._message = f"Enemy down! ({self._progress}/{self._WIN_TARGET})"
                if self._progress >= self._WIN_TARGET:
                    self._on_won()
                    return reward, self._game_over, info

        # 3. Drift horizontally (every K ticks)
        if self._tick_count % self._DRIFT_EVERY == 0:
            for e in self._enemies:
                e[0] += e[2]
                if e[0] <= 0 or e[0] >= self._WIDTH - 1:
                    e[2] *= -1
                    e[0] = max(0, min(self._WIDTH - 1, e[0]))

        # 4. Advance one row down (every K ticks)
        if self._tick_count % self._ADVANCE_EVERY == 0:
            for e in self._enemies:
                e[1] += 1

        # 5. Reached turret row?
        for _ex, ey, _edx in self._enemies:
            if ey >= self._TURRET_Y:
                self._message = "Enemy reached the turret row!"
                self._game_over = True
                self._won = False
                return reward, True, info

        info["progress"] = self._progress
        info["enemies_left"] = len(self._enemies)
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        # Turret row line
        for x in range(self._WIDTH):
            grid[self._TURRET_Y][x] = "_"
        # Enemies
        for ex, ey, _edx in self._enemies:
            if 0 <= ex < self._WIDTH and 0 <= ey < self._HEIGHT:
                grid[ey][ex] = "E"
        # Turret
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            grid[self._player_y][self._player_x] = "Y"

        symbols = {
            " ": "space",
            "_": "turret row (defense line)",
            "E": "enemy ship",
            "Y": "your turret",
        }
        wind_arrow = "→" if self._wind_dx > 0 else "←"
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Killed: {self._progress}/{self._WIN_TARGET}    "
            f"Score: {self._score:.3f}    "
            f"Enemies: {len(self._enemies)}    "
            f"Wind: {wind_arrow}"
        )
        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Assault on a 14x12 grid. 5 enemy ships (E) start in a "
            "V formation on rows 1..3 (the formation centre is sampled "
            "each episode, not pinned to the screen midline). Each "
            "enemy drifts horizontally one cell every 2 ticks (bouncing "
            "off side walls), and the entire formation advances one row "
            "every 4 ticks. Your turret (Y) sits on row 11. LEFT/RIGHT "
            "nudges it directly. NOOP / FIRE allow the WIND to drift "
            "the turret one cell in the wind direction (HUD shows "
            "→ or ←); the wind reflects off the side walls. FIRE "
            "instantly destroys the lowest enemy in the turret's "
            "current column. If any enemy reaches the turret row, the "
            "run ends. Destroy all 5 to win. Reward: +1/5 per kill."
        )
