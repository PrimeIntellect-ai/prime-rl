"""miniatari Atlantis.

Identity: Defend a city from descending raider waves with stationary turrets.
Win condition: repel 3 waves of raiders.
Reward: Pattern D, +1/3 per wave repelled, -1 if any city building is destroyed.
Loss: a raider reaches a city building (terminal -1).

Tuning (Phase 3 hardening): hit-radius is exact (the column shot only
touches the column it fired from — no horizontal spread). Added a
shared 3-tick FIRE cooldown so the agent cannot spam LEFT/CENTER/RIGHT
back-to-back. The cooldown gates ALL turrets together, forcing the
agent to choose which column needs the shot most.

Gym ID: glyphbench/miniatari-atlantis-v0
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniAtlantisEnv(MiniatariBase):
    """Mini Atlantis: 16x8 grid, 3 stationary turrets defend a 3-building city.

    Turrets sit at row 6, columns 3, 8, 13. City buildings at row 7. The
    agent picks one of 3 turrets to fire from each tick: LEFT, CENTER,
    RIGHT each fire upward in that turret's column. Each wave has 5
    raiders that share the 3 turret columns; same-column raiders stagger
    above the visible field at y=0,-2,-4,... (off-screen but still
    hittable by the column shot) and march down 1 row every 2 ticks. A
    column shot kills the lowest raider currently in that column.
    Reaching row 7 destroys a city building => -1 terminal.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "CENTER", "RIGHT"),
        descriptions=(
            "do not fire this tick",
            "fire left turret upward",
            "fire center turret upward",
            "fire right turret upward",
        ),
    )

    default_max_turns = 300

    _WIDTH = 16
    _HEIGHT = 8
    _WIN_TARGET = 3
    _N_WAVES = 3
    _RAIDERS_PER_WAVE = 5
    _TURRET_Y = 6
    _CITY_Y = 7
    _TURRETS = (3, 8, 13)
    _RAIDER_MOVE_EVERY = 2
    _FIRE_COOLDOWN = 3  # shared across all turrets

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._raiders: list[list[int]] = []  # [x, y]
        self._wave_idx: int = 0
        self._wave_kills: int = 0
        self._tick_count: int = 0
        self._progress: int = 0
        self._fire_cd: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-atlantis-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._progress = 0
        self._wave_idx = 0
        self._wave_kills = 0
        self._tick_count = 0
        self._fire_cd = 0
        self._raiders = []
        self._spawn_wave()
        self._player_x = self._TURRETS[1]
        self._player_y = self._TURRET_Y

    def _spawn_wave(self) -> None:
        rng = self.rng
        # Spawn columns must be ON turret columns so raiders are killable.
        # Stagger row offsets so raiders don't pile on row 0 instantly.
        # Raiders at y < 0 are off-screen but still hittable; the column
        # shot uses a true -inf sentinel so even raiders at y=-2/-4/...
        # are selected.
        used_cols: list[int] = []
        for _ in range(self._RAIDERS_PER_WAVE):
            tx = self._TURRETS[int(rng.integers(0, len(self._TURRETS)))]
            stack = sum(1 for c in used_cols if c == tx)
            self._raiders.append([tx, -2 * stack])
            used_cols.append(tx)
        self._wave_kills = 0

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._tick_count += 1
        if self._fire_cd > 0:
            self._fire_cd -= 1

        # 1. Fire selected turret. The marker (which turret is "active")
        # always updates on a fire-button input; the *shot* only fires
        # when the shared cooldown is 0.
        firing_col: int | None = None
        if action_name == "LEFT":
            self._player_x = self._TURRETS[0]
            if self._fire_cd == 0:
                firing_col = self._TURRETS[0]
        elif action_name == "CENTER":
            self._player_x = self._TURRETS[1]
            if self._fire_cd == 0:
                firing_col = self._TURRETS[1]
        elif action_name == "RIGHT":
            self._player_x = self._TURRETS[2]
            if self._fire_cd == 0:
                firing_col = self._TURRETS[2]

        if firing_col is not None:
            self._fire_cd = self._FIRE_COOLDOWN
            # Find nearest raider in firing_col (and ±0 spread; precise
            # column shot). Use a true -inf sentinel so off-screen
            # raiders at y=-2, -4, ... are still selectable: previously
            # target_y=-1 silently skipped them.
            target: int | None = None
            target_y: int = -10**9
            for i, (rx, ry) in enumerate(self._raiders):
                if rx == firing_col and ry < self._TURRET_Y and ry > target_y:
                    target = i
                    target_y = ry
            if target is not None:
                self._raiders.pop(target)
                self._wave_kills += 1
                self._message = "Raider destroyed!"
                if self._wave_kills >= self._RAIDERS_PER_WAVE:
                    self._wave_idx += 1
                    reward += self._progress_reward(self._WIN_TARGET)
                    self._progress += 1
                    self._message = f"Wave {self._wave_idx} repelled! ({self._progress}/{self._WIN_TARGET})"
                    if self._progress >= self._WIN_TARGET:
                        self._on_won()
                        return reward, self._game_over, info
                    else:
                        self._spawn_wave()

        # 2. Raiders march down
        if self._tick_count % self._RAIDER_MOVE_EVERY == 0:
            for r in self._raiders:
                r[1] += 1
            # Check city breach. Use += so we never overwrite a wave-clear
            # bonus (+1/3) earned earlier the same tick.
            for r in self._raiders:
                if r[1] >= self._CITY_Y:
                    self._message = "Raider reached the city!"
                    reward += self._death_reward()
                    self._on_life_lost()
                    return reward, True, info

        info["progress"] = self._progress
        info["wave"] = self._wave_idx + 1
        info["wave_kills"] = self._wave_kills
        info["raiders"] = len(self._raiders)
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        # City row
        for x in range(self._WIDTH):
            grid[self._CITY_Y][x] = "█"
        # Turrets
        for tx in self._TURRETS:
            grid[self._TURRET_Y][tx] = "T"
        # Raiders
        for rx, ry in self._raiders:
            if 0 <= rx < self._WIDTH and 0 <= ry < self._HEIGHT:
                grid[ry][rx] = "R"
        # Player marker (selected turret)
        if 0 <= self._player_x < self._WIDTH:
            grid[self._TURRET_Y][self._player_x] = "Y"

        symbols = {
            " ": "sky",
            "█": "city building",
            "T": "turret (idle)",
            "Y": "selected turret (you)",
            "R": "raider",
        }

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Wave: {self._wave_idx + 1}/{self._N_WAVES}    "
            f"Wave kills: {self._wave_kills}/{self._RAIDERS_PER_WAVE}    "
            f"Score: {self._score:.3f}    "
            f"Raiders: {len(self._raiders)}    "
            f"Fire cd: {self._fire_cd}"
        )

        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Atlantis on a 16x8 grid. Three stationary turrets (T) "
            "sit at row 6 in columns 3, 8, 13, defending the city (█) "
            "at row 7. Each wave spawns 5 raiders sharing the 3 turret "
            "columns; raiders that share a column are staggered above "
            "the visible field (off-screen but still hittable) and "
            "march down one row every 2 ticks until they enter view. "
            "Each tick you choose LEFT, CENTER, or RIGHT to fire the "
            "matching turret straight up its column, killing the lowest "
            "raider currently in that exact column (no horizontal "
            "spread). All three turrets share a single 3-tick FIRE "
            "cooldown — pressing LEFT/CENTER/RIGHT while cooldown > 0 "
            "still moves the active marker but does not fire. NOOP "
            "skips the tick. After all 5 raiders in a wave die, the "
            "next wave spawns. Repel 3 waves to win. If any raider "
            "reaches the city row, you take a -1 terminal penalty. "
            "Reward: +1/3 per wave cleared, -1 on city breach."
        )
