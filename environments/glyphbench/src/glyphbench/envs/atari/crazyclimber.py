"""Atari CrazyClimber environment.

Climb a building while avoiding falling objects and hazards.

Gym ID: glyphbench/atari-crazyclimber-v0
"""

from __future__ import annotations

from typing import Any

import numpy as np

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase

_W = 12
_H = 24
_WINDOW_CHAR = "W"
_LEDGE_CHAR = "="
_FALLING_OBJ = "X"

class CrazyClimberEnv(AtariBase):
    """CrazyClimber: climb a building dodging hazards.

    Grid: 12x24. The building scrolls as the player climbs.
    Windows open/close. Falling objects drop from above.

    Actions: NOOP, UP, DOWN, LEFT, RIGHT
    Pattern D: +1/_WIN_TARGET per building topped (full-scope = 5
    buildings), -1.0 if hit by falling object (single-life model).
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP", "DOWN", "LEFT", "RIGHT"),
        descriptions=(
            "do nothing",
            "climb up one row",
            "climb down one row",
            "move left",
            "move right",
        ),
    )

    # Pattern D full-scope target: 5 buildings climbed.
    _WIN_TARGET: int = 5
    _DEATH_PENALTY: float = -1.0

    def __init__(self, max_turns: int = 1000) -> None:
        super().__init__(max_turns=max_turns)
        self._height_reached: int = 0
        self._scroll_offset: int = 0
        self._building: list[list[str]] = []
        self._bld_height: int = 100
        self._obj_timer: int = 0
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-crazyclimber-v0"

    def _reset(self, seed: int) -> GridObservation:
        self._progress_count = 0
        return super()._reset(seed)

    def _generate_level(self, seed: int) -> None:
        rng = np.random.default_rng(seed + self._level * 1009)
        self._bld_height = 60 + self._level * 20
        self._height_reached = 0
        self._scroll_offset = 0
        self._obj_timer = 0
        self._entities = []

        # Build the full building column map
        self._building = [
            [" " for _ in range(_W)]
            for _ in range(self._bld_height)
        ]
        # Walls on sides
        for y in range(self._bld_height):
            self._building[y][0] = "│"
            self._building[y][1] = "│"
            self._building[y][_W - 1] = "│"
            self._building[y][_W - 2] = "│"
        # Ledges every 8 rows
        for y in range(0, self._bld_height, 8):
            for x in range(2, _W - 2):
                self._building[y][x] = _LEDGE_CHAR
        # Windows between walls
        for y in range(2, self._bld_height - 2, 4):
            for x in [3, _W - 4]:
                if rng.random() < 0.5:
                    self._building[y][x] = _WINDOW_CHAR

        # Player starts near bottom
        self._player_x = _W // 2
        self._player_y = _H - 3
        self._scroll_offset = max(
            0, self._bld_height - _H
        )

        self._rebuild_grid()

    def _rebuild_grid(self) -> None:
        """Copy visible portion of building into the grid."""
        self._init_grid(_W, _H)
        for gy in range(_H):
            by = self._scroll_offset + gy
            if 0 <= by < self._bld_height:
                for gx in range(_W):
                    self._set_cell(gx, gy, self._building[by][gx])
        # Top/bottom borders
        for x in range(_W):
            self._set_cell(x, 0, "─")
            self._set_cell(x, _H - 1, "─")

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        dx, dy = 0, 0
        if action_name == "UP":
            dy = -1
            self._player_dir = (0, -1)
        elif action_name == "DOWN":
            dy = 1
            self._player_dir = (0, 1)
        elif action_name == "LEFT":
            dx = -1
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT":
            dx = 1
            self._player_dir = (1, 0)

        # Calculate building coords
        bld_y = self._scroll_offset + self._player_y
        new_px = self._player_x + dx
        new_by = bld_y + dy

        # Horizontal movement: stay on building surface
        if 2 <= new_px <= _W - 3:
            self._player_x = new_px

        # Vertical movement
        if 0 < new_by < self._bld_height - 1:
            bx = self._player_x
            cell = self._building[new_by][bx]
            if cell != "│":
                old_by = bld_y
                # Scroll if player goes above midpoint
                if dy < 0:
                    self._player_y += dy
                    if self._player_y < _H // 3:
                        shift = _H // 3 - self._player_y
                        self._scroll_offset -= shift
                        self._player_y += shift
                        if self._scroll_offset < 0:
                            self._scroll_offset = 0
                            self._player_y = (
                                new_by - self._scroll_offset
                            )
                    # Track height for HUD only (no reward).
                    h = self._bld_height - new_by
                    if h > self._height_reached:
                        delta = h - self._height_reached
                        self._height_reached = h
                        self._on_point_scored(delta)
                elif dy > 0:
                    self._player_y += dy
                    if self._player_y >= _H - 2:
                        self._player_y = _H - 3

                # Ledge HUD message only (no reward).
                if (
                    self._building[new_by][self._player_x]
                    == _LEDGE_CHAR
                    and old_by != new_by
                ):
                    self._message = "Ledge reached!"

        # Spawn falling objects
        self._obj_timer += 1
        if self._obj_timer >= max(8 - self._level, 3):
            self._obj_timer = 0
            ox = int(self.rng.integers(2, _W - 2))
            self._add_entity(
                "falling", _FALLING_OBJ, ox, 1, dy=1
            )

        # Move falling objects
        for e in self._entities:
            if e.etype != "falling" or not e.alive:
                continue
            e.y += 1
            if e.y >= _H - 1:
                e.alive = False

        # Collision with falling objects (Pattern D death penalty)
        for e in self._entities:
            if (
                e.etype == "falling"
                and e.alive
                and e.x == self._player_x
                and e.y == self._player_y
            ):
                e.alive = False
                self._on_life_lost()
                reward = self._DEATH_PENALTY
                self._message = "Hit by falling object!"
                break

        # Window hazard: open windows push player
        vis_by = self._scroll_offset + self._player_y
        if 0 <= vis_by < self._bld_height:
            cell = self._building[vis_by][self._player_x]
            if cell == _WINDOW_CHAR and self.rng.random() < 0.1:
                self._message = "Window opened! Pushed!"
                self._player_y = min(
                    self._player_y + 2, _H - 3
                )

        # Check win condition (one building topped)
        if (
            self._scroll_offset <= 0
            and self._player_y <= 2
            and not self._game_over
        ):
            if self._progress_count < self._WIN_TARGET:
                reward += 1.0 / self._WIN_TARGET
                self._progress_count += 1
            self._level += 1
            self._message = "Building topped!"
            if self._progress_count >= self._WIN_TARGET:
                self._game_over = True
                info["won"] = True
                self._message = "All buildings climbed!"
            else:
                self._generate_level(self._level * 3001)

        self._entities = [
            e for e in self._entities if e.alive
        ]
        self._rebuild_grid()
        info["height"] = self._height_reached
        return reward, self._game_over, info

    def _advance_entities(self) -> None:
        pass

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "│": "building wall",
            "=": "ledge",
            "W": "window (hazard)",
            "X": "falling object",
            "─": "border",
            " ": "climbable surface",
        }.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        falling = sum(
            1 for e in self._entities
            if e.etype == "falling" and e.alive
        )
        extra = (
            f"Height: {self._height_reached}  "
            f"Danger: {falling}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Climb the building to the top. Move UP to climb, "
            "LEFT/RIGHT to dodge. Avoid falling objects (X) "
            "and open windows (W). Reach ledges for bonus."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Crazy Climber.\n\n"
            "TASK\n"
            "Climb the side of a tall building to the top while dodging "
            "falling objects and open windows. Advance levels by "
            "reaching the roof.\n\n"
            "BOARD\n"
            "12 columns by 24 rows (viewport). The world is a building "
            "of height 60 + 20*level. The viewport scrolls vertically "
            "so you stay roughly a third of the way from the top when "
            "climbing. The climbable face is bordered by side walls; "
            "ledges '=' appear at regular vertical intervals. Windows "
            "'W' are the hazard cells. Falling objects 'X' drop from "
            "the top. You appear as an arrow inside the climbable face.\n\n"
            "MECHANICS\n"
            "UP climbs one row up (and may scroll the viewport). DOWN "
            "climbs down; LEFT/RIGHT shift 1 column but must stay in "
            "the climbable zone (x in 2..W-3). Falling objects spawn "
            "every max(3, 8-level) steps from row 1 and fall 1 cell "
            "per step. Standing on a window cell has a 10 percent "
            "chance per step of pushing you 2 rows down.\n\n"
            "SCORING\n"
            "+0.2 reward per building topped (5 buildings = +1.0 "
            "cumulative). -1.0 penalty if hit by a falling object "
            "(single-life: ends the episode). No per-step penalty.\n\n"
            "TERMINATION\n"
            "Single-life model. Hit by a falling object ends the "
            "episode with -1.0 reward. Episode also ends after "
            "max_turns or after all 5 buildings are climbed.\n\n"
            "HUD\n"
            "Shows score, level, height reached, and number of "
            "active falling objects.\n\n"
            + self.action_spec.render_for_prompt()
        )
