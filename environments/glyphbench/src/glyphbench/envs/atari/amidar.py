"""Atari Amidar environment.

Walk over segments of a grid to paint them. Avoid enemies.

Gym ID: glyphbench/atari-amidar-v0
"""

from __future__ import annotations

from typing import Any

import numpy as np

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase, AtariEntity

_DIRS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}

_W = 20
_H = 16
_UNPAINTED = "·"
_PAINTED = "="
_ENEMY_CHAR = "e"

class AmidarEnv(AtariBase):
    """Amidar: paint grid segments by walking over them.

    Actions: NOOP, UP, RIGHT, LEFT, DOWN
    Pattern D: +1/_WIN_TARGET per segment painted, -1 on enemy contact.
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP", "RIGHT", "LEFT", "DOWN"),
        descriptions=(
            "do nothing this step",
            "move up one cell",
            "move right one cell",
            "move left one cell",
            "move down one cell",
        ),
    )

    # Pattern D full-scope target: 100 unique segments painted across
    # any number of levels.
    _WIN_TARGET: int = 100
    _DEATH_PENALTY: float = -1.0

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._total_segments: int = 0
        self._painted_segments: int = 0
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-amidar-v0"

    def _reset(self, seed: int) -> GridObservation:
        self._progress_count = 0
        return super()._reset(seed)

    def _task_description(self) -> str:
        return (
            "Walk over unpainted segments (.) to paint them (=). "
            "Paint all segments to complete the level. "
            "Avoid enemies (e) that patrol the grid."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Amidar.\n\n"
            "TASK\n"
            "Walk over every unpainted segment of a grid to paint it. "
            "Finish painting the whole grid to clear the level.\n\n"
            "BOARD\n"
            "20x16 grid. Walls are '#'. Unpainted segments are shown as "
            "'.' and painted segments as '='. Enemies are 'e'. You are an "
            "arrow indicating your facing direction. The walkable layout is "
            "a rectangular rack: horizontal lines on every other row and "
            "vertical columns every 3 cells, all connected at "
            "intersections.\n\n"
            "MECHANICS\n"
            "Move one cell per step in any of the four cardinal directions; "
            "you can only enter painted or unpainted segment cells (not "
            "walls). Entering an unpainted cell paints it. Enemies patrol "
            "along the tracks, horizontal and vertical; at intersections "
            "they may turn (30 percent chance) or reverse when blocked.\n\n"
            "SCORING\n"
            "+1 reward per unpainted segment you paint. +5 reward when "
            "you paint the final segment and the level resets with more "
            "enemies. Colliding with an enemy gives -1 reward.\n\n"
            "TERMINATION\n"
            "Single-life: contact with an enemy ends the episode. "
            "The episode also ends after max_turns.\n\n"
            "HUD\n"
            "Shows score, level, and painted / total segments.\n\n"
            + self.action_spec.render_for_prompt()
        )

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "█": "wall",
            "·": "unpainted segment",
            "=": "painted segment",
            " ": "empty",
            "e": "enemy",
        }.get(ch, ch)

    def _generate_level(self, seed: int) -> None:
        rng = np.random.default_rng(seed + self._level * 1000)
        self._init_grid(_W, _H)
        self._entities = []
        self._painted_segments = 0

        # Build grid: border walls + grid of segments
        for y in range(_H):
            for x in range(_W):
                self._set_cell(x, y, "█")

        # Create a grid pattern of walkable segments
        # Horizontal lines
        h_rows = list(range(1, _H - 1, 2))
        for y in h_rows:
            for x in range(1, _W - 1):
                self._set_cell(x, y, _UNPAINTED)

        # Vertical lines connecting horizontals
        v_cols = list(range(1, _W - 1, 3))
        for x in v_cols:
            for y in range(1, _H - 1):
                if self._grid_at(x, y) == "█":
                    self._set_cell(x, y, _UNPAINTED)

        # Count total segments
        self._total_segments = 0
        for y in range(_H):
            for x in range(_W):
                if self._grid_at(x, y) == _UNPAINTED:
                    self._total_segments += 1

        # Place player at top-left segment
        self._player_x = 1
        self._player_y = 1
        # Paint starting cell
        self._set_cell(1, 1, _PAINTED)
        self._painted_segments = 1

        # Place enemies
        n_enemies = min(5, 2 + self._level)
        for i in range(n_enemies):
            for _attempt in range(30):
                if i % 2 == 0:
                    # Horizontal patrol enemy
                    ey = h_rows[int(rng.integers(0, len(h_rows)))]
                    ex = int(rng.integers(3, _W - 3))
                else:
                    # Vertical patrol enemy
                    ex = v_cols[int(rng.integers(0, len(v_cols)))]
                    ey = int(rng.integers(3, _H - 3))
                if (
                    self._grid_at(ex, ey) in (_UNPAINTED, _PAINTED)
                    and abs(ex - self._player_x) + abs(ey - self._player_y) > 4
                ):
                    enemy = self._add_entity("enemy", _ENEMY_CHAR, ex, ey)
                    if i % 2 == 0:
                        enemy.data["dir"] = (1, 0)
                        enemy.data["patrol"] = "horizontal"
                    else:
                        enemy.data["dir"] = (0, 1)
                        enemy.data["patrol"] = "vertical"
                    break

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        # Move player
        if action_name in _DIRS:
            dx, dy = _DIRS[action_name]
            self._player_dir = (dx, dy)
            nx = self._player_x + dx
            ny = self._player_y + dy
            cell = self._grid_at(nx, ny)
            if cell in (_UNPAINTED, _PAINTED):
                self._player_x, self._player_y = nx, ny

        # Paint current cell
        cell = self._grid_at(self._player_x, self._player_y)
        if cell == _UNPAINTED:
            self._set_cell(self._player_x, self._player_y, _PAINTED)
            self._painted_segments += 1
            self._on_point_scored(1)
            if self._progress_count < self._WIN_TARGET:
                reward += 1.0 / self._WIN_TARGET
                self._progress_count += 1

        # Move enemies
        for e in self._entities:
            if e.etype != "enemy" or not e.alive:
                continue
            self._move_enemy(e)

        # Check enemy collision (Pattern D death penalty)
        for e in self._entities:
            if e.etype == "enemy" and e.alive and e.x == self._player_x and e.y == self._player_y:
                self._on_life_lost()
                reward = self._DEATH_PENALTY
                self._message = "Caught by enemy!"
                break

        # Check level complete
        terminated = False
        if self._painted_segments >= self._total_segments and not self._game_over:
            if self._progress_count >= self._WIN_TARGET:
                # Full-scope win.
                self._game_over = True
                terminated = True
                info["won"] = True
                self._message = "All segments painted!"
            else:
                self._message = "Level complete!"
                self._level += 1
                self._generate_level(self._level * 6173)
                info["level_cleared"] = True

        info["painted"] = self._painted_segments
        info["total"] = self._total_segments
        info["progress"] = (
            self._painted_segments / max(1, self._total_segments)
        )
        return reward, terminated or self._game_over, info

    def _move_enemy(self, enemy: AtariEntity) -> None:
        """Enemy patrols along segments."""
        cur_dir = enemy.data.get("dir", (1, 0))
        dx, dy = cur_dir

        nx, ny = enemy.x + dx, enemy.y + dy
        cell = self._grid_at(nx, ny)
        if cell in (_UNPAINTED, _PAINTED):
            enemy.x, enemy.y = nx, ny
        else:
            # At intersection or wall: try to turn
            if enemy.data.get("patrol") == "horizontal":
                # Try vertical
                if self.rng.random() < 0.3:
                    for new_dy in [1, -1]:
                        tnx, tny = enemy.x, enemy.y + new_dy
                        if self._grid_at(tnx, tny) in (_UNPAINTED, _PAINTED):
                            enemy.data["dir"] = (0, new_dy)
                            enemy.data["patrol"] = "vertical"
                            enemy.x, enemy.y = tnx, tny
                            return
                # Reverse
                enemy.data["dir"] = (-dx, -dy)
            else:
                # Try horizontal
                if self.rng.random() < 0.3:
                    for new_dx in [1, -1]:
                        tnx, tny = enemy.x + new_dx, enemy.y
                        if self._grid_at(tnx, tny) in (_UNPAINTED, _PAINTED):
                            enemy.data["dir"] = (new_dx, 0)
                            enemy.data["patrol"] = "horizontal"
                            enemy.x, enemy.y = tnx, tny
                            return
                # Reverse
                enemy.data["dir"] = (-dx, -dy)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        extra = (
            f"Painted: {self._painted_segments}"
            f"/{self._total_segments}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _advance_entities(self) -> None:
        """Override: entities are moved in _game_step."""
        pass
