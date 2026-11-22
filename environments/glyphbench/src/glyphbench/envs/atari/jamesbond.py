"""Atari JamesBond environment.

Side-scrolling vehicle action. Dodge and destroy obstacles.

Gym ID: glyphbench/atari-jamesbond-v0
"""

from __future__ import annotations

from typing import Any

import numpy as np

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase

_W = 30
_H = 16
_GROUND_Y = 13
_DIRS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}

class JamesBondEnv(AtariBase):
    """JamesBond: side-scrolling vehicle action.

    Drive a vehicle, shoot enemies (E) and obstacles (X).
    Collect intel pickups (i) for bonus points.
    The world scrolls left; dodge or destroy threats.

    Grid: 30x16.
    Pattern D: +1/_WIN_TARGET per stage cleared (full-scope = 4
    environments). -1.0 on death.
    """

    action_spec = ActionSpec(
        names=(
            "NOOP", "LEFT", "RIGHT", "UP", "DOWN", "FIRE",
        ),
        descriptions=(
            "do nothing",
            "move left",
            "move right",
            "move up",
            "move down",
            "fire weapon forward",
        ),
    )

    # Pattern D full-scope target: 4 mission stages.
    _WIN_TARGET: int = 4
    _DEATH_PENALTY: float = -1.0

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._scroll_timer: int = 0
        self._spawn_timer: int = 0
        self._distance: int = 0
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-jamesbond-v0"

    def _reset(self, seed: int):
        self._progress_count = 0
        return super()._reset(seed)

    def _generate_level(self, seed: int) -> None:
        rng = np.random.default_rng(
            seed + self._level * 1117
        )
        self._init_grid(_W, _H)
        self._entities = []
        self._scroll_timer = 0
        self._spawn_timer = 0
        self._distance = 0

        # Top/bottom borders
        for x in range(_W):
            self._set_cell(x, 0, "─")
            self._set_cell(x, _H - 1, "─")

        # Ground
        for x in range(_W):
            self._set_cell(x, _GROUND_Y, "=")
            self._set_cell(x, _GROUND_Y + 1, "~")

        # Road markings
        for x in range(0, _W, 4):
            self._set_cell(x, _GROUND_Y - 3, "·")

        # Initial obstacles and enemies
        for _ in range(3 + self._level):
            ox = int(rng.integers(15, _W - 2))
            oy = int(rng.integers(3, _GROUND_Y - 1))
            if rng.random() < 0.4:
                self._add_entity("enemy", "E", ox, oy, dx=-1)
            else:
                self._add_entity(
                    "obstacle", "X", ox, oy, dx=-1
                )

        # Intel pickups
        for _ in range(2):
            ix = int(rng.integers(15, _W - 2))
            iy = int(rng.integers(3, _GROUND_Y - 1))
            self._add_entity("intel", "i", ix, iy, dx=-1)

        # Player position
        self._player_x = 3
        self._player_y = _GROUND_Y - 2

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        fire = action_name == "FIRE"

        # Move player
        if action_name in _DIRS:
            d = _DIRS[action_name]
            self._player_dir = d
            nx = self._player_x + d[0]
            ny = self._player_y + d[1]
            if 1 <= nx <= _W // 2 and 2 <= ny < _GROUND_Y:
                self._player_x = nx
                self._player_y = ny

        # Fire bullet
        if fire:
            self._add_entity(
                "bullet", "→",
                self._player_x + 1, self._player_y,
                dx=2,
            )

        # Move bullets
        for e in self._entities:
            if e.etype == "bullet" and e.alive:
                e.x += e.dx
                if e.x >= _W - 1:
                    e.alive = False

        # Scroll world: move enemies/obstacles left
        self._scroll_timer += 1
        if self._scroll_timer >= 2:
            self._scroll_timer = 0
            self._distance += 1
            for e in self._entities:
                if e.etype in ("enemy", "obstacle", "intel") and e.alive:
                    e.x += e.dx
                    if e.x < 1:
                        e.alive = False

        # Bullet-enemy/obstacle collisions (no direct reward)
        for b in self._entities:
            if b.etype != "bullet" or not b.alive:
                continue
            for t in self._entities:
                if not t.alive:
                    continue
                if t.etype == "enemy" and b.x == t.x and b.y == t.y:
                    b.alive = False
                    t.alive = False
                    self._message = "Enemy destroyed!"
                elif (
                    t.etype == "obstacle"
                    and b.x == t.x
                    and b.y == t.y
                ):
                    b.alive = False
                    t.alive = False
                    self._message = "Obstacle destroyed!"

        # Player pickups (no direct reward)
        for e in self._entities:
            if (
                e.etype == "intel"
                and e.alive
                and e.x == self._player_x
                and e.y == self._player_y
            ):
                e.alive = False
                self._message = "Intel collected!"

        # Player-enemy/obstacle collision (Pattern D death penalty)
        for e in self._entities:
            if not e.alive:
                continue
            if e.etype in ("enemy", "obstacle") and (
                e.x == self._player_x
                and e.y == self._player_y
            ):
                self._on_life_lost()
                reward = self._DEATH_PENALTY
                self._message = f"Hit by {e.etype}!"
                return reward, self._game_over, info

        self._entities = [
            e for e in self._entities if e.alive
        ]

        # Spawn new threats
        self._spawn_timer += 1
        rate = max(8 - self._level, 3)
        if self._spawn_timer >= rate:
            self._spawn_timer = 0
            oy = int(
                self.rng.integers(3, _GROUND_Y - 1)
            )
            r = self.rng.random()
            if r < 0.3:
                self._add_entity(
                    "enemy", "E", _W - 2, oy, dx=-1,
                )
            elif r < 0.7:
                self._add_entity(
                    "obstacle", "X", _W - 2, oy, dx=-1,
                )
            else:
                self._add_entity(
                    "intel", "i", _W - 2, oy, dx=-1,
                )

        # Level progression based on distance (Pattern D progress)
        if self._distance >= 80 + self._level * 20:
            self._level += 1
            if self._progress_count < self._WIN_TARGET:
                reward += 1.0 / self._WIN_TARGET
                self._progress_count += 1
            self._message = "Mission stage complete!"
            if self._progress_count >= self._WIN_TARGET:
                self._game_over = True
                info["won"] = True
                self._message = "All stages complete!"
            else:
                self._generate_level(self._level * 2999)

        # Redraw ground decorations
        for x in range(0, _W, 4):
            off = (x + self._distance) % 4
            self._set_cell(
                (x - off) % _W, _GROUND_Y - 3, "·"
            )

        info["distance"] = self._distance
        return reward, self._game_over, info

    def _advance_entities(self) -> None:
        pass

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "─": "border",
            "=": "ground",
            "~": "water",
            "·": "road marking",
            "E": "enemy vehicle",
            "X": "obstacle",
            "i": "intel pickup",
            "→": "bullet",
            " ": "sky",
        }.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        extra = f"Distance: {self._distance}"
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Drive your vehicle and survive. FIRE to shoot "
            "enemies (E) and obstacles (X). Collect intel (i) "
            "for bonus points. Use UP/DOWN/LEFT/RIGHT to dodge."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari James Bond.\n\n"
            "TASK\n"
            "Pilot a side-scrolling spy vehicle: destroy enemies, "
            "blast obstacles, and grab intel while the world scrolls "
            "leftward toward you. Travel distance to advance stages.\n\n"
            "BOARD\n"
            "30 columns by 16 rows. Sky fills the upper rows; ground "
            "'=' lies near the bottom with water 'tilde' below it. "
            "Road marks '.' scroll at row 10. Enemies 'E', obstacles "
            "'X', intel pickups 'i', your bullets show as right-"
            "arrows. You are an arrow glyph confined to the left "
            "half of the viewport.\n\n"
            "MECHANICS\n"
            "Moves shift you 1 cell; you cannot leave x=1..15 or "
            "rows 2..12. FIRE launches a bullet traveling right with "
            "speed 2; no stated cap on bullets. The world scrolls "
            "leftward (enemies/obstacles/intel move -1 cell every 2 "
            "steps). New threats spawn at x=W-2 every max(3, 8-level) "
            "steps (30 percent enemy, 40 percent obstacle, 30 percent "
            "intel).\n\n"
            "SCORING\n"
            "+1/4 reward per mission stage cleared (Pattern D "
            "full-scope = 4 environments). Destroying enemies / "
            "obstacles and collecting intel yield no direct "
            "reward, only progress toward stage completion. -1.0 "
            "on death (collision with enemy or obstacle).\n\n"
            "TERMINATION\n"
            "Colliding with enemy or obstacle ends the episode "
            "with -1.0. Traveling 80 + 20*level distance advances "
            "to the next stage. Episode ends after 4 stages cleared "
            "(cumulative reward plateaus at +1.0) or after max_turns.\n\n"
            "HUD\n"
            "Shows score, level, and distance traveled.\n\n"
            + self.action_spec.render_for_prompt()
        )
