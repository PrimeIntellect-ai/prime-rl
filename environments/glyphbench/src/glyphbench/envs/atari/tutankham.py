"""Atari Tutankham environment.

Pyramid maze shooter. Collect treasures, fight enemies.

Gym ID: glyphbench/atari-tutankham-v0
"""

from __future__ import annotations

from typing import Any

import numpy as np

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase

_W = 20
_H = 16
_DIRS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}

class TutankhamEnv(AtariBase):
    """Tutankham: pyramid maze shooter.

    Navigate maze rooms, collect treasures ($), shoot snakes (S)
    and mummies (M). Find the key (K) to unlock the exit (D).

    Grid: 20x16.
    Pattern D: +1/_WIN_TARGET per progress unit (treasure
    collected or exit cleared). -1 if a snake or mummy
    catches you.
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
            "shoot in last-faced direction",
        ),
    )

    # Pattern D full-scope: 8 progress events (each room's
    # treasure plus an exit clearance). Each yields +1/8.
    _WIN_TARGET: int = 8
    _DEATH_PENALTY: float = -1.0

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._facing: tuple[int, int] = (1, 0)
        self._has_key: bool = False
        self._treasures_left: int = 0
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-tutankham-v0"

    def _reset(self, seed: int):
        self._progress_count = 0
        return super()._reset(seed)

    def _generate_level(self, seed: int) -> None:
        rng = np.random.default_rng(seed + self._level * 773)
        self._init_grid(_W, _H)
        self._entities = []
        self._facing = (1, 0)
        self._has_key = False
        self._treasures_left = 0

        # Border walls
        for x in range(_W):
            self._set_cell(x, 0, "█")
            self._set_cell(x, _H - 1, "█")
        for y in range(_H):
            self._set_cell(0, y, "█")
            self._set_cell(_W - 1, y, "█")

        # Fill interior with walls, then carve rooms
        for y in range(1, _H - 1):
            for x in range(1, _W - 1):
                self._set_cell(x, y, "█")

        # Carve 3-4 rooms connected by corridors
        rooms: list[tuple[int, int, int, int]] = []
        room_defs = [
            (2, 2, 7, 5),
            (10, 2, 7, 5),
            (2, 9, 7, 5),
            (10, 9, 7, 5),
        ]
        for rx, ry, rw, rh in room_defs:
            if rng.random() < 0.85 or len(rooms) < 2:
                rooms.append((rx, ry, rw, rh))
                for yy in range(ry, min(ry + rh, _H - 1)):
                    for xx in range(
                        rx, min(rx + rw, _W - 1)
                    ):
                        self._set_cell(xx, yy, " ")

        # Corridors between rooms
        for i in range(len(rooms) - 1):
            r1 = rooms[i]
            r2 = rooms[i + 1]
            cx1 = r1[0] + r1[2] // 2
            cy1 = r1[1] + r1[3] // 2
            cx2 = r2[0] + r2[2] // 2
            cy2 = r2[1] + r2[3] // 2
            x = cx1
            while x != cx2:
                self._set_cell(x, cy1, " ")
                x += 1 if cx2 > cx1 else -1
            self._set_cell(cx2, cy1, " ")
            y = cy1
            while y != cy2:
                self._set_cell(cx2, y, " ")
                y += 1 if cy2 > cy1 else -1

        # Place treasures in rooms
        for rx, ry, rw, rh in rooms:
            n_t = int(rng.integers(1, 3))
            for _ in range(n_t):
                tx = int(rng.integers(rx + 1, rx + rw - 1))
                ty = int(rng.integers(ry + 1, ry + rh - 1))
                if self._grid_at(tx, ty) == " ":
                    self._add_entity("treasure", "$", tx, ty)
                    self._treasures_left += 1

        # Place key in a random room
        kr = rooms[int(rng.integers(1, len(rooms)))]
        kx = kr[0] + kr[2] // 2
        ky = kr[1] + kr[3] // 2
        if self._grid_at(kx, ky) == " ":
            self._add_entity("key", "K", kx, ky)

        # Place exit door
        er = rooms[-1]
        ex = er[0] + er[2] - 1
        ey = er[1] + 1
        self._set_cell(ex, ey, " ")
        self._add_entity("door", "D", ex, ey)

        # Place enemies
        n_enemies = min(3 + self._level, 8)
        for _ in range(n_enemies):
            for _att in range(30):
                rm = rooms[
                    int(rng.integers(0, len(rooms)))
                ]
                ex2 = int(
                    rng.integers(rm[0] + 1, rm[0] + rm[2] - 1)
                )
                ey2 = int(
                    rng.integers(rm[1] + 1, rm[1] + rm[3] - 1)
                )
                if self._grid_at(ex2, ey2) == " ":
                    etype = "snake" if rng.random() < 0.5 else "mummy"
                    ch = "S" if etype == "snake" else "M"
                    e = self._add_entity(etype, ch, ex2, ey2)
                    e.data["timer"] = int(rng.integers(2, 6))
                    break

        # Player start
        r0 = rooms[0]
        self._player_x = r0[0] + 1
        self._player_y = r0[1] + 1

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        fire = action_name == "FIRE"

        if action_name in _DIRS:
            d = _DIRS[action_name]
            self._facing = d
            self._player_dir = d
            nx = self._player_x + d[0]
            ny = self._player_y + d[1]
            if not self._is_solid(nx, ny):
                self._player_x = nx
                self._player_y = ny

        # Fire bullet
        if fire:
            bx = self._player_x + self._facing[0]
            by = self._player_y + self._facing[1]
            if not self._is_solid(bx, by):
                b = self._add_entity("bullet", "!", bx, by)
                b.dx = self._facing[0]
                b.dy = self._facing[1]

        # Move bullets
        for e in self._entities:
            if e.etype != "bullet" or not e.alive:
                continue
            e.x += e.dx
            e.y += e.dy
            if self._is_solid(e.x, e.y):
                e.alive = False

        # Enemy AI
        for e in self._entities:
            if e.etype not in ("snake", "mummy"):
                continue
            if not e.alive:
                continue
            e.data["timer"] = e.data.get("timer", 3) - 1
            if e.data["timer"] > 0:
                continue
            e.data["timer"] = int(self.rng.integers(2, 5))
            dx = (
                1 if self._player_x > e.x
                else (-1 if self._player_x < e.x else 0)
            )
            dy = (
                1 if self._player_y > e.y
                else (-1 if self._player_y < e.y else 0)
            )
            if self.rng.random() < 0.5:
                dy = 0
            else:
                dx = 0
            nx2, ny2 = e.x + dx, e.y + dy
            if not self._is_solid(nx2, ny2):
                e.x, e.y = nx2, ny2

        # Bullet-enemy collisions (no extra reward; just clearing)
        for b in self._entities:
            if b.etype != "bullet" or not b.alive:
                continue
            for en in self._entities:
                if en.etype not in ("snake", "mummy"):
                    continue
                if not en.alive:
                    continue
                if b.x == en.x and b.y == en.y:
                    b.alive = False
                    en.alive = False
                    self._message = f"{en.etype} destroyed!"

        # Player pickups
        for e in self._entities:
            if not e.alive:
                continue
            if e.x != self._player_x or e.y != self._player_y:
                continue
            if e.etype == "treasure":
                e.alive = False
                self._on_point_scored(1)
                if self._progress_count < self._WIN_TARGET:
                    reward += 1.0 / self._WIN_TARGET
                    self._progress_count += 1
                self._treasures_left -= 1
                self._message = "Treasure!"
            elif e.etype == "key":
                e.alive = False
                self._has_key = True
                self._message = "Key acquired!"
            elif e.etype == "door":
                if self._has_key:
                    self._level += 1
                    self._on_point_scored(1)
                    if self._progress_count < self._WIN_TARGET:
                        reward += 1.0 / self._WIN_TARGET
                        self._progress_count += 1
                    self._message = "Level cleared!"
                    saved_progress = self._progress_count
                    self._generate_level(self._level * 3571)
                    self._progress_count = saved_progress
                    if self._progress_count >= self._WIN_TARGET:
                        self._game_over = True
                        info["won"] = True
                    return reward, self._game_over, info
                else:
                    self._message = "Door locked. Find the key (K)!"

        # Player-enemy collision -- terminal failure
        for e in self._entities:
            if e.etype not in ("snake", "mummy"):
                continue
            if (
                e.alive
                and e.x == self._player_x
                and e.y == self._player_y
            ):
                self._on_life_lost()
                self._message = f"Hit by {e.etype}! Game Over."
                reward = self._DEATH_PENALTY
                return reward, self._game_over, info

        # Win check
        if (
            self._progress_count >= self._WIN_TARGET
            and not self._game_over
        ):
            self._game_over = True
            info["won"] = True
            self._message = "Pyramid cleared!"

        self._entities = [
            e for e in self._entities if e.alive
        ]
        info["has_key"] = self._has_key
        info["treasures_left"] = self._treasures_left
        return reward, self._game_over, info

    def _advance_entities(self) -> None:
        pass

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "█": "wall",
            " ": "floor",
            "$": "treasure",
            "K": "key",
            "D": "exit door",
            "S": "snake enemy",
            "M": "mummy enemy",
            "!": "bullet",
        }.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        key = "yes" if self._has_key else "no"
        extra = (
            f"Key: {key}  "
            f"Treasures: {self._treasures_left}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Explore the pyramid maze. Collect treasures ($), "
            "find the key (K) to unlock the exit (D). "
            "Shoot snakes (S) and mummies (M) with FIRE."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Tutankham.\n\n"
            "TASK\n"
            "Explore a multi-room pyramid, grab treasures, find "
            "the key, and exit through the door to advance the "
            "level. Fight snakes and mummies along the way.\n\n"
            "BOARD\n"
            "20x16 maze. Walls '#', open floor ' '. Treasures '$', "
            "key 'K', exit door 'D'. Snakes 'S' and mummies 'M' "
            "roam the rooms. Your bullets '!'. You are an arrow "
            "glyph.\n\n"
            "MECHANICS\n"
            "UP / DOWN / LEFT / RIGHT move 1 cell (blocked by "
            "walls) and also set facing direction. FIRE launches "
            "a bullet in facing direction until it hits a wall or "
            "an enemy. Enemies move on a 2-5 step timer toward "
            "you along one axis. Stepping onto the 'K' picks up "
            "the key. Stepping onto 'D' exits only if you have the "
            "key.\n\n"
            "SCORING\n"
            "Pattern D: +1/8 reward per treasure collected and "
            "per door exited with the key. Killing enemies emits "
            "no reward (only clears the path). -1 reward on "
            "contact with a snake or mummy. Cumulative reward "
            "bound: [-1, +1].\n\n"
            "TERMINATION\n"
            "Single-life: contact with an enemy ends the episode "
            "with -1. Reaching 8 progress events ends with "
            "cumulative +1. Episode also ends after max_turns.\n\n"
            "HUD\n"
            "Shows score, level, facing, key status (yes/"
            "no), and treasures remaining.\n\n"
            + self.action_spec.render_for_prompt()
        )
