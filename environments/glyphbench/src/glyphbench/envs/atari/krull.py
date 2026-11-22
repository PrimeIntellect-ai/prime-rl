"""Atari Krull environment.

Multi-stage adventure. Fight enemies, rescue the princess.

Gym ID: glyphbench/atari-krull-v0
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

class KrullEnv(AtariBase):
    """Krull: multi-stage adventure.

    Stage 1: Fight Slayers (S) in an open field.
    Stage 2: Navigate the swamp, avoid hazards.
    Stage 3: Storm the Black Fortress, rescue princess (P).

    Grid: 20x16.
    Pattern D: +1/_WIN_TARGET per stage cleared (full-scope = 3
    stages: field, swamp, fortress). -1.0 on death.
    """

    action_spec = ActionSpec(
        names=(
            "NOOP", "UP", "DOWN", "LEFT", "RIGHT", "FIRE",
        ),
        descriptions=(
            "do nothing",
            "move up",
            "move down",
            "move left",
            "move right",
            "throw glaive at nearest enemy",
        ),
    )

    # Pattern D full-scope target: 3 stages cleared (the env has
    # 3 distinct stages: field, swamp, fortress).
    _WIN_TARGET: int = 3
    _DEATH_PENALTY: float = -1.0

    def __init__(self, max_turns: int = 1000) -> None:
        super().__init__(max_turns=max_turns)
        self._stage: int = 1
        self._glaive_cooldown: int = 0
        self._enemies_to_clear: int = 0
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-krull-v0"

    def _reset(self, seed: int):
        self._progress_count = 0
        return super()._reset(seed)

    def _generate_level(self, seed: int) -> None:
        rng = np.random.default_rng(
            seed + self._level * 877 + self._stage * 100
        )
        self._init_grid(_W, _H)
        self._entities = []
        self._glaive_cooldown = 0
        self._enemies_to_clear = 0

        # Border
        for x in range(_W):
            self._set_cell(x, 0, "█")
            self._set_cell(x, _H - 1, "█")
        for y in range(_H):
            self._set_cell(0, y, "█")
            self._set_cell(_W - 1, y, "█")

        if self._stage == 1:
            self._gen_stage_field(rng)
        elif self._stage == 2:
            self._gen_stage_swamp(rng)
        else:
            self._gen_stage_fortress(rng)

    def _gen_stage_field(self, rng: np.random.Generator) -> None:
        """Open field battle against Slayers."""
        # Scatter some rocks
        for _ in range(6):
            rx = int(rng.integers(3, _W - 3))
            ry = int(rng.integers(3, _H - 3))
            self._set_cell(rx, ry, "·")

        # Spawn slayers
        n_slayers = 4 + self._level
        self._enemies_to_clear = n_slayers
        for _ in range(n_slayers):
            for _att in range(20):
                sx = int(rng.integers(10, _W - 2))
                sy = int(rng.integers(2, _H - 2))
                if self._grid_at(sx, sy) == " ":
                    e = self._add_entity(
                        "slayer", "S", sx, sy,
                    )
                    e.data["timer"] = int(
                        rng.integers(3, 7)
                    )
                    break

        self._player_x = 2
        self._player_y = _H // 2

    def _gen_stage_swamp(self, rng: np.random.Generator) -> None:
        """Swamp navigation with hazards."""
        # Fill with swamp tiles
        for y in range(1, _H - 1):
            for x in range(1, _W - 1):
                if rng.random() < 0.2:
                    self._set_cell(x, y, "~")

        # Path through swamp
        py = _H // 2
        for x in range(1, _W - 1):
            self._set_cell(x, py, " ")
            if rng.random() < 0.3:
                py += int(rng.integers(-1, 2))
                py = max(2, min(py, _H - 3))
                self._set_cell(x, py, " ")

        # Swamp creatures
        n_creatures = 3 + self._level
        self._enemies_to_clear = n_creatures
        for _ in range(n_creatures):
            for _att in range(20):
                cx = int(rng.integers(5, _W - 2))
                cy = int(rng.integers(2, _H - 2))
                if self._grid_at(cx, cy) == " ":
                    e = self._add_entity(
                        "creature", "C", cx, cy,
                    )
                    e.data["timer"] = int(
                        rng.integers(3, 6)
                    )
                    break

        # Exit
        self._add_entity("exit", "D", _W - 2, py)
        self._player_x = 2
        self._player_y = py

    def _gen_stage_fortress(
        self, rng: np.random.Generator,
    ) -> None:
        """Black Fortress: rooms with guards and princess."""
        # Inner walls creating rooms
        mid_x = _W // 2
        for y in range(1, _H - 1):
            self._set_cell(mid_x, y, "█")
        # Doorway
        self._set_cell(mid_x, _H // 2, " ")
        self._set_cell(mid_x, _H // 2 - 1, " ")

        # Guards
        n_guards = 3 + self._level
        self._enemies_to_clear = n_guards
        for _ in range(n_guards):
            for _att in range(20):
                gx = int(rng.integers(2, _W - 2))
                gy = int(rng.integers(2, _H - 2))
                if self._grid_at(gx, gy) == " ":
                    e = self._add_entity(
                        "guard", "G", gx, gy,
                    )
                    e.data["timer"] = int(
                        rng.integers(2, 5)
                    )
                    break

        # Princess in right room
        self._add_entity(
            "princess", "P", _W - 3, _H // 2,
        )
        self._player_x = 2
        self._player_y = _H // 2

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        fire = action_name == "FIRE"

        # Movement
        if action_name in _DIRS:
            d = _DIRS[action_name]
            self._player_dir = d
            nx = self._player_x + d[0]
            ny = self._player_y + d[1]
            if not self._is_solid(nx, ny):
                cell = self._grid_at(nx, ny)
                # Swamp slows and may hurt (Pattern D death)
                if cell == "~" and self.rng.random() < 0.15:
                    self._on_life_lost()
                    reward = self._DEATH_PENALTY
                    self._message = "Sank in swamp!"
                    return (
                        reward, self._game_over, info
                    )
                self._player_x = nx
                self._player_y = ny

        # Fire glaive
        if self._glaive_cooldown > 0:
            self._glaive_cooldown -= 1
        if fire and self._glaive_cooldown <= 0:
            self._glaive_cooldown = 3
            # Find nearest enemy and launch glaive
            nearest = None
            best_dist = 999
            etypes = ("slayer", "creature", "guard")
            for e in self._entities:
                if e.etype in etypes and e.alive:
                    d2 = abs(e.x - self._player_x) + abs(
                        e.y - self._player_y
                    )
                    if d2 < best_dist:
                        best_dist = d2
                        nearest = e
            if nearest and best_dist <= 8:
                dx = (
                    1 if nearest.x > self._player_x
                    else (
                        -1
                        if nearest.x < self._player_x
                        else 0
                    )
                )
                dy = (
                    1 if nearest.y > self._player_y
                    else (
                        -1
                        if nearest.y < self._player_y
                        else 0
                    )
                )
                bx = self._player_x + dx
                by = self._player_y + dy
                if not self._is_solid(bx, by):
                    b = self._add_entity(
                        "glaive", "※", bx, by,
                        dx=dx, dy=dy,
                    )
                    b.data["ttl"] = 6

        # Move glaives
        for e in self._entities:
            if e.etype != "glaive" or not e.alive:
                continue
            e.x += e.dx
            e.y += e.dy
            e.data["ttl"] = e.data.get("ttl", 6) - 1
            if (
                e.data["ttl"] <= 0
                or self._is_solid(e.x, e.y)
            ):
                e.alive = False

        # Enemy AI
        etypes = ("slayer", "creature", "guard")
        for e in self._entities:
            if e.etype not in etypes or not e.alive:
                continue
            e.data["timer"] = (
                e.data.get("timer", 3) - 1
            )
            if e.data["timer"] > 0:
                continue
            e.data["timer"] = int(
                self.rng.integers(2, 5)
            )
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

        # Glaive-enemy collisions (no direct reward)
        for g in self._entities:
            if g.etype != "glaive" or not g.alive:
                continue
            for en in self._entities:
                if en.etype not in etypes:
                    continue
                if not en.alive:
                    continue
                if g.x == en.x and g.y == en.y:
                    g.alive = False
                    en.alive = False
                    self._enemies_to_clear -= 1
                    self._message = (
                        f"{en.etype} destroyed!"
                    )

        # Player-enemy collision (Pattern D death penalty)
        for e in self._entities:
            if e.etype not in etypes or not e.alive:
                continue
            if (
                e.x == self._player_x
                and e.y == self._player_y
            ):
                self._on_life_lost()
                reward = self._DEATH_PENALTY
                self._message = f"Hit by {e.etype}!"
                return reward, self._game_over, info

        # Princess rescue (stage 3 / final stage progress)
        for e in self._entities:
            if (
                e.etype == "princess"
                and e.alive
                and e.x == self._player_x
                and e.y == self._player_y
            ):
                e.alive = False
                if self._progress_count < self._WIN_TARGET:
                    reward += 1.0 / self._WIN_TARGET
                    self._progress_count += 1
                self._message = "Princess rescued!"
                if self._progress_count >= self._WIN_TARGET:
                    self._game_over = True
                    info["won"] = True
                    self._message = "All stages complete!"
                else:
                    self._level += 1
                    self._stage = 1
                    self._generate_level(self._level * 6007)
                return reward, self._game_over, info

        # Exit (stage 2)
        for e in self._entities:
            if (
                e.etype == "exit"
                and e.alive
                and e.x == self._player_x
                and e.y == self._player_y
            ):
                enemies_alive = sum(
                    1 for en in self._entities
                    if en.etype in etypes and en.alive
                )
                if enemies_alive == 0:
                    e.alive = False
                    if self._progress_count < self._WIN_TARGET:
                        reward += 1.0 / self._WIN_TARGET
                        self._progress_count += 1
                    self._stage += 1
                    self._message = "Stage complete!"
                    if self._progress_count >= self._WIN_TARGET:
                        self._game_over = True
                        info["won"] = True
                        self._message = "All stages complete!"
                    else:
                        self._generate_level(
                            self._level * 6007
                            + self._stage
                        )
                    return reward, self._game_over, info
                else:
                    self._message = (
                        "Clear all enemies first!"
                    )

        # Stage 1 clear check
        if self._stage == 1:
            enemies_alive = sum(
                1 for e in self._entities
                if e.etype in etypes and e.alive
            )
            if enemies_alive == 0:
                if self._progress_count < self._WIN_TARGET:
                    reward += 1.0 / self._WIN_TARGET
                    self._progress_count += 1
                self._stage = 2
                self._message = "Field cleared!"
                if self._progress_count >= self._WIN_TARGET:
                    self._game_over = True
                    info["won"] = True
                    self._message = "All stages complete!"
                else:
                    self._generate_level(
                        self._level * 6007 + self._stage
                    )
                return reward, self._game_over, info

        self._entities = [
            e for e in self._entities if e.alive
        ]
        info["stage"] = self._stage
        return reward, self._game_over, info

    def _advance_entities(self) -> None:
        pass

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "█": "wall",
            " ": "open ground",
            "·": "rock",
            "~": "swamp (dangerous)",
            "S": "Slayer enemy",
            "C": "swamp creature",
            "G": "fortress guard",
            "P": "princess",
            "D": "exit door",
            "※": "glaive (projectile)",
        }.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        etypes = ("slayer", "creature", "guard")
        enemies = sum(
            1 for e in self._entities
            if e.etype in etypes and e.alive
        )
        extra = (
            f"Stage: {self._stage}/3  "
            f"Enemies: {enemies}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Adventure through 3 stages: fight Slayers in "
            "the field, navigate the swamp, and storm the "
            "Black Fortress to rescue the princess (P). "
            "FIRE throws the glaive at the nearest enemy."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Krull.\n\n"
            "TASK\n"
            "Complete a 3-stage adventure: (1) clear an open field "
            "of Slayers, (2) cross a swamp and exit through door "
            "'D' after clearing creatures, (3) storm the fortress "
            "and rescue the princess 'P'.\n\n"
            "BOARD\n"
            "20x16 on every stage with wall '#' borders. Stage 1: "
            "scattered rocks '.', Slayers 'S'. Stage 2: swamp "
            "'tilde' tiles (dangerous), a clear path, creatures 'C', "
            "an exit door 'D'. Stage 3: a wall divides the room; "
            "guards 'G' patrol and the princess 'P' is in one of the "
            "chambers. Your glaive is '※' (reference mark), you are "
            "an arrow glyph.\n\n"
            "MECHANICS\n"
            "UP/DOWN/LEFT/RIGHT move you 1 cell (blocked by walls). "
            "Stepping onto swamp tilde has a 15 percent chance per "
            "step of ending the episode (single-life). FIRE throws "
            "a glaive at the nearest enemy within Manhattan "
            "distance 8 (cooldown 3 steps, glaive TTL 6 steps, "
            "moves 1 cell per step). Enemies move on a 2-5 step "
            "timer toward you along one axis.\n\n"
            "SCORING\n"
            "+1/3 reward per stage cleared (Pattern D full-scope = "
            "3 stages: field, swamp, fortress). Killing enemies "
            "yields no direct reward (only progress toward stage "
            "completion). -1.0 on death (enemy contact or sinking "
            "in swamp).\n\n"
            "TERMINATION\n"
            "Enemy contact or sinking in swamp ends the episode "
            "with -1.0. Episode ends after 3 stages cleared "
            "(cumulative reward plateaus at +1.0) or after "
            "max_turns.\n\n"
            "HUD\n"
            "Shows score, level, current stage (1-3), and "
            "enemies remaining.\n\n"
            + self.action_spec.render_for_prompt()
        )
