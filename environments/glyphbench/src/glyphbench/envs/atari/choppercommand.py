"""Atari Chopper Command environment.

Horizontal helicopter shooter. Protect trucks from enemy helis.

Gym ID: glyphbench/atari-choppercommand-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase, AtariEntity


class ChopperCommandEnv(AtariBase):
    """Chopper Command: fly a helicopter and protect trucks.

    30x16 grid. Enemy helicopters attack ground trucks.
    Shoot enemies before they destroy your trucks.

    Actions: NOOP, LEFT, RIGHT, UP, DOWN, FIRE
    Pattern A: +1/_WIN_TARGET per enemy heli destroyed (full-scope =
    5 waves x 12 enemies = 60). No failure penalty.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "UP", "DOWN", "FIRE"),
        descriptions=(
            "do nothing",
            "move left",
            "move right",
            "move up",
            "move down",
            "fire a bullet",
        ),
    )

    _WIDTH = 30
    _HEIGHT = 16
    _GROUND_Y = 14
    _MAX_BULLETS = 3

    # Pattern A full-scope target: 60 enemy helis destroyed.
    _WIN_TARGET: int = 60

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._bullets: list[AtariEntity] = []
        self._enemy_bullets: list[AtariEntity] = []
        self._enemies: list[AtariEntity] = []
        self._trucks: list[AtariEntity] = []
        self._step_counter: int = 0
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-choppercommand-v0"

    def _reset(self, seed: int) -> GridObservation:
        self._progress_count = 0
        return super()._reset(seed)

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._bullets = []
        self._enemy_bullets = []
        self._enemies = []
        self._trucks = []
        self._step_counter = 0

        # Borders
        for x in range(self._WIDTH):
            self._set_cell(x, 0, "─")
            self._set_cell(x, self._HEIGHT - 1, "─")
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "│")
            self._set_cell(self._WIDTH - 1, y, "│")

        # Ground
        for x in range(1, self._WIDTH - 1):
            self._set_cell(x, self._GROUND_Y, "=")

        # Trucks on ground
        rng = self.rng
        num_trucks = 3
        for i in range(num_trucks):
            tx = 4 + i * 8
            if tx < self._WIDTH - 2:
                t = self._add_entity("truck", "T", tx, self._GROUND_Y - 1)
                self._trucks.append(t)

        # Enemy helicopters
        num_enemies = 3 + self._level
        for _ in range(min(num_enemies, 8)):
            side = int(rng.integers(0, 2))
            ex = 1 if side == 0 else self._WIDTH - 2
            ey = int(rng.integers(2, self._GROUND_Y - 3))
            dx = 1 if side == 0 else -1
            e = self._add_entity("enemy", "E", ex, ey, dx=dx)
            e.data["timer"] = 0
            e.data["fire_cd"] = int(rng.integers(10, 25))
            self._enemies.append(e)

        # Player starts mid-left
        self._player_x = 3
        self._player_y = self._GROUND_Y - 4

        self._redraw()

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._step_counter += 1

        # Move player
        if action_name == "LEFT" and self._player_x > 1:
            self._player_x -= 1
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT" and self._player_x < self._WIDTH - 2:
            self._player_x += 1
            self._player_dir = (1, 0)
        elif action_name == "UP" and self._player_y > 1:
            self._player_y -= 1
            self._player_dir = (0, -1)
        elif action_name == "DOWN" and self._player_y < self._GROUND_Y - 1:
            self._player_y += 1
            self._player_dir = (0, 1)

        # Fire
        if action_name == "FIRE" and len(self._bullets) < self._MAX_BULLETS:
            b = self._add_entity(
                "bullet", "·", self._player_x + 1, self._player_y, dx=1
            )
            self._bullets.append(b)

        # Move bullets
        for b in self._bullets:
            if not b.alive:
                continue
            b.x += b.dx
            if b.x >= self._WIDTH - 1 or b.x <= 0:
                b.alive = False
                continue
            # Check enemy hit
            for e in self._enemies:
                if e.alive and e.x == b.x and e.y == b.y:
                    e.alive = False
                    b.alive = False
                    self._on_point_scored(1)
                    if self._progress_count < self._WIN_TARGET:
                        reward += 1.0 / self._WIN_TARGET
                        self._progress_count += 1
                    self._message = "Enemy down!"
                    break
        self._bullets = [b for b in self._bullets if b.alive]

        # Move enemies
        for e in self._enemies:
            if not e.alive:
                continue
            e.data["timer"] += 1
            if e.data["timer"] % 2 == 0:
                e.x += e.dx
                if e.x <= 1 or e.x >= self._WIDTH - 2:
                    e.dx = -e.dx
                # Drift toward a truck
                if self._trucks:
                    target = self._trucks[
                        int(self.rng.integers(len(self._trucks)))
                    ]
                    if target.alive and e.data["timer"] % 6 == 0:
                        if e.y < target.y - 1:
                            e.y += 1
                        elif e.y > target.y - 1:
                            e.y -= 1

            # Enemy fire
            if e.data["timer"] % e.data["fire_cd"] == 0 and len(self._enemy_bullets) < 4:
                eb = self._add_entity(
                    "ebullet", "↓", e.x, e.y + 1, dy=1
                )
                self._enemy_bullets.append(eb)

        # Move enemy bullets
        for eb in self._enemy_bullets:
            if not eb.alive:
                continue
            eb.y += eb.dy
            if eb.y >= self._GROUND_Y:
                # Check truck hit
                for t in self._trucks:
                    if (
                        t.alive
                        and t.x == eb.x
                        and t.y == self._GROUND_Y - 1
                    ):
                        t.alive = False
                        self._message = "Truck destroyed!"
                eb.alive = False
                continue
            if eb.x == self._player_x and eb.y == self._player_y:
                eb.alive = False
                self._on_life_lost()
                self._message = "Hit! Lost a life."
                self._player_x = 3
                self._player_y = self._GROUND_Y - 4
        self._enemy_bullets = [
            eb for eb in self._enemy_bullets if eb.alive
        ]

        # Player collides with enemy
        for e in self._enemies:
            if e.alive and e.x == self._player_x and e.y == self._player_y:
                self._on_life_lost()
                self._message = "Collision! Lost a life."
                self._player_x = 3
                self._player_y = self._GROUND_Y - 4
                break

        # Win check
        if self._progress_count >= self._WIN_TARGET and not self._game_over:
            self._game_over = True
            info["won"] = True
            self._message = "All waves cleared!"

        # Level clear
        alive_enemies = [e for e in self._enemies if e.alive]
        if len(alive_enemies) == 0 and not self._game_over:
            self._message = "Wave cleared!"
            self._level += 1
            self._generate_level(self._level)

        self._redraw()
        return reward, self._game_over, info

    def _redraw(self) -> None:
        for y in range(1, self._HEIGHT - 1):
            for x in range(1, self._WIDTH - 1):
                if y == self._GROUND_Y:
                    self._set_cell(x, y, "=")
                else:
                    self._set_cell(x, y, " ")
        for t in self._trucks:
            if t.alive:
                self._set_cell(t.x, t.y, "T")
        for e in self._enemies:
            if e.alive:
                self._set_cell(e.x, e.y, "E")
        for b in self._bullets:
            if b.alive:
                self._set_cell(b.x, b.y, "·")
        for eb in self._enemy_bullets:
            if eb.alive:
                self._set_cell(eb.x, eb.y, "↓")

    def _advance_entities(self) -> None:
        self._entities = [e for e in self._entities if e.alive]

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "─": "wall",
            "│": "wall",
            "=": "ground",
            "T": "truck (protect)",
            "E": "enemy helicopter",
            "·": "your bullet",
            "↓": "enemy bullet",
            " ": "sky",
        }.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        trucks = sum(1 for t in self._trucks if t.alive)
        enemies = sum(
            1 for e in self._enemies if e.alive
        )
        extra = (
            f"Trucks: {trucks}  Enemies: {enemies}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Fly your helicopter and shoot enemy helicopters. "
            "Protect the trucks on the ground. "
            "Clear all enemies to advance to the next wave."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Chopper Command.\n\n"
            "TASK\n"
            "Pilot a helicopter and shoot down enemy helis while keeping "
            "ground trucks alive. Clear all enemies to finish the wave.\n\n"
            "BOARD\n"
            "30 columns by 16 rows, walled at edges. Ground '=' is at "
            "row 14. Trucks 'T' stand on the ground; enemy helicopters "
            "'E' fly above. Your bullet is '.', enemy bullet is "
            "down-arrow. You are an arrow glyph.\n\n"
            "MECHANICS\n"
            "LEFT/RIGHT/UP/DOWN move you 1 cell (blocked at walls and "
            "ground). FIRE launches a bullet in your last horizontal "
            "direction, starting 1 cell ahead (max 3 in flight). "
            "Enemies move every 2 steps, bouncing off the side walls "
            "and drifting toward the trucks' row every 6 steps. Each "
            "enemy fires downward every fire_cd steps (random 10-25); "
            "enemy bullets destroy a truck if they reach row 13 in the "
            "truck's column.\n\n"
            "SCORING\n"
            "+1 reward for each enemy helicopter destroyed. +3 "
            "reward when the wave is fully cleared (advances level and "
            "respawns 3+level new enemies). No reward or penalty for "
            "trucks directly.\n\n"
            "TERMINATION\n"
            "Single-life: colliding with an enemy or an enemy "
            "bullet ends the episode. The episode also ends "
            "after max_turns.\n\n"
            "HUD\n"
            "Shows score, wave/level, trucks alive, enemies "
            "alive.\n\n"
            + self.action_spec.render_for_prompt()
        )
