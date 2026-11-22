"""Atari Gravitar environment.

Space shooter with gravity and fuel management.

Gym ID: glyphbench/atari-gravitar-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase, AtariEntity


def _sign(a: int, b: int) -> int:
    return 1 if a > b else (-1 if a < b else 0)

class GravitarEnv(AtariBase):
    """Gravitar: space shooter with gravity.

    20x20 grid. Navigate planets with gravity, shoot enemies,
    collect fuel. Multiple planets to clear.

    Actions: NOOP, LEFT, RIGHT, THRUST, FIRE
    Pattern A: +1/_WIN_TARGET per enemy/fuel pickup combined to a
    single progress unit. Full-scope = 12 (4 planets x 3 fights).

    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "THRUST", "FIRE"),
        descriptions=(
            "do nothing", "rotate left", "rotate right",
            "thrust against gravity",
            "fire in facing direction",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 20
    _MAX_FUEL = 120
    _DIRS = (
        (0, -1), (1, -1), (1, 0), (1, 1),
        (0, 1), (-1, 1), (-1, 0), (-1, -1),
    )

    # Pattern A full-scope target: 12 (4 planets x 3 enemies).
    _WIN_TARGET: int = 12
    # Audit BROKEN fix: any death event (terrain crash, enemy
    # bullet, fuel-out) is now terminal with reward -1.
    _DEATH_PENALTY: float = -1.0

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._facing = 0
        self._fuel = self._MAX_FUEL
        self._enemies: list[AtariEntity] = []
        self._bullets: list[AtariEntity] = []
        self._enemy_bullets: list[AtariEntity] = []
        self._fuel_pickups: list[AtariEntity] = []
        self._terrain: set[tuple[int, int]] = set()
        self._step_counter = 0
        self._gravity_timer = 0
        self._planet = 0
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-gravitar-v0"

    def _reset(self, seed: int):
        self._progress_count = 0
        return super()._reset(seed)

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._enemies = []
        self._bullets = []
        self._enemy_bullets = []
        self._fuel_pickups = []
        self._terrain = set()
        self._step_counter = 0
        self._gravity_timer = 0
        self._facing = 0
        self._fuel = self._MAX_FUEL
        rng = self.rng
        for x in range(self._WIDTH):
            self._set_cell(x, 0, "─")
            self._set_cell(x, self._HEIGHT - 1, "─")
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "│")
            self._set_cell(self._WIDTH - 1, y, "│")
        # Terrain (planet surface)
        base_y = self._HEIGHT - 4
        for x in range(1, self._WIDTH - 1):
            v = int(rng.integers(-2, 3))
            ty = max(self._HEIGHT // 2, min(base_y + v, self._HEIGHT - 2))
            for y in range(ty, self._HEIGHT - 1):
                self._terrain.add((x, y))
                self._set_cell(x, y, "=")
        # Enemies on terrain surface (3 per planet)
        for _ in range(3):
            for _ in range(20):
                ex = int(rng.integers(2, self._WIDTH - 2))
                ey = base_y - 1
                for sy in range(self._HEIGHT // 2, self._HEIGHT - 1):
                    if (ex, sy) in self._terrain:
                        ey = sy - 1
                        break
                if ey > 0 and (ex, ey) not in self._terrain:
                    e = self._add_entity("enemy", "E", ex, ey)
                    e.data["fire_cd"] = 0
                    self._enemies.append(e)
                    break
        # Fuel pickups
        for _ in range(3):
            for _ in range(20):
                fx = int(rng.integers(2, self._WIDTH - 2))
                fy = int(rng.integers(3, self._HEIGHT // 2))
                if (fx, fy) not in self._terrain:
                    self._fuel_pickups.append(
                        self._add_entity("fuel_pickup", "F", fx, fy)
                    )
                    break
        self._player_x = self._WIDTH // 2
        self._player_y = 3
        self._redraw()

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._step_counter += 1
        if action_name == "LEFT":
            self._facing = (self._facing - 1) % 8
        elif action_name == "RIGHT":
            self._facing = (self._facing + 1) % 8
        # Update player_dir from facing
        fdx, fdy = self._DIRS[self._facing]
        if abs(fdx) >= abs(fdy):
            self._player_dir = (
                (1 if fdx > 0 else -1, 0)
                if fdx != 0 else (0, 0)
            )
        else:
            self._player_dir = (
                (0, 1 if fdy > 0 else -1)
                if fdy != 0 else (0, 0)
            )
        # Thrust
        if action_name == "THRUST" and self._fuel > 0:
            self._fuel -= 2
            dx, dy = self._DIRS[self._facing]
            nx, ny = self._player_x + dx, self._player_y + dy
            if (0 < nx < self._WIDTH - 1
                    and 0 < ny < self._HEIGHT - 1
                    and (nx, ny) not in self._terrain):
                self._player_x, self._player_y = nx, ny
        # Gravity
        self._gravity_timer += 1
        if self._gravity_timer >= 3:
            self._gravity_timer = 0
            gy = self._player_y + 1
            if (0 < gy < self._HEIGHT - 1
                    and (self._player_x, gy) not in self._terrain):
                self._player_y = gy
            elif (self._player_x, gy) in self._terrain:
                # Audit BROKEN fix: terrain crash is single-life.
                self._message = "Crashed into surface!"
                self._on_life_lost()
                reward = self._DEATH_PENALTY
                return reward, True, info
        # Fire
        if action_name == "FIRE" and len(self._bullets) < 3:
            dx, dy = self._DIRS[self._facing]
            b = self._add_entity(
                "bullet", "*",
                self._player_x + dx, self._player_y + dy,
            )
            b.data.update(bdx=dx, bdy=dy, life=10)
            self._bullets.append(b)
        # Move bullets
        for b in self._bullets:
            if not b.alive:
                continue
            b.data["life"] -= 1
            if b.data["life"] <= 0:
                b.alive = False
                continue
            b.x += b.data["bdx"]
            b.y += b.data["bdy"]
            if (not (0 < b.x < self._WIDTH - 1
                     and 0 < b.y < self._HEIGHT - 1)
                    or (b.x, b.y) in self._terrain):
                b.alive = False
        # Bullet-enemy hits
        for b in self._bullets:
            if not b.alive:
                continue
            for e in self._enemies:
                if e.alive and e.x == b.x and e.y == b.y:
                    e.alive = False
                    b.alive = False
                    self._on_point_scored(2)
                    if self._progress_count < self._WIN_TARGET:
                        reward += 1.0 / self._WIN_TARGET
                        self._progress_count += 1
                    self._message = "Enemy down!"
                    break
        # Fuel pickup
        for fp in self._fuel_pickups:
            if (fp.alive and fp.x == self._player_x
                    and fp.y == self._player_y):
                fp.alive = False
                self._fuel = min(self._MAX_FUEL, self._fuel + 40)
                self._on_point_scored(1)
                self._message = "Fuel +40!"
        # Enemy fire
        for e in self._enemies:
            if not e.alive:
                continue
            e.data["fire_cd"] += 1
            if e.data["fire_cd"] >= 12:
                e.data["fire_cd"] = 0
                fdx = _sign(self._player_x, e.x)
                fdy = _sign(self._player_y, e.y)
                if abs(fdx) >= abs(fdy):
                    fdy = 0
                else:
                    fdx = 0
                if fdx or fdy:
                    eb = self._add_entity(
                        "enemy_bullet", "o",
                        e.x + fdx, e.y + fdy,
                    )
                    eb.data.update(bdx=fdx, bdy=fdy, life=8)
                    self._enemy_bullets.append(eb)
        # Move enemy bullets
        for eb in self._enemy_bullets:
            if not eb.alive:
                continue
            eb.data["life"] -= 1
            if eb.data["life"] <= 0:
                eb.alive = False
                continue
            eb.x += eb.data["bdx"]
            eb.y += eb.data["bdy"]
            if (not (0 < eb.x < self._WIDTH - 1
                     and 0 < eb.y < self._HEIGHT - 1)
                    or (eb.x, eb.y) in self._terrain):
                eb.alive = False
            elif (eb.x == self._player_x
                  and eb.y == self._player_y):
                # Audit BROKEN fix: enemy hit is single-life.
                eb.alive = False
                self._message = "Hit by enemy fire!"
                self._on_life_lost()
                reward = self._DEATH_PENALTY
                return reward, True, info
        # Out of fuel terminates the episode (single-life).
        if self._fuel <= 0:
            self._message = "Out of fuel!"
            self._on_life_lost()
            reward = self._DEATH_PENALTY
            return reward, True, info
        # Win check
        if self._progress_count >= self._WIN_TARGET and not self._game_over:
            self._game_over = True
            info["won"] = True
            self._message = "All planets cleared!"
        # Level clear (only if not won yet)
        elif not any(e.alive for e in self._enemies):
            self._level += 1
            self._planet += 1
            self._message = f"Planet {self._planet} cleared!"
            self._generate_level(self._level + self._planet)
        # Cleanup
        self._bullets = [b for b in self._bullets if b.alive]
        self._enemy_bullets = [
            eb for eb in self._enemy_bullets if eb.alive
        ]
        self._enemies = [e for e in self._enemies if e.alive]
        self._fuel_pickups = [
            f for f in self._fuel_pickups if f.alive
        ]
        self._redraw()
        info["fuel"] = self._fuel
        info["planet"] = self._planet
        return reward, self._game_over, info

    def _redraw(self) -> None:
        for y in range(1, self._HEIGHT - 1):
            for x in range(1, self._WIDTH - 1):
                ch = "=" if (x, y) in self._terrain else " "
                self._set_cell(x, y, ch)
        for e in self._enemies:
            if e.alive:
                self._set_cell(e.x, e.y, "E")
        for fp in self._fuel_pickups:
            if fp.alive:
                self._set_cell(fp.x, fp.y, "F")
        for b in self._bullets:
            if b.alive:
                self._set_cell(b.x, b.y, "*")
        for eb in self._enemy_bullets:
            if eb.alive:
                self._set_cell(eb.x, eb.y, "o")

    def _advance_entities(self) -> None:
        self._entities = [e for e in self._entities if e.alive]

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "─": "wall", "│": "wall", "=": "terrain",
            "E": "enemy (2pts)", "F": "fuel pickup (+40 fuel, +1 pt)",
            "*": "your bullet", "o": "enemy bullet",
            " ": "space",
        }.get(ch, ch)

    _FACING_NAMES = (
        "up", "up-right", "right", "down-right",
        "down", "down-left", "left", "up-left",
    )

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        parts = [f"Fuel: {self._fuel}"]
        if self._DIRS[self._facing] not in self._DIR_CHARS:
            parts.append(f"Ship facing: {self._FACING_NAMES[self._facing]}")
        extra = "  ".join(parts)
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Navigate with gravity pulling you down. "
            "THRUST to fly, LEFT/RIGHT to rotate, FIRE to shoot. "
            "Destroy enemies, collect fuel (F). "
            "Crashing or running out of fuel ends the episode."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Gravitar.\n\n"
            "TASK\n"
            "Fly a ship against a planet's gravity, destroy all "
            "surface enemies, and grab fuel pickups. Clearing a "
            "planet advances the level to the next planet.\n\n"
            "BOARD\n"
            "20x20 arena, bordered by walls ('-', '|'). Jagged "
            "terrain '=' lines the lower part of the arena, with "
            "enemies 'E' on its surface and fuel 'F' floating above. "
            "Your bullets are '*', enemy bullets 'o'. You are an "
            "arrow glyph.\n\n"
            "MECHANICS\n"
            "LEFT / RIGHT rotate in 45-degree steps (8 facings). "
            "THRUST spends 2 fuel and moves 1 cell in the facing "
            "direction (blocked by terrain). Gravity: every 3 "
            "steps you fall 1 row; hitting terrain ends the "
            "episode (single-life). FIRE launches a bullet (max "
            "3 alive, TTL 10). Enemies fire at you every 12 "
            "steps along the dominant axis.\n\n"
            "SCORING\n"
            "+1/12 reward per enemy destroyed (Pattern A full-scope "
            "= 4 planets x 3 enemies). Fuel pickups restore 40 fuel "
            "(no reward). No per-step penalty; fuel decreases only "
            "when thrusting.\n\n"
            "TERMINATION\n"
            "Single-life: crashing into terrain, being hit by an "
            "enemy bullet, or running out of fuel ends the "
            "episode with reward -1. Clearing all enemies "
            "advances level + planet. The episode also ends "
            "after 12 enemies destroyed (cumulative reward "
            "plateaus at +1.0) or after max_turns.\n\n"
            "HUD\n"
            "Shows score, level, fuel remaining, and ship "
            "facing direction.\n\n"
            + self.action_spec.render_for_prompt()
        )
