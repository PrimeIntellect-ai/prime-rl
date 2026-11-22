"""Atari Name This Game environment.

Underwater shooter. Protect your fish from enemies.

Gym ID: glyphbench/atari-namethisgame-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase, AtariEntity


class NameThisGameEnv(AtariBase):
    """Name This Game: underwater shooter protecting fish.

    20x20 grid. Enemies (sharks, octopi) approach from sides.
    Player moves horizontally at surface and shoots down.
    Protect the fish at the bottom.

    Actions: NOOP, LEFT, RIGHT, FIRE
    Pattern D: +1/_WIN_TARGET per enemy shot (full-scope = 5 waves
    x 6 enemies = 30). -1.0 on death (all fish eaten).
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "FIRE"),
        descriptions=(
            "do nothing",
            "move left",
            "move right",
            "fire downward",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 20
    _PLAYER_Y = 2
    _WATER_TOP = 4
    _FISH_Y = 17
    _MAX_BULLETS = 2

    # Pattern D full-scope target: 30 (5 waves x 6 enemies).
    _WIN_TARGET: int = 30
    _DEATH_PENALTY: float = -1.0

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._bullets: list[AtariEntity] = []
        self._enemies: list[AtariEntity] = []
        self._fish: list[AtariEntity] = []
        self._step_counter: int = 0
        self._spawn_cd: int = 0
        self._kills: int = 0
        self._wave_target: int = 0
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-namethisgame-v0"

    def _reset(self, seed: int):
        self._progress_count = 0
        return super()._reset(seed)

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._bullets = []
        self._enemies = []
        self._fish = []
        self._step_counter = 0
        self._spawn_cd = 0
        self._kills = 0
        self._wave_target = 6  # 6 enemies per wave

        # Borders
        for x in range(self._WIDTH):
            self._set_cell(x, 0, "─")
            self._set_cell(x, self._HEIGHT - 1, "─")
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "│")
            self._set_cell(self._WIDTH - 1, y, "│")

        # Water surface
        for x in range(1, self._WIDTH - 1):
            self._set_cell(x, self._WATER_TOP, "~")

        # Sea floor
        for x in range(1, self._WIDTH - 1):
            self._set_cell(x, self._HEIGHT - 2, "=")

        # Fish to protect
        for i in range(3):
            fx = 5 + i * 4
            f = self._add_entity("fish", "f", fx, self._FISH_Y)
            self._fish.append(f)

        # Player at surface
        self._player_x = self._WIDTH // 2
        self._player_y = self._PLAYER_Y

        # Spawn initial enemies
        for _ in range(3):
            self._spawn_enemy()

        self._redraw()

    def _spawn_enemy(self) -> None:
        rng = self.rng
        side = int(rng.integers(0, 2))
        ex = 1 if side == 0 else self._WIDTH - 2
        ey = int(rng.integers(
            self._WATER_TOP + 1, self._HEIGHT - 3
        ))
        dx = 1 if side == 0 else -1
        etype = int(rng.integers(0, 2))
        ch = "S" if etype == 0 else "O"
        name = "shark" if etype == 0 else "octopus"
        e = self._add_entity(name, ch, ex, ey, dx=dx)
        e.data["timer"] = 0
        e.data["speed"] = max(1, 3 - self._level // 3)
        self._enemies.append(e)

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

        # Fire downward
        if action_name == "FIRE" and len(self._bullets) < self._MAX_BULLETS:
            b = self._add_entity(
                "bullet", "!", self._player_x,
                self._WATER_TOP + 1, dy=1
            )
            self._bullets.append(b)

        # Move bullets
        for b in self._bullets:
            if not b.alive:
                continue
            b.y += b.dy
            if b.y >= self._HEIGHT - 2:
                b.alive = False
                continue
            for e in self._enemies:
                if e.alive and e.x == b.x and e.y == b.y:
                    e.alive = False
                    b.alive = False
                    self._on_point_scored(1)
                    if self._progress_count < self._WIN_TARGET:
                        reward += 1.0 / self._WIN_TARGET
                        self._progress_count += 1
                    self._kills += 1
                    self._message = "Enemy hit!"
                    break
        self._bullets = [b for b in self._bullets if b.alive]

        # Move enemies
        for e in self._enemies:
            if not e.alive:
                continue
            e.data["timer"] += 1
            spd = e.data.get("speed", 2)
            if e.data["timer"] % spd == 0:
                e.x += e.dx
                if e.x <= 0 or e.x >= self._WIDTH - 1:
                    e.dx = -e.dx
                    e.x = max(1, min(self._WIDTH - 2, e.x))

                # Drift toward fish
                if e.data["timer"] % 6 == 0 and e.y < self._FISH_Y:
                    e.y += 1

        # Enemies eating fish (no per-fish reward)
        for e in self._enemies:
            if not e.alive:
                continue
            for f in self._fish:
                if (
                    f.alive
                    and abs(e.x - f.x) <= 1
                    and e.y == f.y
                ):
                    f.alive = False
                    e.alive = False
                    self._message = "Fish eaten!"
                    break

        # Check all fish dead (Pattern D death penalty)
        alive_fish = sum(1 for f in self._fish if f.alive)
        if alive_fish == 0 and not self._game_over:
            self._on_life_lost()
            reward = self._DEATH_PENALTY
            self._message = "All fish eaten!"

        # Spawn enemies
        self._spawn_cd -= 1
        alive_enemies = sum(1 for e in self._enemies if e.alive)
        if self._spawn_cd <= 0 and alive_enemies < 5:
            self._spawn_enemy()
            self._spawn_cd = max(4, 10 - self._level)

        # Wave clear (no direct reward; per-enemy progress drives it)
        if self._kills >= self._wave_target and not self._game_over:
            self._message = "Wave cleared!"
            self._level += 1
            if self._progress_count >= self._WIN_TARGET:
                self._game_over = True
                info["won"] = True
                self._message = "All waves cleared!"
            else:
                self._generate_level(self._level)

        self._enemies = [e for e in self._enemies if e.alive]
        self._redraw()
        info["fish_alive"] = alive_fish
        return reward, self._game_over, info

    def _redraw(self) -> None:
        for y in range(1, self._HEIGHT - 1):
            for x in range(1, self._WIDTH - 1):
                if y == self._WATER_TOP:
                    self._set_cell(x, y, "~")
                elif y == self._HEIGHT - 2:
                    self._set_cell(x, y, "=")
                elif y < self._WATER_TOP:
                    self._set_cell(x, y, " ")
                else:
                    self._set_cell(x, y, "·")

        for f in self._fish:
            if f.alive:
                self._set_cell(f.x, f.y, "f")
        for e in self._enemies:
            if e.alive:
                self._set_cell(e.x, e.y, e.char)
        for b in self._bullets:
            if b.alive:
                self._set_cell(b.x, b.y, "!")

    def _advance_entities(self) -> None:
        self._entities = [e for e in self._entities if e.alive]

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "─": "wall",
            "│": "wall",
            "~": "water surface",
            "=": "sea floor",
            "·": "water",
            "S": "shark",
            "O": "octopus",
            "f": "your fish (protect)",
            "!": "your bullet",
            " ": "air",
        }.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        alive_fish = sum(1 for f in self._fish if f.alive)
        extra = (
            f"Kills: {self._kills}"
            f"/{self._wave_target}  "
            f"Fish: {alive_fish}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Move at the surface and shoot downward to destroy "
            "sharks and octopi. Protect your fish at the bottom. "
            "Clear the wave target to advance."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Name This Game.\n\n"
            "TASK\n"
            "Patrol the ocean surface and shoot down sharks and "
            "octopi before they eat the fish at the bottom. Clear "
            "each wave's target kills to advance.\n\n"
            "BOARD\n"
            "20x20 underwater scene. Walls '-' and '|'. Water "
            "surface 'tilde' tops the scene; the sea floor '=' lies "
            "at the bottom. Three friendly fish 'f' rest on the "
            "floor. Sharks 'S' and octopi 'O' enter from the sides "
            "and swim horizontally. Your bullets '!' travel down. "
            "You patrol the surface as an arrow glyph.\n\n"
            "MECHANICS\n"
            "LEFT / RIGHT move you 1 cell along the surface. FIRE "
            "drops a bullet from the surface (max 2 bullets). "
            "Bullets travel down 1 cell per step, dying at the sea "
            "floor. Enemies advance horizontally at per-step speed "
            "max(1, 3 - level/3), reverse at walls, and drift one "
            "row down every 6 steps. An enemy within 1 column of a "
            "fish at the fish row eats the fish and dies.\n\n"
            "SCORING\n"
            "+1/30 reward per enemy you shoot (Pattern D "
            "full-scope = 5 waves x 6 enemies = 30). -1.0 if all "
            "fish are eaten (failure terminates).\n\n"
            "TERMINATION\n"
            "Losing all fish ends the episode with -1.0. Episode "
            "ends after 30 enemies shot (cumulative reward "
            "plateaus at +1.0) or after max_turns.\n\n"
            "HUD\n"
            "Shows score, level, kills toward wave target, "
            "fish alive.\n\n"
            + self.action_spec.render_for_prompt()
        )
