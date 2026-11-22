"""Atari Star Gunner environment.

Side-scrolling space shooter with waves of enemies.

Gym ID: glyphbench/atari-stargunner-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase, AtariEntity


class StarGunnerEnv(AtariBase):
    """Star Gunner: side-scrolling space shooter.

    30x16 grid. Enemies scroll in from the right.
    Shoot them before they pass or hit you.

    Actions: NOOP, LEFT, RIGHT, UP, DOWN, FIRE
    Pattern D: +1/_WIN_TARGET per enemy destroyed,
    -1 on collision with enemy or enemy bullet.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "UP", "DOWN", "FIRE"),
        descriptions=(
            "do nothing",
            "move left",
            "move right",
            "move up",
            "move down",
            "fire a bullet right",
        ),
    )

    _WIDTH = 30
    _HEIGHT = 16
    _MAX_BULLETS = 4

    # Pattern D full-scope: 30 enemies destroyed across waves.
    _WIN_TARGET: int = 30
    _DEATH_PENALTY: float = -1.0

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._bullets: list[AtariEntity] = []
        self._enemies: list[AtariEntity] = []
        self._enemy_bullets: list[AtariEntity] = []
        self._step_counter: int = 0
        self._spawn_cd: int = 0
        self._enemies_killed: int = 0
        self._wave_target: int = 0
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-stargunner-v0"

    def _reset(self, seed: int):
        self._progress_count = 0
        return super()._reset(seed)

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._bullets = []
        self._enemies = []
        self._enemy_bullets = []
        self._step_counter = 0
        self._spawn_cd = 0
        self._enemies_killed = 0
        self._wave_target = 8 + self._level * 2

        # Borders
        for x in range(self._WIDTH):
            self._set_cell(x, 0, "─")
            self._set_cell(x, self._HEIGHT - 1, "─")
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "│")
            self._set_cell(self._WIDTH - 1, y, "│")

        # Stars in background
        rng = self.rng
        for _ in range(10):
            sx = int(rng.integers(1, self._WIDTH - 1))
            sy = int(rng.integers(1, self._HEIGHT - 1))
            self._set_cell(sx, sy, "*")

        self._player_x = 3
        self._player_y = self._HEIGHT // 2
        self._redraw()

    def _spawn_enemy(self) -> None:
        rng = self.rng
        ey = int(rng.integers(2, self._HEIGHT - 2))
        etype = int(rng.integers(0, 3))
        chars = ("V", "W", "X")
        speeds = (-1, -1, -2)
        e = self._add_entity(
            "enemy", chars[etype], self._WIDTH - 2, ey,
            dx=speeds[etype]
        )
        e.data["timer"] = 0
        e.data["fire_cd"] = int(rng.integers(8, 20))
        e.data["pts"] = etype + 1
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
        elif action_name == "RIGHT" and self._player_x < self._WIDTH // 2:
            self._player_x += 1
            self._player_dir = (1, 0)
        elif action_name == "UP" and self._player_y > 1:
            self._player_y -= 1
            self._player_dir = (0, -1)
        elif action_name == "DOWN" and self._player_y < self._HEIGHT - 2:
            self._player_y += 1
            self._player_dir = (0, 1)

        # Fire
        if action_name == "FIRE" and len(self._bullets) < self._MAX_BULLETS:
            b = self._add_entity(
                "bullet", "→", self._player_x + 1,
                self._player_y, dx=2
            )
            self._bullets.append(b)

        # Move bullets
        for b in self._bullets:
            if not b.alive:
                continue
            b.x += b.dx
            if b.x >= self._WIDTH - 1:
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
                    self._enemies_killed += 1
                    self._message = "Enemy destroyed!"
                    break
        self._bullets = [b for b in self._bullets if b.alive]

        # Move enemies
        for e in self._enemies:
            if not e.alive:
                continue
            e.data["timer"] += 1
            if e.data["timer"] % 2 == 0:
                e.x += e.dx
                if e.x <= 0:
                    e.alive = False
                    continue

            # Enemy fire
            if e.data["timer"] % e.data["fire_cd"] == 0 and len(self._enemy_bullets) < 3:
                eb = self._add_entity(
                    "ebullet", "←", e.x - 1, e.y, dx=-1
                )
                self._enemy_bullets.append(eb)

        # Move enemy bullets
        for eb in self._enemy_bullets:
            if not eb.alive:
                continue
            eb.x += eb.dx
            if eb.x <= 0:
                eb.alive = False
                continue
            if eb.x == self._player_x and eb.y == self._player_y:
                eb.alive = False
                self._on_life_lost()
                self._message = "Hit! Game Over."
                reward = self._DEATH_PENALTY
                return reward, self._game_over, info
        self._enemy_bullets = [
            eb for eb in self._enemy_bullets if eb.alive
        ]

        # Player collision with enemy -- terminal failure
        for e in self._enemies:
            if (
                e.alive
                and e.x == self._player_x
                and e.y == self._player_y
            ):
                e.alive = False
                self._on_life_lost()
                self._message = "Collision! Game Over."
                reward = self._DEATH_PENALTY
                return reward, self._game_over, info

        # Spawn enemies
        self._spawn_cd -= 1
        alive_enemies = [e for e in self._enemies if e.alive]
        total_spawned = self._enemies_killed + len(alive_enemies)
        if (
            self._spawn_cd <= 0
            and total_spawned < self._wave_target
            and len(alive_enemies) < 6
        ):
            self._spawn_enemy()
            self._spawn_cd = max(5, 12 - self._level)

        # Level clear (no extra reward — kills already counted)
        if self._enemies_killed >= self._wave_target:
            self._message = "Wave cleared!"
            self._level += 1
            saved_progress = self._progress_count
            self._generate_level(self._level)
            self._progress_count = saved_progress

        # Win check
        if (
            self._progress_count >= self._WIN_TARGET
            and not self._game_over
        ):
            self._game_over = True
            info["won"] = True
            self._message = "Star sector cleared!"

        self._enemies = [e for e in self._enemies if e.alive]
        self._redraw()
        return reward, self._game_over, info

    def _redraw(self) -> None:
        for y in range(1, self._HEIGHT - 1):
            for x in range(1, self._WIDTH - 1):
                self._set_cell(x, y, " ")
        # Redraw stars
        for y in range(1, self._HEIGHT - 1):
            for x in range(1, self._WIDTH - 1):
                if (x * 7 + y * 13) % 37 == 0:
                    self._set_cell(x, y, "*")

        for e in self._enemies:
            if e.alive:
                self._set_cell(e.x, e.y, e.char)
        for b in self._bullets:
            if b.alive:
                self._set_cell(b.x, b.y, "→")
        for eb in self._enemy_bullets:
            if eb.alive:
                self._set_cell(eb.x, eb.y, "←")

    def _advance_entities(self) -> None:
        self._entities = [e for e in self._entities if e.alive]

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "─": "wall",
            "│": "wall",
            "*": "star",
            "V": "enemy fighter (10pts)",
            "W": "enemy bomber (20pts)",
            "X": "enemy ace (30pts)",
            "→": "your bullet",
            "←": "enemy bullet",
            " ": "space",
        }.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        extra = (
            f"Kills: {self._enemies_killed}"
            f"/{self._wave_target}  "
            f"Wave: {self._level}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Shoot enemies scrolling in from the right. "
            "Dodge enemy fire and collisions. "
            "Destroy all wave targets to advance."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Star Gunner.\n\n"
            "TASK\n"
            "Pilot a ship in the left half of the screen and shoot "
            "enemies flying in from the right. Kill enough enemies "
            "per wave to clear it and advance the level.\n\n"
            "BOARD\n"
            "30x16 starfield bordered by walls. Stars '*' scatter "
            "for decoration. Three enemy types: 'V' fighter (10 "
            "pts), 'W' bomber (20 pts), 'X' ace (30 pts, moves "
            "faster). Your bullets '->', enemy bullets '<-'. You "
            "are confined to the left half of the screen.\n\n"
            "MECHANICS\n"
            "LEFT/RIGHT/UP/DOWN move 1 cell. FIRE launches a "
            "bullet traveling right with dx=2 (max 4 alive). "
            "Enemies spawn at x=W-2 on random rows every max(5, "
            "12-level) steps, moving left every 2 steps; aces "
            "move dx=-2. Each enemy fires a bullet leftward every "
            "fire_cd steps (8-20, random).\n\n"
            "SCORING\n"
            "Pattern D: +1/30 reward per enemy destroyed "
            "(regardless of type). Wave clears continue without "
            "extra reward. -1 reward when hit by an enemy bullet "
            "or you collide with an enemy. Cumulative reward "
            "bound: [-1, +1].\n\n"
            "TERMINATION\n"
            "Single-life: any death event ends with -1. "
            "Destroying 30 enemies ends with cumulative +1. "
            "Episode also ends after max_turns.\n\n"
            "HUD\n"
            "Shows score, wave, kills toward wave target.\n\n"
            + self.action_spec.render_for_prompt()
        )
