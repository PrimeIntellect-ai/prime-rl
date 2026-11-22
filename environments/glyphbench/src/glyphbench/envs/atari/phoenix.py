"""Atari Phoenix environment.

Waves of birds descending, boss phases.

Gym ID: glyphbench/atari-phoenix-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase, AtariEntity


class PhoenixEnv(AtariBase):
    """Phoenix: shoot waves of descending birds.

    20x24 grid. Player at bottom shoots up. Birds swoop down.
    Every 3 waves a boss appears.

    Actions: NOOP, LEFT, RIGHT, FIRE, SHIELD
    Pattern D: +1/_WIN_TARGET per bird shot or boss hit
    (full-scope = 5 waves x 8 phoenixes = 40). -1.0 on death.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "FIRE", "SHIELD"),
        descriptions=(
            "do nothing",
            "move left",
            "move right",
            "fire a bullet upward",
            "activate shield (brief invincibility)",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 24
    _PLAYER_Y = 22

    # Pattern D full-scope target: 40 (5 waves x 8 phoenixes).
    _WIN_TARGET: int = 40
    _DEATH_PENALTY: float = -1.0

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._birds: list[AtariEntity] = []
        self._bullets: list[AtariEntity] = []
        self._enemy_bullets: list[AtariEntity] = []
        self._step_counter: int = 0
        self._shield_timer: int = 0
        self._boss: AtariEntity | None = None
        self._boss_hp: int = 0
        self._wave_type: str = "birds"
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-phoenix-v0"

    def _reset(self, seed: int):
        self._progress_count = 0
        return super()._reset(seed)

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._birds = []
        self._bullets = []
        self._enemy_bullets = []
        self._step_counter = 0
        self._shield_timer = 0
        self._boss = None
        self._boss_hp = 0

        for x in range(self._WIDTH):
            self._set_cell(x, 0, "─")
            self._set_cell(x, self._HEIGHT - 1, "─")
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "│")
            self._set_cell(self._WIDTH - 1, y, "│")

        self._player_x = self._WIDTH // 2
        self._player_y = self._PLAYER_Y

        if self._level % 3 == 0:
            self._wave_type = "boss"
            self._spawn_boss()
        else:
            self._wave_type = "birds"
            self._spawn_birds()

    def _spawn_birds(self) -> None:
        rows = min(2 + self._level, 5)
        cols = min(3 + self._level, 8)
        for r in range(rows):
            for c in range(cols):
                x = 2 + c * 2
                y = 2 + r * 2
                if x < self._WIDTH - 1 and y < self._HEIGHT - 1:
                    b = self._add_entity("bird", "W", x, y)
                    b.data["swoop"] = 0
                    b.data["dir"] = 1 if r % 2 == 0 else -1
                    self._birds.append(b)

    def _spawn_boss(self) -> None:
        bx = self._WIDTH // 2
        self._boss = self._add_entity("boss", "B", bx, 3)
        self._boss.data["dir"] = 1
        self._boss_hp = 5 + self._level

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._step_counter += 1

        if action_name == "LEFT" and self._player_x > 1:
            self._player_x -= 1
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT" and self._player_x < self._WIDTH - 2:
            self._player_x += 1
            self._player_dir = (1, 0)
        elif action_name == "FIRE" and len(self._bullets) < 2:
            b = self._add_entity(
                "bullet", "!", self._player_x, self._player_y - 1,
                dy=-1,
            )
            self._bullets.append(b)
        elif action_name == "SHIELD":
            self._shield_timer = 5

        if self._shield_timer > 0:
            self._shield_timer -= 1

        # Move bullets
        for b in self._bullets:
            if not b.alive:
                continue
            b.y += b.dy
            if b.y <= 0:
                b.alive = False

        # Bullet-bird collisions
        for b in self._bullets:
            if not b.alive:
                continue
            for bird in self._birds:
                if (
                    bird.alive
                    and bird.x == b.x
                    and bird.y == b.y
                ):
                    bird.alive = False
                    b.alive = False
                    self._on_point_scored(1)
                    if self._progress_count < self._WIN_TARGET:
                        reward += 1.0 / self._WIN_TARGET
                        self._progress_count += 1
                    self._message = "Bird hit!"
                    break

        # Bullet-boss collision
        if self._boss and self._boss.alive:
            for b in self._bullets:
                if (
                    b.alive
                    and abs(b.x - self._boss.x) <= 1
                    and b.y == self._boss.y
                ):
                    b.alive = False
                    self._boss_hp -= 1
                    if self._progress_count < self._WIN_TARGET:
                        reward += 1.0 / self._WIN_TARGET
                        self._progress_count += 1
                    if self._boss_hp <= 0:
                        self._boss.alive = False
                        self._message = "Boss destroyed!"
                    else:
                        self._message = f"Boss hit! HP={self._boss_hp}"

        self._bullets = [b for b in self._bullets if b.alive]

        # Move birds
        if self._step_counter % 2 == 0:
            for bird in self._birds:
                if not bird.alive:
                    continue
                d = bird.data.get("dir", 1)
                bird.x += d
                if bird.x <= 1 or bird.x >= self._WIDTH - 2:
                    bird.data["dir"] = -d
                    bird.y += 1
                # Swoop occasionally
                bird.data["swoop"] = bird.data.get("swoop", 0) + 1
                if bird.data["swoop"] % 8 == 0:
                    bird.y += 1

        # Move boss
        if self._boss and self._boss.alive and self._step_counter % 3 == 0:
            d = self._boss.data.get("dir", 1)
            self._boss.x += d
            if (
                self._boss.x <= 2
                or self._boss.x >= self._WIDTH - 3
            ):
                self._boss.data["dir"] = -d

        # Enemy fire
        if self._step_counter % 8 == 0:
            shooters = [
                b for b in self._birds if b.alive
            ]
            if self._boss and self._boss.alive:
                shooters.append(self._boss)
            if shooters and len(self._enemy_bullets) < 3:
                s = shooters[
                    int(self.rng.integers(len(shooters)))
                ]
                eb = self._add_entity(
                    "enemy_bullet", "↓", s.x, s.y + 1, dy=1,
                )
                self._enemy_bullets.append(eb)

        # Move enemy bullets
        for eb in self._enemy_bullets:
            if not eb.alive:
                continue
            eb.y += eb.dy
            if eb.y >= self._HEIGHT - 1:
                eb.alive = False
            elif (
                eb.x == self._player_x
                and eb.y == self._player_y
                and self._shield_timer <= 0
            ):
                eb.alive = False
                self._on_life_lost()
                reward = self._DEATH_PENALTY
                self._message = "Hit by enemy fire!"

        self._enemy_bullets = [
            eb for eb in self._enemy_bullets if eb.alive
        ]

        # Bird reaches player row
        for bird in self._birds:
            if bird.alive and bird.y >= self._PLAYER_Y:
                bird.alive = False
                if self._shield_timer <= 0 and not self._game_over:
                    self._on_life_lost()
                    reward = self._DEATH_PENALTY
                    self._message = "Bird swooped you!"

        self._birds = [b for b in self._birds if b.alive]

        # Level clear
        if (
            self._wave_type == "birds"
            and not self._birds
            and not self._game_over
        ):
            self._level += 1
            self._message = "Wave cleared!"
            if self._progress_count >= self._WIN_TARGET:
                self._game_over = True
                info["won"] = True
                self._message = "All waves cleared!"
            else:
                self._generate_level(self._level)
        elif (
            self._wave_type == "boss"
            and (self._boss is None or not self._boss.alive)
            and not self._game_over
        ):
            self._level += 1
            self._message = "Boss defeated!"
            if self._progress_count >= self._WIN_TARGET:
                self._game_over = True
                info["won"] = True
                self._message = "All waves cleared!"
            else:
                self._generate_level(self._level)

        # Win check (in case progress hits target before wave clear)
        if self._progress_count >= self._WIN_TARGET and not self._game_over:
            self._game_over = True
            info["won"] = True
            self._message = "All waves cleared!"

        self._redraw()
        return reward, self._game_over, info

    def _redraw(self) -> None:
        for y in range(1, self._HEIGHT - 1):
            for x in range(1, self._WIDTH - 1):
                self._set_cell(x, y, " ")
        for bird in self._birds:
            if bird.alive:
                self._set_cell(bird.x, bird.y, bird.char)
        if self._boss and self._boss.alive:
            self._set_cell(self._boss.x, self._boss.y, "B")
            self._set_cell(self._boss.x - 1, self._boss.y, "[")
            self._set_cell(self._boss.x + 1, self._boss.y, "]")
        for b in self._bullets:
            if b.alive:
                self._set_cell(b.x, b.y, "!")
        for eb in self._enemy_bullets:
            if eb.alive:
                self._set_cell(eb.x, eb.y, "↓")
        if self._shield_timer > 0:
            self._set_cell(
                self._player_x - 1, self._player_y, "("
            )
            self._set_cell(
                self._player_x + 1, self._player_y, ")"
            )

    def _advance_entities(self) -> None:
        self._entities = [e for e in self._entities if e.alive]

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "─": "wall", "│": "wall",
            "W": "bird", "B": "boss",
            "[": "boss wing", "]": "boss wing",
            "!": "your bullet", "↓": "enemy bullet",
            "(": "shield aura", ")": "shield aura",
            " ": "empty",
        }.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        birds = sum(1 for b in self._birds if b.alive)
        shield = (
            str(self._shield_timer)
            if self._shield_timer > 0 else "OFF"
        )
        extra = (
            f"Wave: {self._level} ({self._wave_type})  "
            f"Birds: {birds}  Shield: {shield}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Shoot descending birds. Use SHIELD for brief "
            "invincibility. Every 3 waves a boss appears."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Phoenix.\n\n"
            "TASK\n"
            "Shoot waves of descending birds; every 3rd level is a "
            "boss wave where a large mothership appears. Clearing "
            "a wave advances the level.\n\n"
            "BOARD\n"
            "20x24 arena bordered by walls. Birds 'W' spawn in a "
            "formation near the top of the arena; the boss 'B' "
            "appears in the same area on boss waves with wings '[' "
            "and ']'. Your bullets are '!', enemy bullets "
            "down-arrows. You patrol along the bottom as an arrow "
            "glyph; when shield is active, '(' and ')' show beside "
            "you.\n\n"
            "MECHANICS\n"
            "LEFT / RIGHT move you 1 cell. FIRE launches a bullet "
            "straight up (max 2 alive). SHIELD activates 5 steps "
            "of invulnerability. Birds move every 2 steps, "
            "bouncing off walls and stepping down; every 8 steps "
            "each bird swoops 1 row further. Boss moves every 3 "
            "steps bouncing between columns 2 and W-3. Every 8 "
            "steps a random enemy fires a bullet downward (max 3 "
            "enemy bullets).\n\n"
            "SCORING\n"
            "+1/40 reward per bird shot, plus +1/40 per boss hit "
            "(Pattern D full-scope = 5 waves x 8 phoenixes = 40). "
            "-1.0 on death (enemy bullet or bird swoop while not "
            "shielded).\n\n"
            "TERMINATION\n"
            "Being hit by an enemy bullet or swooped by a bird "
            "(unless shield active) ends the episode with -1.0. "
            "Episode ends after 40 enemies neutralized (cumulative "
            "reward plateaus at +1.0) or after max_turns.\n\n"
            "HUD\n"
            "Shows score, level, wave type (birds/boss), "
            "bird count, shield timer.\n\n"
            + self.action_spec.render_for_prompt()
        )
