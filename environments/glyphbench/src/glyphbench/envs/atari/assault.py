"""Atari Assault environment.

Fixed turret at bottom, enemies descend in formations.

Gym ID: glyphbench/atari-assault-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase, AtariEntity


class AssaultEnv(AtariBase):
    """Assault: defend with a turret against descending enemies.

    20x20 grid. Enemies descend in formations. Player turret
    at bottom can move left/right and fire upward.

    Actions: NOOP, LEFT, RIGHT, FIRE
    Pattern A: +1/_WIN_TARGET per enemy killed (full-scope = 5 waves x
    6 enemies = 30 progress units). No failure penalty.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "FIRE"),
        descriptions=(
            "do nothing",
            "move turret left",
            "move turret right",
            "fire a bullet upward",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 20
    _PLAYER_Y = 18

    # Pattern A full-scope target: 5 waves x 6 enemies = 30 progress units.
    _WIN_TARGET: int = 30

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._enemies: list[AtariEntity] = []
        self._bullets: list[AtariEntity] = []
        self._enemy_bullets: list[AtariEntity] = []
        self._step_counter: int = 0
        self._spawn_timer: int = 0
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-assault-v0"

    def _reset(self, seed: int) -> GridObservation:
        self._progress_count = 0
        return super()._reset(seed)

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._enemies = []
        self._bullets = []
        self._enemy_bullets = []
        self._step_counter = 0
        self._spawn_timer = 0

        for x in range(self._WIDTH):
            self._set_cell(x, 0, "─")
            self._set_cell(x, self._HEIGHT - 1, "─")
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "│")
            self._set_cell(self._WIDTH - 1, y, "│")

        self._player_x = self._WIDTH // 2
        self._player_y = self._PLAYER_Y

        self._spawn_formation()
        self._redraw()

    def _spawn_formation(self) -> None:
        rng = self.rng
        cx = int(rng.integers(4, self._WIDTH - 4))
        # Leader
        leader = self._add_entity("leader", "A", cx, 2)
        leader.data["dir"] = 1
        self._enemies.append(leader)
        # Wingmen
        for dx in (-2, 2):
            nx = cx + dx
            if 1 < nx < self._WIDTH - 1:
                e = self._add_entity("enemy", "a", nx, 3)
                e.data["dir"] = 1
                self._enemies.append(e)

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._step_counter += 1

        if action_name == "LEFT" and self._player_x > 1:
            self._player_x -= 1
            self._player_dir = (-1, 0)
        elif (
            action_name == "RIGHT"
            and self._player_x < self._WIDTH - 2
        ):
            self._player_x += 1
            self._player_dir = (1, 0)
        elif action_name == "FIRE" and len(self._bullets) < 2:
            b = self._add_entity(
                "bullet", "!", self._player_x,
                self._player_y - 1, dy=-1,
            )
            self._bullets.append(b)

        # Move bullets
        for b in self._bullets:
            if not b.alive:
                continue
            b.y += b.dy
            if b.y <= 0:
                b.alive = False

        # Bullet-enemy collisions
        for b in self._bullets:
            if not b.alive:
                continue
            for e in self._enemies:
                if e.alive and e.x == b.x and e.y == b.y:
                    e.alive = False
                    b.alive = False
                    pts = 2 if e.etype == "leader" else 1
                    self._on_point_scored(pts)
                    if self._progress_count < self._WIN_TARGET:
                        reward += 1.0 / self._WIN_TARGET
                        self._progress_count += 1
                    self._message = "Enemy hit!"
                    break
        self._bullets = [b for b in self._bullets if b.alive]

        # Move enemies
        if self._step_counter % 3 == 0:
            for e in self._enemies:
                if not e.alive:
                    continue
                d = e.data.get("dir", 1)
                e.x += d
                if e.x <= 1 or e.x >= self._WIDTH - 2:
                    e.data["dir"] = -d
                    e.y += 1

        # Enemy fire
        if self._step_counter % 5 == 0:
            alive = [e for e in self._enemies if e.alive]
            if alive and len(self._enemy_bullets) < 3:
                s = alive[int(self.rng.integers(len(alive)))]
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
            ):
                eb.alive = False
                self._on_life_lost()
                self._message = "Hit! Lost a life."
                self._player_x = self._WIDTH // 2
        self._enemy_bullets = [
            eb for eb in self._enemy_bullets if eb.alive
        ]

        # Enemy reaches bottom
        for e in self._enemies:
            if e.alive and e.y >= self._PLAYER_Y:
                e.alive = False
                self._on_life_lost()
                self._message = "Enemy reached you!"
        self._enemies = [e for e in self._enemies if e.alive]

        # Win check
        if self._progress_count >= self._WIN_TARGET and not self._game_over:
            self._game_over = True
            info["won"] = True
            self._message = "All waves cleared!"

        # Spawn new formations periodically
        self._spawn_timer += 1
        if (not self._enemies or self._spawn_timer >= 30) and not self._game_over:
            if not self._enemies:
                self._level += 1
                self._message = "Wave cleared!"
            self._spawn_timer = 0
            self._spawn_formation()

        self._redraw()
        return reward, self._game_over, info

    def _redraw(self) -> None:
        for y in range(1, self._HEIGHT - 1):
            for x in range(1, self._WIDTH - 1):
                self._set_cell(x, y, " ")
        for e in self._enemies:
            if e.alive:
                self._set_cell(e.x, e.y, e.char)
        for b in self._bullets:
            if b.alive:
                self._set_cell(b.x, b.y, "!")
        for eb in self._enemy_bullets:
            if eb.alive:
                self._set_cell(eb.x, eb.y, "↓")
        # Draw turret base
        if 1 < self._player_x < self._WIDTH - 2:
            self._set_cell(
                self._player_x - 1, self._PLAYER_Y + 0, "["
            )
            self._set_cell(
                self._player_x + 1, self._PLAYER_Y + 0, "]"
            )

    def _advance_entities(self) -> None:
        self._entities = [e for e in self._entities if e.alive]

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "─": "wall", "│": "wall",
            "A": "formation leader (2pts)",
            "a": "enemy (1pt)",
            "!": "your bullet", "↓": "enemy bullet",
            "[": "turret base", "]": "turret base",
            " ": "empty",
        }.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        alive = sum(1 for e in self._enemies if e.alive)
        extra = (
            f"Enemies: {alive}  Wave: {self._level}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Defend against descending enemy formations. "
            "Shoot leaders for bonus points. "
            "New waves spawn continuously."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Assault.\n\n"
            "TASK\n"
            "Hold the ground: operate a mobile turret along the bottom "
            "of the screen and shoot down formations of alien attackers "
            "before they reach you.\n\n"
            "BOARD\n"
            "20x20 field with wall borders ('-', '|'). Your turret "
            "patrols along the bottom of the field, with base markers "
            "'[' and ']' drawn in adjacent cells; you appear as an "
            "arrow glyph. Formation leaders 'A' spawn near the top "
            "with wingmen 'a' just below them. Your bullets are '!', "
            "enemy bullets are a down-arrow glyph.\n\n"
            "MECHANICS\n"
            "LEFT / RIGHT move the turret 1 cell. FIRE launches a "
            "bullet straight up (max 2 player bullets alive). Formation "
            "moves together every 3 steps: horizontal bounce at walls "
            "with a 1-row drop. Every 5 steps a random alive enemy "
            "drops a bullet straight down (max 3 enemy bullets). "
            "Formations respawn when cleared or every 30 steps; "
            "clearing a formation advances the wave level.\n\n"
            "SCORING\n"
            "+1 progress unit per kill (formation leader 'A' or "
            "wingman 'a'), regardless of enemy type. No per-step "
            "penalty.\n\n"
            "TERMINATION\n"
            "Single-life: being hit by an enemy bullet or an enemy "
            "reaching your row ends the episode. The episode also "
            "ends after max_turns.\n\n"
            "HUD\n"
            "Shows score, wave/level, alive enemy count.\n\n"
            + self.action_spec.render_for_prompt()
        )
