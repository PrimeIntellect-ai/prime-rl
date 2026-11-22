"""Atari Time Pilot environment.

360-degree shooter with enemies approaching from all sides.

Gym ID: glyphbench/atari-timepilot-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase, AtariEntity


class TimePilotEnv(AtariBase):
    """Time Pilot: 360-degree aerial combat.

    20x20 grid. Enemies approach from all edges.
    Player stays roughly centered, enemies swarm in.

    Actions: NOOP, UP, DOWN, LEFT, RIGHT, FIRE
    Pattern D: +1/_WIN_TARGET per enemy or boss destroyed,
    -1 on collision with enemy/boss.
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP", "DOWN", "LEFT", "RIGHT", "FIRE"),
        descriptions=(
            "do nothing",
            "move up",
            "move down",
            "move left",
            "move right",
            "fire in current facing direction",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 20
    _MAX_BULLETS = 3
    _BOSS_THRESHOLD = 10  # enemies to kill per wave

    # Pattern D full-scope: 25 progress events (~5 eras × 5 enemies
    # each, including bosses). Each kill yields +1/25 reward.
    _WIN_TARGET: int = 25
    _DEATH_PENALTY: float = -1.0

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._bullets: list[AtariEntity] = []
        self._enemies: list[AtariEntity] = []
        self._step_counter: int = 0
        self._facing_dx: int = 1
        self._facing_dy: int = 0
        self._kills: int = 0
        self._boss: AtariEntity | None = None
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-timepilot-v0"

    def _reset(self, seed: int):
        self._progress_count = 0
        return super()._reset(seed)

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._bullets = []
        self._enemies = []
        self._step_counter = 0
        self._kills = 0
        self._boss = None
        self._facing_dx = 1
        self._facing_dy = 0

        # Borders
        for x in range(self._WIDTH):
            self._set_cell(x, 0, "─")
            self._set_cell(x, self._HEIGHT - 1, "─")
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "│")
            self._set_cell(self._WIDTH - 1, y, "│")

        # Player at center
        self._player_x = self._WIDTH // 2
        self._player_y = self._HEIGHT // 2

        # Spawn initial enemies
        for _ in range(4):
            self._spawn_enemy()

        self._redraw()

    def _spawn_enemy(self) -> None:
        rng = self.rng
        side = int(rng.integers(0, 4))
        if side == 0:  # top
            ex, ey = int(rng.integers(1, self._WIDTH - 1)), 1
        elif side == 1:  # bottom
            ex, ey = int(rng.integers(1, self._WIDTH - 1)), self._HEIGHT - 2
        elif side == 2:  # left
            ex, ey = 1, int(rng.integers(1, self._HEIGHT - 1))
        else:  # right
            ex, ey = self._WIDTH - 2, int(rng.integers(1, self._HEIGHT - 1))

        e = self._add_entity("enemy", "E", ex, ey)
        e.data["timer"] = 0
        self._enemies.append(e)

    def _spawn_boss(self) -> None:
        rng = self.rng
        bx = int(rng.integers(3, self._WIDTH - 3))
        self._boss = self._add_entity("boss", "B", bx, 1)
        self._boss.data["hp"] = 3
        self._boss.data["timer"] = 0

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._step_counter += 1

        # Move player and update facing direction
        if action_name == "LEFT" and self._player_x > 1:
            self._player_x -= 1
            self._facing_dx, self._facing_dy = -1, 0
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT" and self._player_x < self._WIDTH - 2:
            self._player_x += 1
            self._facing_dx, self._facing_dy = 1, 0
            self._player_dir = (1, 0)
        elif action_name == "UP" and self._player_y > 1:
            self._player_y -= 1
            self._facing_dx, self._facing_dy = 0, -1
            self._player_dir = (0, -1)
        elif action_name == "DOWN" and self._player_y < self._HEIGHT - 2:
            self._player_y += 1
            self._facing_dx, self._facing_dy = 0, 1
            self._player_dir = (0, 1)

        # Fire
        if action_name == "FIRE" and len(self._bullets) < self._MAX_BULLETS:
            bx = self._player_x + self._facing_dx
            by = self._player_y + self._facing_dy
            b = self._add_entity(
                "bullet", "*", bx, by,
                dx=self._facing_dx, dy=self._facing_dy
            )
            self._bullets.append(b)

        # Move bullets
        for b in self._bullets:
            if not b.alive:
                continue
            b.x += b.dx
            b.y += b.dy
            if (
                b.x <= 0 or b.x >= self._WIDTH - 1
                or b.y <= 0 or b.y >= self._HEIGHT - 1
            ):
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
                    self._kills += 1
                    self._message = "Enemy shot!"
                    break
            # Check boss hit
            if (
                b.alive and self._boss is not None
                and self._boss.alive
                and self._boss.x == b.x
                and self._boss.y == b.y
            ):
                b.alive = False
                self._boss.data["hp"] -= 1
                if self._boss.data["hp"] <= 0:
                    self._boss.alive = False
                    self._on_point_scored(1)
                    if self._progress_count < self._WIN_TARGET:
                        reward += 1.0 / self._WIN_TARGET
                        self._progress_count += 1
                    self._message = "Boss destroyed!"
                    self._level += 1
                    saved_progress = self._progress_count
                    self._generate_level(self._level)
                    self._progress_count = saved_progress
                    if self._progress_count >= self._WIN_TARGET:
                        self._game_over = True
                        info["won"] = True
                    return reward, self._game_over, info
                else:
                    self._message = "Boss hit!"
        self._bullets = [b for b in self._bullets if b.alive]

        # Move enemies toward player
        for e in self._enemies:
            if not e.alive:
                continue
            e.data["timer"] += 1
            if e.data["timer"] % 3 == 0:
                if e.x < self._player_x:
                    e.x += 1
                elif e.x > self._player_x:
                    e.x -= 1
                if e.y < self._player_y:
                    e.y += 1
                elif e.y > self._player_y:
                    e.y -= 1
                # Clamp
                e.x = max(1, min(self._WIDTH - 2, e.x))
                e.y = max(1, min(self._HEIGHT - 2, e.y))

        # Move boss
        if self._boss is not None and self._boss.alive:
            self._boss.data["timer"] += 1
            if self._boss.data["timer"] % 2 == 0:
                if self._boss.x < self._player_x:
                    self._boss.x += 1
                elif self._boss.x > self._player_x:
                    self._boss.x -= 1
                if self._boss.y < self._player_y:
                    self._boss.y += 1
                elif self._boss.y > self._player_y:
                    self._boss.y -= 1

        # Player collision -- terminal failure
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

        if (
            self._boss is not None
            and self._boss.alive
            and self._boss.x == self._player_x
            and self._boss.y == self._player_y
        ):
            self._on_life_lost()
            self._message = "Boss collision! Game Over."
            reward = self._DEATH_PENALTY
            return reward, self._game_over, info

        # Spawn more enemies
        self._enemies = [e for e in self._enemies if e.alive]
        if self._step_counter % 8 == 0 and len(self._enemies) < 6:
            self._spawn_enemy()

        # Spawn boss when enough kills
        if (
            self._kills >= self._BOSS_THRESHOLD
            and self._boss is None
        ):
            self._spawn_boss()
            self._message = "Boss incoming!"

        # Win check
        if (
            self._progress_count >= self._WIN_TARGET
            and not self._game_over
        ):
            self._game_over = True
            info["won"] = True
            self._message = "All eras conquered!"

        self._redraw()
        return reward, self._game_over, info

    def _redraw(self) -> None:
        for y in range(1, self._HEIGHT - 1):
            for x in range(1, self._WIDTH - 1):
                self._set_cell(x, y, " ")
        for e in self._enemies:
            if e.alive:
                self._set_cell(e.x, e.y, "E")
        if self._boss is not None and self._boss.alive:
            self._set_cell(self._boss.x, self._boss.y, "B")
        for b in self._bullets:
            if b.alive:
                self._set_cell(b.x, b.y, "*")

    def _advance_entities(self) -> None:
        self._entities = [e for e in self._entities if e.alive]

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "─": "wall",
            "│": "wall",
            "E": "enemy plane",
            "B": "boss (3 HP)",
            "*": "your bullet",
            " ": "sky",
        }.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        boss_hp = "none"
        if (
            self._boss is not None
            and self._boss.alive
        ):
            boss_hp = str(self._boss.data.get("hp", 0))
        extra = (
            f"Kills: {self._kills}/10  Boss: {boss_hp}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Shoot enemies approaching from all directions. "
            "After 10 kills a boss appears -- destroy it to "
            "advance to the next era."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Time Pilot.\n\n"
            "TASK\n"
            "Fly a plane at the center of an arena under attack "
            "from enemies swarming in from all 4 edges. After 10 "
            "kills a boss appears; destroying it advances to the "
            "next era (level).\n\n"
            "BOARD\n"
            "20x20 arena with wall border. Enemies 'E' spawn on a "
            "random side edge. Boss 'B' (HP 3). Your bullets '*'.\n\n"
            "MECHANICS\n"
            "LEFT / RIGHT / UP / DOWN move 1 cell and set your "
            "facing (and firing direction). FIRE launches a bullet "
            "in facing direction (max 3 alive). Enemies update "
            "every 3 steps, moving 1 cell toward you each axis "
            "(8-direction chase). Boss moves every 2 steps. More "
            "enemies spawn every 8 steps (cap 6).\n\n"
            "SCORING\n"
            "Pattern D: +1/25 reward per enemy or boss "
            "destroyed. Killing a boss advances the era (level). "
            "-1 reward on collision with any enemy or boss. "
            "Cumulative reward bound: [-1, +1].\n\n"
            "TERMINATION\n"
            "Single-life: any collision ends with -1. Reaching 25 "
            "kills ends with cumulative +1. Episode also ends "
            "after max_turns.\n\n"
            "HUD\n"
            "Shows score, level/era, facing, kills toward "
            "10 (boss threshold), and boss HP.\n\n"
            + self.action_spec.render_for_prompt()
        )
