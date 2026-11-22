"""Atari BattleZone environment.

Top-down tank combat. Destroy enemy tanks on a battlefield.

Gym ID: glyphbench/atari-battlezone-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase, AtariEntity


def _sign(a: int, b: int) -> int:
    return 1 if a > b else (-1 if a < b else 0)

class BattleZoneEnv(AtariBase):
    """BattleZone: top-down tank combat.

    20x20 grid. Destroy enemy tanks that pursue you.
    Obstacles provide cover. Enemies fire back.

    Actions: NOOP, LEFT, RIGHT, UP, DOWN, FIRE
    Pattern D: +1/_WIN_TARGET per enemy tank destroyed (full-scope = 10
    tanks). -1.0 if hit (single-life model).
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "UP", "DOWN", "FIRE"),
        descriptions=(
            "do nothing", "move left", "move right",
            "move up", "move down",
            "fire in last moved direction",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 20
    _MAX_ENEMIES = 3
    _SPAWN_INTERVAL = 20

    # Pattern D full-scope target: 10 tanks destroyed.
    _WIN_TARGET: int = 10
    _DEATH_PENALTY: float = -1.0

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._tanks: list[AtariEntity] = []
        self._bullets: list[AtariEntity] = []
        self._enemy_bullets: list[AtariEntity] = []
        self._step_counter = 0
        self._fire_dx = 0
        self._fire_dy = -1
        self._obstacles: set[tuple[int, int]] = set()
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-battlezone-v0"

    def _reset(self, seed: int) -> GridObservation:
        self._progress_count = 0
        return super()._reset(seed)

    def _reset_pos(self) -> None:
        self._player_x = self._WIDTH // 2
        self._player_y = self._HEIGHT - 3

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._tanks = []
        self._bullets = []
        self._enemy_bullets = []
        self._step_counter = 0
        self._obstacles = set()
        for x in range(self._WIDTH):
            self._set_cell(x, 0, "─")
            self._set_cell(x, self._HEIGHT - 1, "─")
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "│")
            self._set_cell(self._WIDTH - 1, y, "│")
        rng = self.rng
        for _ in range(min(6 + self._level, 15)):
            ox = int(rng.integers(3, self._WIDTH - 3))
            oy = int(rng.integers(3, self._HEIGHT - 3))
            self._obstacles.add((ox, oy))
        self._reset_pos()
        for _ in range(min(2, self._MAX_ENEMIES)):
            self._spawn_enemy()
        self._redraw()

    def _spawn_enemy(self) -> None:
        rng = self.rng
        for _ in range(20):
            ex = int(rng.integers(2, self._WIDTH - 2))
            ey = int(rng.integers(2, self._HEIGHT // 2))
            if ((ex, ey) not in self._obstacles
                    and abs(ex - self._player_x)
                    + abs(ey - self._player_y) > 5):
                t = self._add_entity("tank", "T", ex, ey)
                t.data.update(timer=0, fire_timer=0)
                self._tanks.append(t)
                return

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._step_counter += 1
        nx, ny = self._player_x, self._player_y
        if action_name == "LEFT":
            nx -= 1
            self._fire_dx, self._fire_dy = -1, 0
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT":
            nx += 1
            self._fire_dx, self._fire_dy = 1, 0
            self._player_dir = (1, 0)
        elif action_name == "UP":
            ny -= 1
            self._fire_dx, self._fire_dy = 0, -1
            self._player_dir = (0, -1)
        elif action_name == "DOWN":
            ny += 1
            self._fire_dx, self._fire_dy = 0, 1
            self._player_dir = (0, 1)
        if (0 < nx < self._WIDTH - 1
                and 0 < ny < self._HEIGHT - 1
                and (nx, ny) not in self._obstacles):
            self._player_x, self._player_y = nx, ny
        # Fire
        if action_name == "FIRE" and len(self._bullets) < 3:
            b = self._add_entity(
                "bullet", "*",
                self._player_x + self._fire_dx,
                self._player_y + self._fire_dy,
            )
            b.data.update(bdx=self._fire_dx, bdy=self._fire_dy)
            self._bullets.append(b)
        # Move player bullets
        for b in self._bullets:
            if not b.alive:
                continue
            b.x += b.data["bdx"]
            b.y += b.data["bdy"]
            if (not (0 < b.x < self._WIDTH - 1
                     and 0 < b.y < self._HEIGHT - 1)
                    or (b.x, b.y) in self._obstacles):
                b.alive = False
        # Bullet-tank hits
        for b in self._bullets:
            if not b.alive:
                continue
            for t in self._tanks:
                if t.alive and t.x == b.x and t.y == b.y:
                    t.alive = False
                    b.alive = False
                    self._on_point_scored(2)
                    if self._progress_count < self._WIN_TARGET:
                        reward += 1.0 / self._WIN_TARGET
                        self._progress_count += 1
                    self._message = "Tank destroyed!"
                    break
        # Enemy AI
        for t in self._tanks:
            if not t.alive:
                continue
            t.data["timer"] += 1
            if t.data["timer"] >= 3:
                t.data["timer"] = 0
                dx = _sign(self._player_x, t.x)
                dy = _sign(self._player_y, t.y)
                if abs(self._player_x - t.x) >= abs(
                    self._player_y - t.y
                ):
                    ntx, nty = t.x + dx, t.y
                else:
                    ntx, nty = t.x, t.y + dy
                if (0 < ntx < self._WIDTH - 1
                        and 0 < nty < self._HEIGHT - 1
                        and (ntx, nty) not in self._obstacles):
                    t.x, t.y = ntx, nty
            t.data["fire_timer"] += 1
            if t.data["fire_timer"] >= 8:
                t.data["fire_timer"] = 0
                fdx = _sign(self._player_x, t.x)
                fdy = _sign(self._player_y, t.y)
                if fdx or fdy:
                    if abs(fdx) >= abs(fdy):
                        fdy = 0
                    else:
                        fdx = 0
                    eb = self._add_entity(
                        "enemy_bullet", "o",
                        t.x + fdx, t.y + fdy,
                    )
                    eb.data.update(bdx=fdx, bdy=fdy)
                    self._enemy_bullets.append(eb)
        # Move enemy bullets (Pattern D death penalty if hit)
        for eb in self._enemy_bullets:
            if not eb.alive:
                continue
            eb.x += eb.data["bdx"]
            eb.y += eb.data["bdy"]
            if (not (0 < eb.x < self._WIDTH - 1
                     and 0 < eb.y < self._HEIGHT - 1)
                    or (eb.x, eb.y) in self._obstacles):
                eb.alive = False
            elif (eb.x == self._player_x
                  and eb.y == self._player_y):
                eb.alive = False
                self._on_life_lost()
                reward = self._DEATH_PENALTY
                self._message = "Hit by enemy fire!"
        # Player-tank collision (Pattern D death penalty)
        for t in self._tanks:
            if (t.alive and t.x == self._player_x
                    and t.y == self._player_y):
                t.alive = False
                self._on_life_lost()
                reward = self._DEATH_PENALTY
                self._message = "Rammed by tank!"
        # Win check
        if self._progress_count >= self._WIN_TARGET and not self._game_over:
            self._game_over = True
            info["won"] = True
            self._message = "All tanks destroyed!"
        # Spawn
        alive = [t for t in self._tanks if t.alive]
        if (len(alive) < self._MAX_ENEMIES
                and self._step_counter % self._SPAWN_INTERVAL == 0
                and not self._game_over):
            self._spawn_enemy()
        self._bullets = [b for b in self._bullets if b.alive]
        self._enemy_bullets = [
            e for e in self._enemy_bullets if e.alive
        ]
        self._tanks = [t for t in self._tanks if t.alive]
        self._redraw()
        return reward, self._game_over, info

    def _redraw(self) -> None:
        for y in range(1, self._HEIGHT - 1):
            for x in range(1, self._WIDTH - 1):
                ch = "█" if (x, y) in self._obstacles else " "
                self._set_cell(x, y, ch)
        for t in self._tanks:
            if t.alive:
                self._set_cell(t.x, t.y, "T")
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
            "─": "wall", "│": "wall", "█": "obstacle",
            "T": "enemy tank (2pts)", "*": "your bullet",
            "o": "enemy bullet", " ": "ground",
        }.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        tanks = sum(1 for t in self._tanks if t.alive)
        extra = f"Tanks: {tanks}"
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Destroy enemy tanks on the battlefield. "
            "Use obstacles (#) for cover. "
            "Enemy tanks pursue you and fire back."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari BattleZone.\n\n"
            "TASK\n"
            "Pilot a tank in a top-down battlefield and destroy enemy "
            "tanks while avoiding their fire.\n\n"
            "BOARD\n"
            "20x20 field with wall borders ('-', '|'). Random obstacles "
            "'#' provide cover (6-15 of them, more as level rises). "
            "Enemy tanks 'T' spawn elsewhere on the battlefield. Your "
            "bullets are '*', enemy bullets are 'o'. "
            "Your player glyph is an arrow showing last move direction.\n\n"
            "MECHANICS\n"
            "Each step: a direction key moves you 1 cell and also sets "
            "your firing direction; FIRE launches a bullet from the cell "
            "in front of you. At most 3 player bullets alive at a time. "
            "Enemy tanks update every 3 steps: they move toward you "
            "along the longer axis and fire every 8 steps along the "
            "dominant axis. Bullets travel 1 cell per step and die on "
            "walls, obstacles, or edges.\n\n"
            "SCORING\n"
            "+2 reward for each enemy tank destroyed. No per-step "
            "penalty. Up to 3 enemies alive at once; a new one spawns "
            "every 20 steps.\n\n"
            "TERMINATION\n"
            "Single-life: being hit by an enemy bullet or "
            "colliding with a tank ends the episode. The episode "
            "also ends after max_turns.\n\n"
            "HUD\n"
            "Shows score, level, your facing direction, and "
            "number of enemy tanks alive.\n\n"
            + self.action_spec.render_for_prompt()
        )
