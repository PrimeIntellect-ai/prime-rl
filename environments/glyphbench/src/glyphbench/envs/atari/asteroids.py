"""Atari Asteroids environment.

Top-down space shooter. Destroy asteroids that split into smaller ones.

Gym ID: glyphbench/atari-asteroids-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase, AtariEntity


class AsteroidsEnv(AtariBase):
    """Asteroids: destroy drifting rocks in space.

    20x20 grid. Ship rotates and thrusts. Asteroids drift and
    split when shot (large->medium->small->gone).
    Screen wraps on all edges.

    Actions: NOOP, LEFT, RIGHT, THRUST, FIRE
    Pattern A: +1/_WIN_TARGET per asteroid destroyed (full-scope =
    5 waves x 5 rocks). No failure penalty.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "THRUST", "FIRE"),
        descriptions=(
            "do nothing",
            "rotate left",
            "rotate right",
            "thrust forward",
            "fire bullet in facing direction",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 20
    _DIRS = ((0, -1), (1, -1), (1, 0), (1, 1),
             (0, 1), (-1, 1), (-1, 0), (-1, -1))
    _FACING_CHARS = ("↑", "/", "→", "\\", "↓", "/", "←", "\\")
    _INIT_ASTEROIDS = 4

    # Pattern A full-scope target: 25 rocks destroyed (5 waves x 5 rocks).
    _WIN_TARGET: int = 25

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._facing: int = 0
        self._bullets: list[AtariEntity] = []
        self._asteroids: list[AtariEntity] = []
        self._step_counter: int = 0
        self._ship_dx: int = 0
        self._ship_dy: int = 0
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-asteroids-v0"

    def _reset(self, seed: int) -> GridObservation:
        self._progress_count = 0
        return super()._reset(seed)

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._bullets = []
        self._asteroids = []
        self._step_counter = 0
        self._facing = 0
        self._ship_dx = 0
        self._ship_dy = 0

        self._player_x = self._WIDTH // 2
        self._player_y = self._HEIGHT // 2

        rng = self.rng
        n = self._INIT_ASTEROIDS + self._level - 1
        for _ in range(n):
            while True:
                ax = int(rng.integers(1, self._WIDTH - 1))
                ay = int(rng.integers(1, self._HEIGHT - 1))
                dist = abs(ax - self._player_x) + abs(
                    ay - self._player_y
                )
                if dist > 5:
                    break
            adx = int(rng.integers(-1, 2))
            ady = int(rng.integers(-1, 2))
            if adx == 0 and ady == 0:
                adx = 1
            a = self._add_entity("asteroid", "O", ax, ay)
            a.data["size"] = 3
            a.data["adx"] = adx
            a.data["ady"] = ady
            a.data["timer"] = 0
            self._asteroids.append(a)

        self._redraw()

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._step_counter += 1

        # Rotate
        if action_name == "LEFT":
            self._facing = (self._facing - 1) % 8
        elif action_name == "RIGHT":
            self._facing = (self._facing + 1) % 8

        # Update player_dir from facing
        fdx, fdy = self._DIRS[self._facing]
        # Map 8-dir to 4-dir for player char
        if abs(fdx) >= abs(fdy):
            self._player_dir = (1 if fdx > 0 else -1, 0) if fdx != 0 else (0, 0)
        else:
            self._player_dir = (0, 1 if fdy > 0 else -1) if fdy != 0 else (0, 0)

        # Thrust
        if action_name == "THRUST":
            dx, dy = self._DIRS[self._facing]
            self._ship_dx = dx
            self._ship_dy = dy

        # Move ship
        self._player_x = (
            (self._player_x + self._ship_dx) % self._WIDTH
        )
        self._player_y = (
            (self._player_y + self._ship_dy) % self._HEIGHT
        )
        # Friction
        self._ship_dx = 0
        self._ship_dy = 0

        # Fire
        if action_name == "FIRE" and len(self._bullets) < 4:
            dx, dy = self._DIRS[self._facing]
            bx = (self._player_x + dx) % self._WIDTH
            by = (self._player_y + dy) % self._HEIGHT
            b = self._add_entity("bullet", "·", bx, by)
            b.data["bdx"] = dx
            b.data["bdy"] = dy
            b.data["life"] = 12
            self._bullets.append(b)

        # Move bullets
        for b in self._bullets:
            if not b.alive:
                continue
            b.data["life"] -= 1
            if b.data["life"] <= 0:
                b.alive = False
                continue
            b.x = (b.x + b.data["bdx"]) % self._WIDTH
            b.y = (b.y + b.data["bdy"]) % self._HEIGHT

        # Move asteroids
        for a in self._asteroids:
            if not a.alive:
                continue
            a.data["timer"] += 1
            spd = max(1, 4 - a.data["size"])
            if a.data["timer"] >= spd:
                a.data["timer"] = 0
                a.x = (a.x + a.data["adx"]) % self._WIDTH
                a.y = (a.y + a.data["ady"]) % self._HEIGHT

        # Bullet-asteroid collisions
        new_asteroids: list[AtariEntity] = []
        for b in self._bullets:
            if not b.alive:
                continue
            for a in self._asteroids:
                if not a.alive:
                    continue
                if a.x == b.x and a.y == b.y:
                    b.alive = False
                    a.alive = False
                    sz = a.data["size"]
                    pts = {3: 1, 2: 2, 1: 3}.get(sz, 1)
                    self._on_point_scored(pts)
                    if self._progress_count < self._WIN_TARGET:
                        reward += 1.0 / self._WIN_TARGET
                        self._progress_count += 1
                    self._message = "Asteroid!"
                    if sz > 1:
                        for _ in range(2):
                            rng = self.rng
                            ndx = int(rng.integers(-1, 2))
                            ndy = int(rng.integers(-1, 2))
                            if ndx == 0 and ndy == 0:
                                ndx = 1
                            ch = "o" if sz - 1 == 2 else "·"
                            na = self._add_entity(
                                "asteroid", ch, a.x, a.y
                            )
                            na.data["size"] = sz - 1
                            na.data["adx"] = ndx
                            na.data["ady"] = ndy
                            na.data["timer"] = 0
                            new_asteroids.append(na)
                    break

        self._asteroids.extend(new_asteroids)

        # Player-asteroid collision
        for a in self._asteroids:
            if (
                a.alive
                and a.x == self._player_x
                and a.y == self._player_y
            ):
                a.alive = False
                self._on_life_lost()
                self._message = "Crashed into asteroid!"
                self._player_x = self._WIDTH // 2
                self._player_y = self._HEIGHT // 2
                self._ship_dx = 0
                self._ship_dy = 0

        # Cleanup
        self._bullets = [b for b in self._bullets if b.alive]
        self._asteroids = [a for a in self._asteroids if a.alive]

        # Win check
        if self._progress_count >= self._WIN_TARGET and not self._game_over:
            self._game_over = True
            info["won"] = True
            self._message = "All asteroids cleared!"

        # Level clear
        if len(self._asteroids) == 0 and not self._game_over:
            self._level += 1
            self._message = "Level cleared!"
            self._generate_level(self._level)

        self._redraw()
        return reward, self._game_over, info

    def _redraw(self) -> None:
        for y in range(self._HEIGHT):
            for x in range(self._WIDTH):
                self._set_cell(x, y, " ")
        for a in self._asteroids:
            if a.alive:
                self._set_cell(a.x, a.y, a.char)
        for b in self._bullets:
            if b.alive:
                self._set_cell(b.x, b.y, "!")

    _FACING_NAMES = (
        "up", "up-right", "right", "down-right",
        "down", "down-left", "left", "up-left",
    )

    def _render_current_observation(self, **kw: Any):  # type: ignore[override]
        obs = super()._render_current_observation()
        facing_vec = self._DIRS[self._facing]
        n_ast = len([
            a for a in self._asteroids if a.alive
        ])
        parts = [
            f"Ship vel: ({self._ship_dx},{self._ship_dy})",
            f"Asteroids: {n_ast}",
        ]
        if facing_vec not in self._DIR_CHARS:
            parts.insert(0, f"Ship facing: {self._FACING_NAMES[self._facing]}")
        extra = "  ".join(parts)
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _advance_entities(self) -> None:
        self._entities = [e for e in self._entities if e.alive]

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "O": "large asteroid (1pt)",
            "o": "medium asteroid (2pts)",
            "·": "small asteroid (3pts)",
            "!": "your bullet",
            " ": "space",
        }.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "Destroy asteroids by shooting them. "
            "Large asteroids split into medium, medium into small. "
            "Rotate with LEFT/RIGHT, THRUST to move, FIRE to shoot. "
            "Screen wraps around all edges."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Asteroids.\n\n"
            "TASK\n"
            "Destroy all asteroids in the field; when one is shot it may "
            "split into smaller, faster pieces. Clearing the field advances "
            "the level and spawns more starting asteroids.\n\n"
            "BOARD\n"
            "20x20 wrap-around space (no walls; leaving an edge reappears "
            "on the opposite side). Your ship is drawn as an arrow pointing "
            "in one of 8 compass directions. Large asteroids are 'O', "
            "medium 'o', small '.' (period). Bullets are '!'.\n\n"
            "MECHANICS\n"
            "LEFT / RIGHT rotate by 45 degrees (8 facings). THRUST moves "
            "you one cell in the current facing (and applies no inertia, "
            "so thrusting only moves during that step). FIRE launches a "
            "bullet in the facing direction; up to 4 bullets can be alive "
            "and each expires after 12 steps. Asteroids drift 1 cell per N "
            "steps where N = max(1, 4 - size); smaller rocks are faster. "
            "A hit large rock spawns two medium rocks with random "
            "velocities; a medium spawns two small; a small disappears.\n\n"
            "SCORING\n"
            "+1 progress unit per asteroid destroyed, regardless "
            "of size. No per-step penalty. Level clear awards no "
            "bonus but restarts with level+1 asteroids.\n\n"
            "TERMINATION\n"
            "Single-life: colliding with any asteroid destroys "
            "your ship and ends the episode. The episode also "
            "ends after max_turns.\n\n"
            "HUD\n"
            "Shows score, level, ship facing direction, last "
            "velocity, and number of asteroids remaining.\n\n"
            + self.action_spec.render_for_prompt()
        )
