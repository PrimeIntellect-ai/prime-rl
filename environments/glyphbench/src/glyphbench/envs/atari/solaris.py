"""Atari Solaris environment.

Space combat and exploration across sectors.

Gym ID: glyphbench/atari-solaris-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase, AtariEntity


def _sign(a: int, b: int) -> int:
    return 1 if a > b else (-1 if a < b else 0)

class SolarisEnv(AtariBase):
    """Solaris: space combat and sector exploration.

    20x20 grid. Navigate between sectors, fight enemies,
    protect allied bases. WARP moves to adjacent sectors.

    Actions: NOOP, UP, DOWN, LEFT, RIGHT, FIRE, WARP
    Pattern D: +1/_WIN_TARGET per enemy destroyed,
    -1 on contact with enemy fire.
    """

    action_spec = ActionSpec(
        names=(
            "NOOP", "UP", "DOWN", "LEFT",
            "RIGHT", "FIRE", "WARP",
        ),
        descriptions=(
            "do nothing", "move up", "move down",
            "move left", "move right",
            "fire forward", "warp to next sector",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 20
    _NUM_SECTORS = 4

    # Pattern D full-scope: 16 enemy kills (~4 per sector × 4
    # sectors). Each kill yields +1/16 reward; being hit by
    # enemy fire ends the episode with -1.
    _WIN_TARGET: int = 16
    _DEATH_PENALTY: float = -1.0

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._enemies: list[AtariEntity] = []
        self._bullets: list[AtariEntity] = []
        self._enemy_bullets: list[AtariEntity] = []
        self._step_counter = 0
        self._sector = 0
        self._sectors_cleared: set[int] = set()
        self._fire_dx = 0
        self._fire_dy = -1
        self._bases: list[AtariEntity] = []
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-solaris-v0"

    def _reset(self, seed: int):
        self._progress_count = 0
        self._sector = 0
        self._sectors_cleared = set()
        self._fire_dx, self._fire_dy = 0, -1
        return super()._reset(seed)

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._enemies = []
        self._bullets = []
        self._enemy_bullets = []
        self._bases = []
        self._step_counter = 0
        for x in range(self._WIDTH):
            self._set_cell(x, 0, "─")
            self._set_cell(x, self._HEIGHT - 1, "─")
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "│")
            self._set_cell(self._WIDTH - 1, y, "│")
        rng = self.rng
        # Stars
        for _ in range(10):
            self._set_cell(
                int(rng.integers(2, self._WIDTH - 2)),
                int(rng.integers(2, self._HEIGHT - 2)), "·",
            )
        self._player_x = self._WIDTH // 2
        self._player_y = self._HEIGHT - 4
        # Enemies for this sector
        if self._sector not in self._sectors_cleared:
            for _ in range(min(3 + self._level + self._sector, 8)):
                e = self._add_entity(
                    "enemy", "E",
                    int(rng.integers(2, self._WIDTH - 2)),
                    int(rng.integers(2, self._HEIGHT // 2)),
                )
                e.data.update(timer=0, fire_cd=0)
                self._enemies.append(e)
        # Allied base in even sectors
        if self._sector % 2 == 0:
            base = self._add_entity(
                "base", "B",
                int(rng.integers(5, self._WIDTH - 5)),
                int(rng.integers(
                    self._HEIGHT // 3, 2 * self._HEIGHT // 3
                )),
            )
            base.data["hp"] = 3
            self._bases.append(base)
        # Sector indicator
        for x in range(1, min(self._sector + 2, self._WIDTH - 1)):
            self._set_cell(x, 0, "=")
        self._redraw()

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._step_counter += 1
        nx, ny = self._player_x, self._player_y
        if action_name == "UP":
            ny -= 1
            self._fire_dx, self._fire_dy = 0, -1
            self._player_dir = (0, -1)
        elif action_name == "DOWN":
            ny += 1
            self._fire_dx, self._fire_dy = 0, 1
            self._player_dir = (0, 1)
        elif action_name == "LEFT":
            nx -= 1
            self._fire_dx, self._fire_dy = -1, 0
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT":
            nx += 1
            self._fire_dx, self._fire_dy = 1, 0
            self._player_dir = (1, 0)
        if 0 < nx < self._WIDTH - 1 and 0 < ny < self._HEIGHT - 1:
            self._player_x, self._player_y = nx, ny
        # Warp
        if action_name == "WARP":
            old = self._sector
            self._sector = (self._sector + 1) % self._NUM_SECTORS
            self._message = f"Warped: sector {old} -> {self._sector}"
            self._generate_level(self._level * 100 + self._sector)
            return reward, False, info
        # Fire
        if action_name == "FIRE" and len(self._bullets) < 3:
            b = self._add_entity(
                "bullet", "*",
                self._player_x + self._fire_dx,
                self._player_y + self._fire_dy,
            )
            b.data.update(bdx=self._fire_dx, bdy=self._fire_dy)
            self._bullets.append(b)
        # Move bullets
        for b in self._bullets:
            if not b.alive:
                continue
            b.x += b.data["bdx"]
            b.y += b.data["bdy"]
            if not (0 < b.x < self._WIDTH - 1
                    and 0 < b.y < self._HEIGHT - 1):
                b.alive = False
        # Bullet-enemy hits
        for b in self._bullets:
            if not b.alive:
                continue
            for e in self._enemies:
                if e.alive and e.x == b.x and e.y == b.y:
                    e.alive = False
                    b.alive = False
                    self._on_point_scored(1)
                    if self._progress_count < self._WIN_TARGET:
                        reward += 1.0 / self._WIN_TARGET
                        self._progress_count += 1
                    self._message = "Enemy destroyed!"
                    break
        # Enemy AI
        for e in self._enemies:
            if not e.alive:
                continue
            e.data["timer"] += 1
            if e.data["timer"] >= 3:
                e.data["timer"] = 0
                tx, ty = self._player_x, self._player_y
                if (self._bases and self._bases[0].alive
                        and self.rng.random() < 0.3):
                    tx, ty = self._bases[0].x, self._bases[0].y
                dx = _sign(tx, e.x)
                dy = _sign(ty, e.y)
                if abs(dx) >= abs(dy):
                    ntx = e.x + dx
                    if 0 < ntx < self._WIDTH - 1:
                        e.x = ntx
                else:
                    nty = e.y + dy
                    if 0 < nty < self._HEIGHT - 1:
                        e.y = nty
            e.data["fire_cd"] += 1
            if e.data["fire_cd"] >= 10:
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
                    eb.data.update(bdx=fdx, bdy=fdy)
                    self._enemy_bullets.append(eb)
        # Move enemy bullets
        for eb in self._enemy_bullets:
            if not eb.alive:
                continue
            eb.x += eb.data["bdx"]
            eb.y += eb.data["bdy"]
            if not (0 < eb.x < self._WIDTH - 1
                    and 0 < eb.y < self._HEIGHT - 1):
                eb.alive = False
            elif (eb.x == self._player_x
                  and eb.y == self._player_y):
                eb.alive = False
                self._on_life_lost()
                self._message = "Hit by enemy fire! Game Over."
                reward += self._DEATH_PENALTY
                return reward, self._game_over, info
        # Base damage
        for base in self._bases:
            if not base.alive:
                continue
            for e in self._enemies:
                if e.alive and e.x == base.x and e.y == base.y:
                    base.data["hp"] -= 1
                    e.alive = False
                    if base.data["hp"] <= 0:
                        base.alive = False
                        self._message = "Base destroyed!"
        # Sector cleared (no extra reward — kills already counted)
        alive_enemies = [e for e in self._enemies if e.alive]
        if (not alive_enemies
                and self._sector not in self._sectors_cleared):
            self._sectors_cleared.add(self._sector)
            self._message = f"Sector {self._sector} cleared!"
            if len(self._sectors_cleared) >= self._NUM_SECTORS:
                self._level += 1
                self._sectors_cleared = set()
                self._message = "All sectors clear! Level up!"

        # Win check
        if (
            self._progress_count >= self._WIN_TARGET
            and not self._game_over
        ):
            self._game_over = True
            info["won"] = True
            self._message = "Star system secured!"
        # Cleanup
        self._bullets = [b for b in self._bullets if b.alive]
        self._enemy_bullets = [
            eb for eb in self._enemy_bullets if eb.alive
        ]
        self._enemies = [e for e in self._enemies if e.alive]
        self._redraw()
        info["sector"] = self._sector
        info["sectors_cleared"] = len(self._sectors_cleared)
        return reward, self._game_over, info

    def _redraw(self) -> None:
        for y in range(1, self._HEIGHT - 1):
            for x in range(1, self._WIDTH - 1):
                self._set_cell(x, y, " ")
        for e in self._enemies:
            if e.alive:
                self._set_cell(e.x, e.y, "E")
        for base in self._bases:
            if base.alive:
                self._set_cell(base.x, base.y, "B")
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
            "─": "wall", "│": "wall", "=": "sector indicator",
            "·": "star", "E": "enemy (30pts)",
            "B": "allied base (protect!)",
            "*": "your bullet", "o": "enemy bullet",
            " ": "space",
        }.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        enemies = sum(
            1 for e in self._enemies if e.alive
        )
        extra = (
            f"Sector: {self._sector}"
            f"  Enemies: {enemies}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Explore sectors and destroy enemies. "
            "WARP to travel between sectors. "
            "Protect allied bases (B). Clear all sectors to advance."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Solaris.\n\n"
            "TASK\n"
            "Hop between 4 sectors of a star system, clearing "
            "enemies from each while protecting allied bases. "
            "Clearing all 4 sectors advances the level.\n\n"
            "BOARD\n"
            "20x20 space scene, bordered by walls. Stars '.' as "
            "decoration. Sector indicator bar '=' on top row shows "
            "current sector. Enemies 'E', allied bases 'B' (in "
            "even sectors, HP=3). Your bullets '*', enemy bullets "
            "'o'. You are an arrow glyph near bottom.\n\n"
            "MECHANICS\n"
            "UP/DOWN/LEFT/RIGHT move 1 cell and set facing. FIRE "
            "launches a bullet in facing direction (max 3 alive). "
            "WARP moves you to sector (s+1) mod 4 and regenerates "
            "that sector (its enemies respawn unless already "
            "cleared). Enemies move every 3 steps toward you "
            "(or toward the base 30 percent of the time) along the "
            "dominant axis; they fire every 10 steps.\n\n"
            "SCORING\n"
            "Pattern D: +1/16 reward per enemy destroyed (full "
            "scope ~ 4 enemies × 4 sectors). Sector clears emit "
            "no extra reward. -1 reward when hit by enemy fire. "
            "Cumulative reward bound: [-1, +1].\n\n"
            "TERMINATION\n"
            "Single-life: hit by an enemy bullet ends the episode "
            "with -1. Destroying 16 enemies ends with cumulative "
            "+1. Episode also ends after max_turns.\n\n"
            "HUD\n"
            "Shows score, level, facing, current sector, "
            "enemies remaining.\n\n"
            + self.action_spec.render_for_prompt()
        )
