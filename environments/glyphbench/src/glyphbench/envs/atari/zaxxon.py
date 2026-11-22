"""Atari Zaxxon environment.

Top-down scrolling shooter with fuel management.

Gym ID: glyphbench/atari-zaxxon-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase, AtariEntity


class ZaxxonEnv(AtariBase):
    """Zaxxon: scrolling shooter with fuel.

    20x24 grid. Battlefield scrolls downward. Shoot enemies,
    dodge walls, manage fuel. Destroy fuel depots to refuel.

    Actions: NOOP, LEFT, RIGHT, UP, DOWN, FIRE
    Pattern D: +1/_WIN_TARGET per progress unit (enemy
    destroyed or fuel depot collected). -1 on collision /
    out-of-fuel.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "UP", "DOWN", "FIRE"),
        descriptions=(
            "do nothing", "move left", "move right",
            "move up (forward)", "move down (back)",
            "fire forward",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 24
    _MAX_FUEL = 100
    _FUEL_BURN = 1

    # Pattern D full-scope: 20 progress events (~5 events per
    # level × 4 levels). Each yields +1/20.
    _WIN_TARGET: int = 20
    _DEATH_PENALTY: float = -1.0

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._enemies: list[AtariEntity] = []
        self._bullets: list[AtariEntity] = []
        self._walls: set[tuple[int, int]] = set()
        self._fuel = self._MAX_FUEL
        self._step_counter = 0
        self._scroll_timer = 0
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-zaxxon-v0"

    def _reset(self, seed: int):
        self._progress_count = 0
        return super()._reset(seed)

    def _reset_pos(self) -> None:
        self._player_x = self._WIDTH // 2
        self._player_y = self._HEIGHT - 4

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._enemies = []
        self._bullets = []
        self._walls = set()
        self._step_counter = 0
        self._scroll_timer = 0
        self._fuel = self._MAX_FUEL
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "│")
            self._set_cell(self._WIDTH - 1, y, "│")
        for x in range(self._WIDTH):
            self._set_cell(x, self._HEIGHT - 1, "─")
        self._reset_pos()
        rng = self.rng
        for y in range(1, self._HEIGHT // 2):
            if y <= 0 or y >= self._HEIGHT - 1:
                continue
            if rng.random() < 0.15:
                wx = int(rng.integers(2, self._WIDTH - 2))
                for dx in range(int(rng.integers(2, 5))):
                    px = wx + dx
                    if 0 < px < self._WIDTH - 1:
                        self._walls.add((px, y))
            if rng.random() < 0.2:
                ex = int(rng.integers(2, self._WIDTH - 2))
                if (ex, y) not in self._walls:
                    is_fuel = rng.random() < 0.3
                    ch = "F" if is_fuel else "E"
                    tp = "fuel" if is_fuel else "enemy"
                    e = self._add_entity(tp, ch, ex, y)
                    e.data["hp"] = 1
                    self._enemies.append(e)
        self._redraw()

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._step_counter += 1
        self._fuel -= self._FUEL_BURN
        if self._fuel <= 0:
            self._on_life_lost()
            self._message = "Out of fuel! Game Over."
            reward = self._DEATH_PENALTY
            return reward, self._game_over, info
        # Move player
        nx, ny = self._player_x, self._player_y
        if action_name == "LEFT":
            nx -= 1
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT":
            nx += 1
            self._player_dir = (1, 0)
        elif action_name == "UP":
            ny -= 1
            self._player_dir = (0, -1)
        elif action_name == "DOWN":
            ny += 1
            self._player_dir = (0, 1)
        if (0 < nx < self._WIDTH - 1
                and 0 < ny < self._HEIGHT - 1
                and (nx, ny) not in self._walls):
            self._player_x, self._player_y = nx, ny
        if (self._player_x, self._player_y) in self._walls:
            self._on_life_lost()
            self._message = "Hit a wall! Game Over."
            reward = self._DEATH_PENALTY
            return reward, self._game_over, info
        # Fire
        if action_name == "FIRE" and len(self._bullets) < 3:
            b = self._add_entity(
                "bullet", "*",
                self._player_x, self._player_y - 1,
            )
            b.data["bdy"] = -1
            self._bullets.append(b)
        # Move bullets
        for b in self._bullets:
            if not b.alive:
                continue
            b.y += b.data["bdy"]
            if b.y <= 0:
                b.alive = False
            elif (b.x, b.y) in self._walls:
                b.alive = False
                self._walls.discard((b.x, b.y))
        # Bullet-enemy hits
        for b in self._bullets:
            if not b.alive:
                continue
            for e in self._enemies:
                if e.alive and e.x == b.x and e.y == b.y:
                    b.alive = False
                    e.alive = False
                    if e.etype == "fuel":
                        self._fuel = min(
                            self._MAX_FUEL, self._fuel + 30
                        )
                        self._on_point_scored(1)
                        if self._progress_count < self._WIN_TARGET:
                            reward += 1.0 / self._WIN_TARGET
                            self._progress_count += 1
                        self._message = "Fuel depot destroyed!"
                    else:
                        self._on_point_scored(1)
                        if self._progress_count < self._WIN_TARGET:
                            reward += 1.0 / self._WIN_TARGET
                            self._progress_count += 1
                        self._message = "Enemy down!"
                    break
        # Player-enemy collision
        for e in self._enemies:
            if not e.alive:
                continue
            if e.x == self._player_x and e.y == self._player_y:
                e.alive = False
                if e.etype == "fuel":
                    self._fuel = min(
                        self._MAX_FUEL, self._fuel + 30
                    )
                    self._on_point_scored(1)
                    if self._progress_count < self._WIN_TARGET:
                        reward += 1.0 / self._WIN_TARGET
                        self._progress_count += 1
                    self._message = "Fuel collected!"
                else:
                    self._on_life_lost()
                    self._message = "Hit by enemy! Game Over."
                    reward = self._DEATH_PENALTY
                    return reward, self._game_over, info
        # Scroll
        self._scroll_timer += 1
        spd = max(1, 3 - self._level // 2)
        if self._scroll_timer >= spd:
            self._scroll_timer = 0
            self._scroll_down()
            if self._game_over:
                # Wall-crush during scroll: terminal failure.
                reward = self._DEATH_PENALTY
                self._bullets = [b for b in self._bullets if b.alive]
                self._enemies = [e for e in self._enemies if e.alive]
                info["fuel"] = self._fuel
                return reward, self._game_over, info

        # Win check
        if (
            self._progress_count >= self._WIN_TARGET
            and not self._game_over
        ):
            self._game_over = True
            info["won"] = True
            self._message = "Battlefield cleared!"

        self._bullets = [b for b in self._bullets if b.alive]
        self._enemies = [e for e in self._enemies if e.alive]
        self._redraw()
        info["fuel"] = self._fuel
        return reward, self._game_over, info

    def _scroll_down(self) -> None:
        new_walls: set[tuple[int, int]] = set()
        for wx, wy in self._walls:
            if wy + 1 < self._HEIGHT - 1:
                new_walls.add((wx, wy + 1))
        self._walls = new_walls
        for e in self._enemies:
            if e.alive:
                e.y += 1
                if e.y >= self._HEIGHT - 1:
                    e.alive = False
        rng = self.rng
        if rng.random() < 0.3:
            wx = int(rng.integers(2, self._WIDTH - 2))
            for dx in range(int(rng.integers(2, 4))):
                px = wx + dx
                if 0 < px < self._WIDTH - 1:
                    self._walls.add((px, 1))
        if rng.random() < 0.25:
            ex = int(rng.integers(2, self._WIDTH - 2))
            if (ex, 1) not in self._walls:
                is_fuel = rng.random() < 0.3
                ch = "F" if is_fuel else "E"
                tp = "fuel" if is_fuel else "enemy"
                e = self._add_entity(tp, ch, ex, 1)
                e.data["hp"] = 1
                self._enemies.append(e)
        if (self._player_x, self._player_y) in self._walls:
            self._on_life_lost()
            self._message = "Crushed by wall! Game Over."

    def _redraw(self) -> None:
        for y in range(1, self._HEIGHT - 1):
            for x in range(1, self._WIDTH - 1):
                self._set_cell(x, y, " ")
        for wx, wy in self._walls:
            self._set_cell(wx, wy, "█")
        for e in self._enemies:
            if e.alive:
                self._set_cell(e.x, e.y, e.char)
        for b in self._bullets:
            if b.alive:
                self._set_cell(b.x, b.y, "*")

    def _advance_entities(self) -> None:
        self._entities = [e for e in self._entities if e.alive]

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "─": "wall", "│": "wall", "█": "obstacle",
            "E": "enemy (20pts)",
            "F": "fuel depot (+30 fuel, 10pts)",
            "*": "your bullet", " ": "empty",
        }.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        extra = f"Fuel: {self._fuel}"
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Fly through the scrolling battlefield. "
            "Shoot enemies (E) and collect fuel depots (F). "
            "Avoid walls (#) and manage your fuel."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Zaxxon.\n\n"
            "TASK\n"
            "Fly through a vertically-scrolling battlefield: shoot "
            "enemies, grab fuel, and dodge walls. Survive as long "
            "as possible before running out of fuel or lives.\n\n"
            "BOARD\n"
            "20x24 battlefield. Walls '|' on sides and '-' on "
            "bottom. Wall obstacles '#' scatter in the upper half. "
            "Enemies 'E' and fuel depots 'F' scattered. Your "
            "bullets '*'. You are an arrow glyph near the bottom.\n\n"
            "MECHANICS\n"
            "LEFT / RIGHT / UP / DOWN move 1 cell (blocked by "
            "walls / obstacles). FIRE launches a bullet upward "
            "(max 3 alive). Every max(1, 3 - level/2) steps the "
            "world scrolls down: walls and enemies shift 1 row "
            "down. New walls and entities spawn at row 1. Fuel "
            "decreases 1 per step.\n\n"
            "SCORING\n"
            "Pattern D: +1/20 reward per progress event (enemy "
            "'E' destroyed or fuel depot 'F' collected -- "
            "fuel depots also refill +30 fuel, capped at 100). "
            "-1 reward on hitting a wall, colliding with an "
            "enemy, being crushed by a scrolling wall, or running "
            "out of fuel. Cumulative reward bound: [-1, +1].\n\n"
            "TERMINATION\n"
            "Single-life: any death event ends with -1. Reaching "
            "20 progress events ends with cumulative +1. Episode "
            "also ends after max_turns.\n\n"
            "HUD\n"
            "Shows score, level, and fuel remaining.\n\n"
            + self.action_spec.render_for_prompt()
        )
