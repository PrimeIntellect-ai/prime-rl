"""Atari River Raid environment.

Vertical scrolling river shooter with fuel management.

Gym ID: glyphbench/atari-riverraid-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec

from .base import AtariBase, AtariEntity


class RiverRaidEnv(AtariBase):
    """River Raid: fly up a scrolling river, shoot enemies, manage fuel.

    20x24 viewport. River scrolls down; enemies, fuel depots appear.
    Running out of fuel or crashing ends the game.

    Actions: NOOP, UP, DOWN, LEFT, RIGHT, FIRE
    Pattern D: +1/_WIN_TARGET per progress unit (enemy destroyed
    or fuel depot collected). -1 on collision / out-of-fuel.
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP", "DOWN", "LEFT", "RIGHT", "FIRE"),
        descriptions=(
            "do nothing",
            "accelerate (scroll faster)",
            "decelerate",
            "move left",
            "move right",
            "fire a bullet forward",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 24
    _PLAYER_Y = 20
    _RIVER_LEFT = 4
    _RIVER_RIGHT = 15
    _MAX_FUEL = 100
    _MAX_BULLETS = 2

    # Pattern D full-scope: 31 progress events (~30 obstacles plus boss).
    _WIN_TARGET: int = 31
    _DEATH_PENALTY: float = -1.0

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._bullets: list[AtariEntity] = []
        self._obstacles: list[AtariEntity] = []
        self._fuel: int = self._MAX_FUEL
        self._scroll_timer: int = 0
        self._scroll_speed: int = 2
        self._step_counter: int = 0
        self._river_l: list[int] = []
        self._river_r: list[int] = []
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-riverraid-v0"

    def _reset(self, seed: int):
        self._progress_count = 0
        return super()._reset(seed)

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._bullets = []
        self._obstacles = []
        self._step_counter = 0
        self._scroll_timer = 0
        self._scroll_speed = 2
        self._fuel = self._MAX_FUEL

        # Init river banks
        self._river_l = [self._RIVER_LEFT] * self._HEIGHT
        self._river_r = [self._RIVER_RIGHT] * self._HEIGHT

        self._player_x = self._WIDTH // 2
        self._player_y = self._PLAYER_Y

        # Spawn some initial enemies
        rng = self.rng
        for row in range(2, self._HEIGHT - 5, 3):
            if int(rng.integers(0, 3)) > 0:
                ex = int(rng.integers(
                    self._river_l[row] + 1,
                    self._river_r[row]
                ))
                etype = "ship" if int(rng.integers(0, 2)) == 0 else "heli"
                ch = "S" if etype == "ship" else "H"
                e = self._add_entity(etype, ch, ex, row)
                e.data["dx"] = 1 if int(rng.integers(0, 2)) == 0 else -1
                e.data["timer"] = 0
                self._obstacles.append(e)
            # Occasionally place fuel depot
            if int(rng.integers(0, 5)) == 0:
                fx = int(rng.integers(
                    self._river_l[row] + 1,
                    self._river_r[row]
                ))
                f = self._add_entity("fuel", "F", fx, row)
                self._obstacles.append(f)

        self._redraw()

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
        elif action_name == "UP":
            self._scroll_speed = min(3, self._scroll_speed + 1)
        elif action_name == "DOWN":
            self._scroll_speed = max(1, self._scroll_speed - 1)

        # Fire
        if action_name == "FIRE" and len(self._bullets) < self._MAX_BULLETS:
            b = self._add_entity(
                "bullet", "!", self._player_x,
                self._player_y - 1, dy=-1
            )
            self._bullets.append(b)

        # Scroll (move obstacles down)
        self._scroll_timer += 1
        do_scroll = self._scroll_timer >= (4 - self._scroll_speed)
        if do_scroll:
            self._scroll_timer = 0
            for obs in self._obstacles:
                if obs.alive:
                    obs.y += 1
                    if obs.y >= self._HEIGHT - 1:
                        obs.alive = False

            # Randomly spawn new obstacles at top
            rng = self.rng
            rl, rr = self._river_l[2], self._river_r[2]
            if int(rng.integers(0, 3)) > 0 and rr > rl + 1:
                ex = int(rng.integers(rl + 1, rr))
                roll = int(rng.integers(0, 6))
                if roll < 4:
                    nm = "ship" if roll < 2 else "heli"
                    ch = "S" if roll < 2 else "H"
                    e = self._add_entity(nm, ch, ex, 1)
                    d = 1 if int(rng.integers(0, 2)) == 0 else -1
                    e.data.update(dx=d, timer=0)
                    self._obstacles.append(e)
                elif roll == 4:
                    f = self._add_entity("fuel", "F", ex, 1)
                    self._obstacles.append(f)

        # Move bullets
        for b in self._bullets:
            if not b.alive:
                continue
            b.y += b.dy
            if b.y <= 0:
                b.alive = False
                continue
            for obs in self._obstacles:
                if not obs.alive:
                    continue
                if obs.x == b.x and obs.y == b.y:
                    b.alive = False
                    if obs.etype == "fuel":
                        obs.alive = False
                        self._on_point_scored(1)
                        if self._progress_count < self._WIN_TARGET:
                            reward += 1.0 / self._WIN_TARGET
                            self._progress_count += 1
                        self._fuel = min(
                            self._MAX_FUEL, self._fuel + 30
                        )
                        self._message = "Fuel & refuel"
                    else:
                        obs.alive = False
                        self._on_point_scored(1)
                        if self._progress_count < self._WIN_TARGET:
                            reward += 1.0 / self._WIN_TARGET
                            self._progress_count += 1
                        self._message = "Enemy destroyed!"
                    break
        self._bullets = [b for b in self._bullets if b.alive]

        # Move enemies horizontally
        for obs in self._obstacles:
            if not obs.alive:
                continue
            if obs.etype in ("ship", "heli"):
                obs.data["timer"] = obs.data.get("timer", 0) + 1
                if obs.data["timer"] % 3 == 0:
                    obs.x += obs.data.get("dx", 0)
                    rl = self._river_l[min(obs.y, self._HEIGHT - 1)]
                    rr = self._river_r[min(obs.y, self._HEIGHT - 1)]
                    if obs.x <= rl or obs.x >= rr:
                        obs.data["dx"] = -obs.data.get("dx", 1)

        # Player picks up fuel by touching
        for obs in self._obstacles:
            if (
                obs.alive
                and obs.etype == "fuel"
                and obs.x == self._player_x
                and obs.y == self._player_y
            ):
                obs.alive = False
                self._fuel = min(self._MAX_FUEL, self._fuel + 30)
                self._on_point_scored(1)
                if self._progress_count < self._WIN_TARGET:
                    reward += 1.0 / self._WIN_TARGET
                    self._progress_count += 1
                self._message = "Fuel pickup!"

        # Player collision with enemies -- terminal failure
        for obs in self._obstacles:
            if (
                obs.alive
                and obs.etype in ("ship", "heli")
                and obs.x == self._player_x
                and obs.y == self._player_y
            ):
                obs.alive = False
                self._on_life_lost()
                self._message = "Crash! Game Over."
                reward = self._DEATH_PENALTY
                return reward, self._game_over, info

        # Bank collision -- terminal failure
        rl = self._river_l[self._player_y]
        rr = self._river_r[self._player_y]
        if self._player_x <= rl or self._player_x >= rr:
            self._on_life_lost()
            self._message = "Hit the bank! Game Over."
            reward = self._DEATH_PENALTY
            return reward, self._game_over, info

        # Fuel consumption
        self._fuel -= 1
        if self._fuel <= 0:
            self._game_over = True
            self._message = "Out of fuel! Game Over."
            reward = self._DEATH_PENALTY
            return reward, self._game_over, info

        # Win check
        if (
            self._progress_count >= self._WIN_TARGET
            and not self._game_over
        ):
            self._game_over = True
            info["won"] = True
            self._message = "River cleared!"

        self._obstacles = [o for o in self._obstacles if o.alive]
        self._redraw()
        info["fuel"] = self._fuel
        return reward, self._game_over, info

    def _redraw(self) -> None:
        for y in range(self._HEIGHT):
            for x in range(self._WIDTH):
                rl = self._river_l[y]
                rr = self._river_r[y]
                if y == 0 or y == self._HEIGHT - 1:
                    self._set_cell(x, y, "─")
                elif x <= rl or x >= rr:
                    self._set_cell(x, y, "█")
                else:
                    self._set_cell(x, y, "~")

        for obs in self._obstacles:
            if obs.alive:
                self._set_cell(obs.x, obs.y, obs.char)
        for b in self._bullets:
            if b.alive:
                self._set_cell(b.x, b.y, "!")

    def _render_current_observation(self) -> Any:
        """Append fuel + enemy count to base HUD.

        Base HUD provides ``Step: T / N`` (X4 cross-cutting).
        """
        from glyphbench.core.observation import GridObservation as GO

        obs = super()._render_current_observation()
        enemies = sum(
            1 for o in self._obstacles
            if o.alive and o.etype in ("ship", "heli")
        )
        extra = (
            f"Fuel: {self._fuel}    "
            f"Enemies: {enemies}"
        )
        hud = obs.hud + "\n" + extra
        return GO(
            grid=obs.grid, legend=obs.legend,
            hud=hud, message=obs.message,
        )

    def _advance_entities(self) -> None:
        self._entities = [e for e in self._entities if e.alive]

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "─": "border",
            "█": "river bank (deadly)",
            "~": "river (safe)",
            "S": "enemy ship (10pts)",
            "H": "enemy helicopter (10pts)",
            "F": "fuel depot (+30 fuel, 20pts)",
            "!": "your bullet",
            " ": "empty",
        }.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "Fly up the river, shoot enemies and collect fuel. "
            "Don't crash into banks or enemies. "
            "Running out of fuel ends the game."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari River Raid.\n\n"
            "TASK\n"
            "Fly a plane up a vertically-scrolling river, shooting "
            "down enemy ships and helicopters and collecting fuel "
            "depots. Survive as long as possible before crashing or "
            "running out of fuel.\n\n"
            "BOARD\n"
            "20x24 viewport. Banks '#' line both sides of a vertical "
            "river of water 'tilde'. Enemies: ships 'S' and "
            "helicopters 'H'. Fuel depots 'F'. Your bullets '!'. "
            "You fly along the river near the bottom of the "
            "viewport as an arrow glyph.\n\n"
            "MECHANICS\n"
            "LEFT / RIGHT shift 1 cell. UP accelerates the scroll "
            "(speed 1-3), DOWN decelerates. FIRE launches a bullet "
            "upward (max 2). Every (4 - speed) steps the river "
            "scrolls by moving all obstacles down 1 cell; new "
            "obstacles spawn at row 1. Enemies drift horizontally "
            "every 3 steps, reversing at the banks. Fuel decreases "
            "1 per step.\n\n"
            "SCORING\n"
            "Pattern D: +1/31 reward per progress event (enemy "
            "destroyed or fuel depot collected). -1 reward on "
            "collision with bank, enemy, or running out of fuel. "
            "Cumulative reward bound: [-1, +1].\n\n"
            "TERMINATION\n"
            "Single-life: any collision or out-of-fuel ends the "
            "episode with reward -1. Reaching 31 progress events "
            "ends with cumulative +1. Episode also ends after "
            "max_turns.\n\n"
            "HUD\n"
            "Shows score, level, fuel, and enemy count.\n\n"
            + self.action_spec.render_for_prompt()
        )
