"""Atari Defender environment.

Horizontal scrolling shooter. Defend humanoids from aliens.

Gym ID: glyphbench/atari-defender-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase, AtariEntity


class DefenderEnv(AtariBase):
    """Defender: horizontal scrolling shooter.

    30x16 viewport over a wider world. Protect humanoids on the
    ground from abducting aliens. Thrust to move faster.

    Actions: NOOP, LEFT, RIGHT, UP, DOWN, FIRE, THRUST
    Pattern A: +1/_WIN_TARGET per humanoid rescued (full-scope = 10
    rescues). No failure penalty.
    """

    action_spec = ActionSpec(
        names=(
            "NOOP", "LEFT", "RIGHT", "UP", "DOWN",
            "FIRE", "THRUST",
        ),
        descriptions=(
            "do nothing",
            "move left",
            "move right",
            "move up",
            "move down",
            "fire a laser forward",
            "boost speed in current direction",
        ),
    )

    _VP_WIDTH = 30
    _VP_HEIGHT = 16
    _WORLD_WIDTH = 80
    _GROUND_Y = 14
    _PLAYER_START_X = 10
    _PLAYER_START_Y = 8

    # Pattern A full-scope target: 10 humanoids rescued.
    _WIN_TARGET: int = 10

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._aliens: list[AtariEntity] = []
        self._humanoids: list[AtariEntity] = []
        self._lasers: list[AtariEntity] = []
        self._step_counter: int = 0
        self._camera_x: int = 0
        self._world_x: int = 0
        self._facing: int = 1  # 1=right, -1=left
        self._thrust: bool = False
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-defender-v0"

    def _reset(self, seed: int) -> GridObservation:
        self._progress_count = 0
        return super()._reset(seed)

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._VP_WIDTH, self._VP_HEIGHT)
        self._entities = []
        self._aliens = []
        self._humanoids = []
        self._lasers = []
        self._step_counter = 0
        self._thrust = False
        self._facing = 1

        self._world_x = self._PLAYER_START_X
        self._player_y = self._PLAYER_START_Y
        self._camera_x = 0

        rng = self.rng
        # Spawn humanoids across the world
        for _i in range(6):
            hx = int(rng.integers(5, self._WORLD_WIDTH - 5))
            h = self._add_entity(
                "humanoid", "H", hx, self._GROUND_Y,
            )
            self._humanoids.append(h)

        # Spawn aliens
        n_aliens = min(4 + self._level * 2, 12)
        for _ in range(n_aliens):
            ax = int(rng.integers(0, self._WORLD_WIDTH))
            ay = int(rng.integers(2, self._GROUND_Y - 2))
            a = self._add_entity("alien", "A", ax, ay)
            a.data["state"] = "patrol"
            a.data["dy"] = int(rng.choice([-1, 1]))
            self._aliens.append(a)

        self._player_x = self._world_x
        self._redraw()

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._step_counter += 1
        speed = 2 if self._thrust else 1
        self._thrust = False

        if action_name == "THRUST":
            self._thrust = True
            speed = 3
        elif action_name == "LEFT":
            self._facing = -1
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT":
            self._facing = 1
            self._player_dir = (1, 0)
        elif action_name == "UP" and self._player_y > 1:
            self._player_y -= 1
            self._player_dir = (0, -1)
        elif (
            action_name == "DOWN"
            and self._player_y < self._GROUND_Y - 1
        ):
            self._player_y += 1
            self._player_dir = (0, 1)
        elif action_name == "FIRE" and len(self._lasers) < 2:
            lx = self._world_x + self._facing
            laser = self._add_entity(
                "laser", "─" if self._facing > 0 else "─",
                lx, self._player_y,
                dx=self._facing * 2,
            )
            laser.data["ttl"] = 8
            self._lasers.append(laser)

        # Move player in world
        if action_name in ("LEFT", "RIGHT", "THRUST"):
            self._world_x = (
                self._world_x + self._facing * speed
            ) % self._WORLD_WIDTH
        self._player_x = self._world_x

        # Camera follows player
        self._camera_x = (
            self._world_x - self._VP_WIDTH // 3
        ) % self._WORLD_WIDTH

        # Move lasers
        for laser in self._lasers:
            if not laser.alive:
                continue
            laser.x += laser.dx
            laser.data["ttl"] = laser.data.get("ttl", 8) - 1
            if laser.data["ttl"] <= 0:
                laser.alive = False

        # Laser-alien collisions.
        # Audit BROKEN fix: rescue credit now triggers on the SHOOT
        # event when the alien is carrying a humanoid, instead of on
        # the passive landing of the humanoid. This decouples
        # progress from a chain of unfocused events and ties it to
        # an active skill (shoot the abductor mid-flight).
        for laser in self._lasers:
            if not laser.alive:
                continue
            for alien in self._aliens:
                if (
                    alien.alive
                    and abs(alien.x - laser.x) <= 1
                    and alien.y == laser.y
                ):
                    alien.alive = False
                    laser.alive = False
                    self._on_point_scored(1)
                    self._message = "Alien destroyed!"
                    # Release any carried humanoid; rescue credit
                    # goes to the agent now (active skill).
                    if "carrying" in alien.data:
                        h = alien.data["carrying"]
                        h.data["falling"] = True
                        h.data["rescued_by_agent"] = True
                        if self._progress_count < self._WIN_TARGET:
                            reward += 1.0 / self._WIN_TARGET
                            self._progress_count += 1
                        self._message = "Humanoid rescued!"
                    break
        self._lasers = [
            laser for laser in self._lasers if laser.alive
        ]

        # Move aliens
        if self._step_counter % 3 == 0:
            for alien in self._aliens:
                if not alien.alive:
                    continue
                state = alien.data.get("state", "patrol")
                if state == "patrol":
                    dy = alien.data.get("dy", 1)
                    alien.y += dy
                    if alien.y <= 1 or alien.y >= self._GROUND_Y - 1:
                        alien.data["dy"] = -dy
                    # Drift toward nearest humanoid
                    nearest = None
                    best_dist = 999
                    for h in self._humanoids:
                        if h.alive and "carried" not in h.data:
                            d = abs(h.x - alien.x)
                            if d < best_dist:
                                best_dist = d
                                nearest = h
                    if nearest and best_dist < 15:
                        if alien.x < nearest.x:
                            alien.x += 1
                        elif alien.x > nearest.x:
                            alien.x -= 1
                        if (
                            abs(alien.x - nearest.x) <= 1
                            and alien.y >= self._GROUND_Y - 1
                        ):
                            alien.data["state"] = "abducting"
                            alien.data["carrying"] = nearest
                            nearest.data["carried"] = True
                elif state == "abducting":
                    alien.y -= 1
                    h = alien.data.get("carrying")
                    if h and h.alive:
                        h.x = alien.x
                        h.y = alien.y + 1
                    if alien.y <= 1:
                        # Humanoid lost
                        if h and h.alive:
                            h.alive = False
                        alien.data["state"] = "patrol"

                # Collision with player
                if (
                    alien.alive
                    and abs(alien.x - self._world_x) <= 1
                    and alien.y == self._player_y
                ):
                    alien.alive = False
                    self._on_life_lost()
                    self._message = "Alien collision!"
                    self._world_x = self._PLAYER_START_X
                    self._player_y = self._PLAYER_START_Y

        # Handle falling humanoids.
        # Audit BROKEN fix: landing no longer credits a rescue (that
        # is now done at SHOOT time on the carrying alien). Landing
        # only restores the humanoid to the ground.
        for h in self._humanoids:
            if not h.alive:
                continue
            if h.data.get("falling"):
                h.y += 1
                if h.y >= self._GROUND_Y:
                    h.y = self._GROUND_Y
                    h.data["falling"] = False
                    h.data.pop("carried", None)

        self._aliens = [a for a in self._aliens if a.alive]

        # Win check
        if self._progress_count >= self._WIN_TARGET and not self._game_over:
            self._game_over = True
            info["won"] = True
            self._message = "All humanoids saved!"

        # Level clear
        if not self._aliens and not self._game_over:
            self._level += 1
            self._message = "Wave cleared!"
            self._generate_level(self._level)

        self._redraw()
        info["aliens"] = len(self._aliens)
        info["humanoids"] = sum(
            1 for h in self._humanoids if h.alive
        )
        return reward, self._game_over, info

    def _world_to_screen(self, wx: int) -> int:
        """Convert world x to screen x."""
        return (wx - self._camera_x) % self._WORLD_WIDTH

    def _redraw(self) -> None:
        # Clear
        for y in range(self._VP_HEIGHT):
            for x in range(self._VP_WIDTH):
                self._set_cell(x, y, " ")
        # Top/bottom borders
        for x in range(self._VP_WIDTH):
            self._set_cell(x, 0, "─")
            self._set_cell(x, self._VP_HEIGHT - 1, "─")
        # Ground
        for x in range(self._VP_WIDTH):
            self._set_cell(x, self._GROUND_Y, "=")

        # Draw entities in viewport
        for h in self._humanoids:
            if not h.alive:
                continue
            sx = self._world_to_screen(h.x)
            if 0 <= sx < self._VP_WIDTH:
                self._set_cell(sx, h.y, "H")
        for a in self._aliens:
            if not a.alive:
                continue
            sx = self._world_to_screen(a.x)
            if 0 <= sx < self._VP_WIDTH:
                self._set_cell(sx, a.y, "A")
        for laser in self._lasers:
            if not laser.alive:
                continue
            sx = self._world_to_screen(laser.x)
            if 0 <= sx < self._VP_WIDTH:
                self._set_cell(sx, laser.y, "~")

        # Player position on screen
        self._player_x = self._world_to_screen(self._world_x)
        if self._player_x >= self._VP_WIDTH:
            self._player_x = self._VP_WIDTH // 3

    def _advance_entities(self) -> None:
        self._entities = [e for e in self._entities if e.alive]

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "─": "border", "=": "ground",
            "A": "alien", "H": "humanoid",
            "~": "laser", " ": "sky",
        }.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        n_humans = sum(
            1 for h in self._humanoids if h.alive
        )
        extra = f"Humanoids: {n_humans}"
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Fly your ship and destroy aliens. "
            "Protect humanoids on the ground from abduction. "
            "Use THRUST for speed boost. FIRE shoots forward."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Defender.\n\n"
            "TASK\n"
            "Fly a ship horizontally across a scrolling wrap-around "
            "world, destroy aliens, and save humanoids from being "
            "abducted off the top edge.\n\n"
            "BOARD\n"
            "Viewport is 30 columns by 16 rows showing a wider "
            "wrap-around world of width 80. Borders '-' top and "
            "bottom; ground '=' at row 14. Humanoids 'H' walk the "
            "ground; aliens 'A' patrol the sky above. Your laser shows "
            "as '~'. You are an arrow glyph.\n\n"
            "MECHANICS\n"
            "LEFT / RIGHT set facing and move you 1 cell in world "
            "coordinates. THRUST boosts your move to 3 cells that "
            "step. UP / DOWN move vertically in the viewport. FIRE "
            "launches a laser in your facing with speed 2/step and "
            "TTL 8 (max 2 lasers). Aliens update every 3 steps: "
            "patrol (bounce vertically while drifting to nearest "
            "humanoid within 15 columns) or abduct (grab humanoid "
            "and climb 1 row/step; reaches top -> humanoid lost). "
            "Shooting a carrying alien is a RESCUE: the humanoid "
            "falls and you receive immediate progress credit.\n\n"
            "SCORING\n"
            "+1 progress unit per humanoid rescued (i.e. per "
            "carrying alien you shoot before it reaches the top). "
            "Hitting a non-carrying alien only adds to your score, "
            "not to progress. Clearing all aliens advances the "
            "level.\n\n"
            "TERMINATION\n"
            "Single-life: colliding with an alien ends the "
            "episode. The episode also ends after max_turns.\n\n"
            "HUD\n"
            "Shows score, wave/level, facing, world x-position, "
            "and humanoids alive.\n\n"
            + self.action_spec.render_for_prompt()
        )
