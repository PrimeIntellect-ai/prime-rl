"""Atari Pitfall environment.

Side-scrolling platformer. Jump over pits, swing on vines,
collect treasures across 255 screens.

Gym ID: glyphbench/atari-pitfall-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase


class PitfallEnv(AtariBase):
    """Pitfall: side-scrolling platformer.

    The player traverses 255 screens connected left-right.
    Each screen has pits, logs, vines, and treasures.
    Jump over pits, collect treasures, avoid hazards.

    Grid: 40 wide x 12 tall.
    Gravity: agent falls if no platform below.
    Pattern D: +1/_WIN_TARGET per treasure collected (full-scope =
    32 treasures). -1.0 on death (enemy / pit fall).
    """

    action_spec = ActionSpec(
        names=("NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN"),
        descriptions=(
            "do nothing",
            "jump",
            "jump / grab vine",
            "move right",
            "move left",
            "duck / release vine",
        ),
    )

    _WIDTH = 40
    _HEIGHT = 12
    _TOTAL_SCREENS = 255

    # Pattern D full-scope target: 32 treasures collected.
    _WIN_TARGET: int = 32
    _DEATH_PENALTY: float = -1.0

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._lives = 1
        self._screen: int = 0
        self._jumping: bool = False
        self._jump_vy: int = 0
        self._on_vine: bool = False
        self._collected_screens: set[int] = set()
        self._timer: int = 2000  # countdown timer
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-pitfall-v0"

    def _reset(self, seed: int):
        self._progress_count = 0
        return super()._reset(seed)

    def _generate_level(self, seed: int) -> None:
        self._lives = 1
        self._screen = 0
        self._jumping = False
        self._jump_vy = 0
        self._on_vine = False
        self._collected_screens = set()
        self._timer = 2000
        self._player_x = 5
        self._player_y = self._HEIGHT - 3
        self._build_screen()

    def _build_screen(self) -> None:
        """Build current screen layout."""
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        scr = self._screen
        s = scr * 13 + 7  # deterministic per-screen seed

        # Sky
        for x in range(self._WIDTH):
            self._set_cell(x, 0, "─")

        # Ground
        for x in range(self._WIDTH):
            self._set_cell(x, self._HEIGHT - 1, "█")
            self._set_cell(x, self._HEIGHT - 2, "=")

        # Pits (gaps in ground)
        num_pits = 1 + (s % 3)
        for i in range(num_pits):
            pit_x = 10 + ((s + i * 11) % (self._WIDTH - 20))
            pit_w = 3 + ((s + i * 3) % 3)
            for px in range(pit_x, min(pit_x + pit_w, self._WIDTH - 1)):
                self._set_cell(px, self._HEIGHT - 2, " ")
                self._set_cell(px, self._HEIGHT - 1, "~")  # water/danger

        # Logs (obstacles on ground)
        num_logs = (s + 3) % 3
        for i in range(num_logs):
            lx = 5 + ((s + i * 17) % (self._WIDTH - 10))
            self._set_cell(lx, self._HEIGHT - 3, "L")

        # Vines (hanging from top)
        num_vines = (s + 5) % 2 + 1
        for i in range(num_vines):
            vx = 8 + ((s + i * 13) % (self._WIDTH - 16))
            for vy in range(1, self._HEIGHT - 5):
                self._set_cell(vx, vy, "V")

        # Treasure
        if scr not in self._collected_screens:
            tx = 20 + ((s + 19) % (self._WIDTH - 25))
            ty = self._HEIGHT - 3
            self._add_entity("treasure", "$", tx, ty)

        # Hazards (scorpions, snakes)
        if scr > 0:
            num_hazards = 1 + (s % 2)
            for i in range(num_hazards):
                hx = 15 + ((s + i * 7) % (self._WIDTH - 20))
                hy = self._HEIGHT - 3
                hdx = 1 if i % 2 == 0 else -1
                self._add_entity("enemy", "S", hx, hy, dx=hdx, dy=0)

    def _is_platform(self, x: int, y: int) -> bool:
        ch = self._grid_at(x, y)
        return ch in ("=", "█", "L")

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        self._timer -= 1
        if self._timer <= 0:
            self._game_over = True
            self._message = "Time's up!"
            return 0.0, True, info

        dx, dy = 0, 0
        jump = False

        if action_name == "LEFT":
            dx = -1
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT":
            dx = 1
            self._player_dir = (1, 0)
        elif action_name in ("UP", "FIRE"):
            if self._on_vine:
                pass  # stay on vine
            else:
                jump = True
        elif action_name == "DOWN" and self._on_vine:
            self._on_vine = False

        # Vine grab check
        if not self._on_vine and (action_name == "UP"):
            ch = self._grid_at(self._player_x, self._player_y)
            if ch == "V":
                self._on_vine = True
                self._jumping = False
                self._jump_vy = 0

        # On vine movement
        if self._on_vine:
            if action_name == "UP":
                new_y = self._player_y - 1
                if new_y > 0 and self._grid_at(self._player_x, new_y) == "V":
                    self._player_y = new_y
            elif action_name == "DOWN":
                self._on_vine = False
            # Can still move left/right on vine
            new_x = self._player_x + dx
            if 0 < new_x < self._WIDTH - 1:
                self._player_x = new_x
                if self._grid_at(self._player_x, self._player_y) != "V":
                    self._on_vine = False
        else:
            # Jumping
            if jump and not self._jumping:
                self._jumping = True
                self._jump_vy = -2

            if self._jumping:
                self._jump_vy += 1
                dy = self._jump_vy
                if self._jump_vy >= 0:
                    land_y = self._player_y + dy
                    for check_y in range(self._player_y + 1, min(land_y + 1, self._HEIGHT)):
                        if self._is_platform(self._player_x, check_y):
                            dy = check_y - 1 - self._player_y
                            self._jumping = False
                            self._jump_vy = 0
                            break
                    else:
                        if land_y >= self._HEIGHT - 2:
                            dy = self._HEIGHT - 3 - self._player_y
                            self._jumping = False
                            self._jump_vy = 0

            # Horizontal movement
            new_x = self._player_x + dx
            if 0 < new_x < self._WIDTH:
                self._player_x = new_x

            # Vertical movement
            new_y = self._player_y + dy
            if 0 < new_y < self._HEIGHT - 1:
                self._player_y = new_y

            # Gravity
            if not self._jumping:
                below = self._player_y + 1
                if below < self._HEIGHT and not self._is_platform(self._player_x, below):
                    self._player_y = below
                    # Check if fell into pit (Pattern D death)
                    if self._grid_at(self._player_x, self._player_y) == "~":
                        self._on_life_lost()
                        reward = self._DEATH_PENALTY
                        self._message = "Fell into pit!"

        # Screen transitions
        if self._player_x >= self._WIDTH - 1:
            self._screen = (self._screen + 1) % self._TOTAL_SCREENS
            self._player_x = 1
            self._build_screen()
        elif self._player_x <= 0:
            self._screen = (self._screen - 1) % self._TOTAL_SCREENS
            self._player_x = self._WIDTH - 2
            self._build_screen()

        # Enemy bounce
        for e in self._entities:
            if e.etype == "enemy" and e.alive:
                nx = e.x + e.dx
                if nx <= 1 or nx >= self._WIDTH - 2:
                    e.dx = -e.dx

        # Entity collisions
        for e in self._entities:
            if not e.alive:
                continue
            if e.x == self._player_x and e.y == self._player_y:
                if e.etype == "treasure":
                    e.alive = False
                    self._collected_screens.add(self._screen)
                    if self._progress_count < self._WIN_TARGET:
                        reward += 1.0 / self._WIN_TARGET
                        self._progress_count += 1
                    self._message = "Treasure!"
                elif e.etype == "enemy":
                    self._on_life_lost()
                    reward = self._DEATH_PENALTY
                    self._message = "Hit by hazard!"
                    break

        # Win check
        if self._progress_count >= self._WIN_TARGET and not self._game_over:
            self._game_over = True
            info["won"] = True
            self._message = "All treasures collected!"

        info["screen"] = self._screen
        info["timer"] = self._timer
        info["treasures"] = len(self._collected_screens)

        return reward, self._game_over, info

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "█": "ground",
            "=": "platform",
            "─": "sky",
            " ": "empty",
            "~": "water/pit",
            "V": "vine",
            "L": "log",
            "$": "treasure",
            "S": "scorpion/snake",
        }.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        count = len(self._collected_screens)
        extra = (
            f"Screen: {self._screen}/255"
            f"  Timer: {self._timer}"
            f"  Treasures: {count}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Run through the jungle collecting treasures. "
            "Jump over pits and logs. Grab vines to swing across gaps. "
            "Avoid scorpions and snakes. Don't fall in the water."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Pitfall!\n\n"
            "TASK\n"
            "Run through a jungle of 255 side-scrolling screens, "
            "collecting treasures while dodging pits, logs, "
            "scorpions and snakes. Finish as many treasures as "
            "possible before a 2000-tick timer runs out.\n\n"
            "BOARD\n"
            "40 columns by 12 rows per screen. Sky '-' tops each "
            "screen; ground '#' runs along the bottom with a "
            "platform '=' partway down. Pits, water, logs and vines "
            "decorate the screen — read the [Grid]. Treasure '$' "
            "and hazards 'S' (scorpions/snakes) on the ground. "
            "Exiting left or right transitions to neighboring "
            "screen.\n\n"
            "MECHANICS\n"
            "LEFT / RIGHT move 1 cell. UP or FIRE initiates a jump "
            "(2-row parabolic arc with gravity). UP on a vine "
            "climbs it; DOWN releases. When not jumping / on a "
            "vine, gravity pulls you 1 row down until you land on "
            "'=', '#', or 'L'. Screen transitions reset the "
            "position to x=1 or x=W-2 of the new screen.\n\n"
            "SCORING\n"
            "+1/32 reward per treasure '$' collected (Pattern D "
            "full-scope = 32 treasures across 255 screens). -1.0 "
            "on death (falling in water or touching scorpion/snake).\n\n"
            "TERMINATION\n"
            "Falling in pit or touching scorpion/snake ends the "
            "episode with -1.0. Timer running out (2000 ticks) ends "
            "without penalty. Episode ends after 32 treasures "
            "(cumulative reward plateaus at +1.0).\n\n"
            "HUD\n"
            "Shows score, current screen (0-254), timer "
            "remaining, treasures collected.\n\n"
            + self.action_spec.render_for_prompt()
        )
