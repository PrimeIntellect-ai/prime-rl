"""Atari Up'n Down environment.

Vertical road game: jump on or avoid cars on a scrolling road.

Gym ID: glyphbench/atari-upndown-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase, AtariEntity


class UpNDownEnv(AtariBase):
    """Up'n Down: drive on a scrolling road, jump on or dodge cars.

    12x20 grid. Road scrolls; cars appear ahead and behind.
    Jump on cars for points or dodge them.

    Actions: NOOP, UP, DOWN, LEFT, RIGHT, JUMP
    Pattern D: +1/_WIN_TARGET per progress event (jump-on-car
    or checkpoint reached). -1 on collision / going off-road.
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP", "DOWN", "LEFT", "RIGHT", "JUMP"),
        descriptions=(
            "do nothing",
            "accelerate (road scrolls faster)",
            "decelerate",
            "move left",
            "move right",
            "jump over or onto a car",
        ),
    )

    _WIDTH = 12
    _HEIGHT = 20
    _PLAYER_Y = 16
    _ROAD_LEFT = 2
    _ROAD_RIGHT = 9
    _NUM_LANES = 3

    # Pattern D full-scope: 50 progress events (cars stomped or
    # checkpoints reached). Each yields +1/50.
    _WIN_TARGET: int = 50
    _DEATH_PENALTY: float = -1.0

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._cars: list[AtariEntity] = []
        self._step_counter: int = 0
        self._scroll_speed: int = 1
        self._scroll_timer: int = 0
        self._jumping: bool = False
        self._jump_timer: int = 0
        self._distance: int = 0
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-upndown-v0"

    def _reset(self, seed: int):
        self._progress_count = 0
        return super()._reset(seed)

    def _lane_x(self, lane: int) -> int:
        """Get x coordinate for a lane (0-2)."""
        return self._ROAD_LEFT + 1 + lane * 2

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._cars = []
        self._step_counter = 0
        self._scroll_timer = 0
        self._scroll_speed = 1
        self._jumping = False
        self._jump_timer = 0
        self._distance = 0

        # Borders
        for x in range(self._WIDTH):
            self._set_cell(x, 0, "─")
            self._set_cell(x, self._HEIGHT - 1, "─")

        # Player in middle lane
        self._player_x = self._lane_x(1)
        self._player_y = self._PLAYER_Y

        # Spawn initial cars
        rng = self.rng
        for row in range(3, self._PLAYER_Y - 2, 3):
            lane = int(rng.integers(0, self._NUM_LANES))
            cx = self._lane_x(lane)
            going_down = int(rng.integers(0, 2)) == 0
            ch = "V" if going_down else "A"
            dy = 1 if going_down else -1
            c = self._add_entity("car", ch, cx, row, dy=dy)
            c.data["timer"] = 0
            self._cars.append(c)

        self._redraw()

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._step_counter += 1

        # Move player
        if action_name == "LEFT":
            nx = self._player_x - 2
            if nx >= self._ROAD_LEFT + 1:
                self._player_x = nx
                self._player_dir = (-1, 0)
        elif action_name == "RIGHT":
            nx = self._player_x + 2
            if nx <= self._ROAD_RIGHT - 1:
                self._player_x = nx
                self._player_dir = (1, 0)
        elif action_name == "UP":
            self._scroll_speed = min(3, self._scroll_speed + 1)
        elif action_name == "DOWN":
            self._scroll_speed = max(0, self._scroll_speed - 1)
        elif action_name == "JUMP" and not self._jumping:
            self._jumping = True
            self._jump_timer = 3

        # Handle jump
        if self._jumping:
            self._jump_timer -= 1
            if self._jump_timer <= 0:
                self._jumping = False

        # Scroll cars down
        self._scroll_timer += 1
        do_scroll = self._scroll_timer >= (4 - self._scroll_speed)
        if do_scroll and self._scroll_speed > 0:
            self._scroll_timer = 0
            self._distance += 1
            for c in self._cars:
                if not c.alive:
                    continue
                c.y += 1  # world scrolls down
                if c.y >= self._HEIGHT - 1:
                    c.alive = False

            # Move cars that go opposite direction
            for c in self._cars:
                if not c.alive:
                    continue
                c.data["timer"] = c.data.get("timer", 0) + 1
                if c.dy == -1 and c.data["timer"] % 2 == 0:
                    c.y -= 2  # coming toward player
                    if c.y <= 0:
                        c.alive = False

        # Spawn new cars
        rng = self.rng
        if do_scroll and int(rng.integers(0, 3)) == 0:
            lane = int(rng.integers(0, self._NUM_LANES))
            cx = self._lane_x(lane)
            going_down = int(rng.integers(0, 3)) > 0
            ch = "V" if going_down else "A"
            dy = 1 if going_down else -1
            c = self._add_entity("car", ch, cx, 1, dy=dy)
            c.data["timer"] = 0
            self._cars.append(c)

        # Check collisions
        for c in self._cars:
            if not c.alive:
                continue
            if c.x == self._player_x and c.y == self._player_y:
                if self._jumping:
                    # Jump on car -- progress unit
                    c.alive = False
                    self._on_point_scored(1)
                    if self._progress_count < self._WIN_TARGET:
                        reward += 1.0 / self._WIN_TARGET
                        self._progress_count += 1
                    self._message = "Jumped on car!"
                else:
                    # Crash -- terminal failure
                    c.alive = False
                    self._on_life_lost()
                    self._message = "Crash! Game Over."
                    reward = self._DEATH_PENALTY
                    return reward, self._game_over, info

        # Off-road check -- terminal failure
        if (
            self._player_x < self._ROAD_LEFT + 1
            or self._player_x > self._ROAD_RIGHT - 1
        ):
            self._on_life_lost()
            self._message = "Off road! Game Over."
            reward = self._DEATH_PENALTY
            return reward, self._game_over, info

        # Level progression by distance -- progress unit
        if self._distance >= 50 + self._level * 10:
            self._on_point_scored(1)
            if self._progress_count < self._WIN_TARGET:
                reward += 1.0 / self._WIN_TARGET
                self._progress_count += 1
            self._message = "Checkpoint!"
            self._level += 1
            self._distance = 0

        # Win check
        if (
            self._progress_count >= self._WIN_TARGET
            and not self._game_over
        ):
            self._game_over = True
            info["won"] = True
            self._message = "Distance traveled!"

        self._cars = [c for c in self._cars if c.alive]
        self._redraw()
        info["distance"] = self._distance
        return reward, self._game_over, info

    def _redraw(self) -> None:
        for y in range(1, self._HEIGHT - 1):
            for x in range(self._WIDTH):
                if self._ROAD_LEFT <= x <= self._ROAD_RIGHT:
                    self._set_cell(x, y, ":")
                else:
                    self._set_cell(x, y, "█")

        # Lane dividers
        for y in range(1, self._HEIGHT - 1):
            self._set_cell(self._ROAD_LEFT, y, "│")
            self._set_cell(self._ROAD_RIGHT, y, "│")

        for c in self._cars:
            if c.alive:
                self._set_cell(c.x, c.y, c.char)

        # Show jump indicator
        if self._jumping:
            jx = self._player_x
            jy = self._player_y - 1
            if 0 < jy < self._HEIGHT - 1:
                self._set_cell(jx, jy, "↑")

    def _advance_entities(self) -> None:
        self._entities = [e for e in self._entities if e.alive]

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "─": "border",
            "█": "off-road",
            "│": "road edge",
            ":": "road",
            "V": "car (same direction)",
            "A": "car (oncoming)",
            "↑": "jump arc",
            " ": "empty",
        }.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        state = "airborne" if self._jumping else "grounded"
        extra = (
            f"Jump: {state}  Distance: {self._distance}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Drive on the road and use JUMP to land on cars "
            "for points. Avoid collisions without jumping. "
            "Stay on the road. Reach checkpoints to advance."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Up'n Down.\n\n"
            "TASK\n"
            "Drive on a 3-lane scrolling road. Use JUMP to land on "
            "cars (score points) or avoid them. Travel far enough "
            "to reach checkpoints and advance the level.\n\n"
            "BOARD\n"
            "12x20 road. Borders '-' top/bottom. The road has "
            "three lanes bordered by '|' edges with off-road '#' "
            "shoulders; road surface ':' inside. Cars in your "
            "direction are 'V', oncoming cars 'A'. You drive near "
            "the bottom of the viewport as an arrow glyph. When "
            "airborne, a jump arc 'up-arrow' appears above you.\n\n"
            "MECHANICS\n"
            "LEFT / RIGHT change lane (step 2 cells). UP "
            "accelerates (scroll_speed 0-3), DOWN decelerates. "
            "JUMP lifts you for 3 steps. Every (4 - speed) steps "
            "the road scrolls: all cars shift down 1 row; "
            "oncoming cars also move up 2 rows every 2 steps. New "
            "cars spawn at row 1 ~33 percent of scroll steps.\n\n"
            "SCORING\n"
            "Pattern D: +1/50 reward per progress event (landing "
            "on a car while airborne, or reaching a checkpoint). "
            "-1 reward on collision with a car without jumping "
            "or going off-road. Cumulative reward bound: "
            "[-1, +1].\n\n"
            "TERMINATION\n"
            "Single-life: any collision/off-road event ends with "
            "-1. Reaching 50 progress events ends with cumulative "
            "+1. Episode also ends after max_turns.\n\n"
            "HUD\n"
            "Shows score, level, jump state, distance "
            "traveled (toward next checkpoint).\n\n"
            + self.action_spec.render_for_prompt()
        )
