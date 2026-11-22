"""Atari Berzerk environment.

Navigate rooms, shoot robots, avoid walls. Exit through doors.

Gym ID: glyphbench/atari-berzerk-v0
"""

from __future__ import annotations

from typing import Any

import numpy as np

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase, AtariEntity

_DIRS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}

_W = 20
_H = 16
_MAX_ROBOTS = 8
_BULLET_CHAR = "!"
_ROBOT_CHAR = "r"
_EXIT_CHAR = "D"

class BerzerkEnv(AtariBase):
    """Berzerk: shoot robots, escape rooms, don't touch walls.

    Actions: NOOP, FIRE, UP, RIGHT, LEFT, DOWN,
             UP_FIRE, DOWN_FIRE, LEFT_FIRE, RIGHT_FIRE
    Pattern D: +1/_WIN_TARGET per robot destroyed (full-scope = 5
    rooms x 6 robots = 30), -1.0 on damage (single-life model).
    Level clears when all robots dead or agent exits through door.
    """

    action_spec = ActionSpec(
        names=(
            "NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN",
            "UP_FIRE", "DOWN_FIRE", "LEFT_FIRE", "RIGHT_FIRE",
        ),
        descriptions=(
            "do nothing this step",
            "fire bullet in last-faced direction",
            "move up one cell",
            "move right one cell",
            "move left one cell",
            "move down one cell",
            "move up and fire",
            "move down and fire",
            "move left and fire",
            "move right and fire",
        ),
    )

    # Pattern D full-scope target: 30 robots destroyed (5 rooms x 6).
    _WIN_TARGET: int = 30
    _DEATH_PENALTY: float = -1.0

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._facing: tuple[int, int] = (1, 0)
        self._robots_killed: int = 0
        self._total_robots: int = 0
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-berzerk-v0"

    def _reset(self, seed: int) -> GridObservation:
        self._progress_count = 0
        return super()._reset(seed)

    def _task_description(self) -> str:
        return (
            "Shoot robots (r) and exit through the door (D). "
            "Touching a wall (#) or a robot kills you. "
            "Fire bullets (!) to destroy robots."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Berzerk.\n\n"
            "TASK\n"
            "Survive a series of rooms filled with shooting robots. "
            "Destroy all robots or escape through the exit door to "
            "advance to the next room.\n\n"
            "BOARD\n"
            "20x16 room. Border and random interior walls are '#' and "
            "are lethal on contact. The exit door 'D' is in the right "
            "wall at a random row. Robots are 'r'. Bullets (yours or "
            "theirs) are '!'. You spawn somewhere on the floor and are "
            "an arrow glyph showing facing direction.\n\n"
            "MECHANICS\n"
            "Each step you pick one of: NOOP, FIRE, a move direction, or "
            "a move+FIRE combo. Moves change facing and move 1 cell; "
            "FIRE emits a bullet in facing direction traveling 1 cell "
            "per step until it hits a wall or robot. Robots chase you "
            "(30 percent chance per step, one-axis) and fire at you on a "
            "5-20 step cooldown along the dominant axis.\n\n"
            "SCORING\n"
            "+1 reward per robot destroyed by your bullet. -1 reward "
            "each time you take damage (wall touch, robot contact, "
            "enemy bullet hit). No reward for clearing a room or "
            "escaping.\n\n"
            "TERMINATION\n"
            "Single-life: any damage (wall, robot contact, enemy "
            "bullet) ends the episode. Walking onto the door "
            "advances the level and generates a new room. The "
            "episode also ends after max_turns.\n\n"
            "HUD\n"
            "Shows score, level, facing direction, and robots "
            "remaining.\n\n"
            + self.action_spec.render_for_prompt()
        )

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "█": "wall",
            " ": "empty",
            "r": "robot enemy",
            "!": "bullet",
            "D": "exit door",
        }.get(ch, ch)

    def _generate_level(self, seed: int) -> None:
        rng = np.random.default_rng(seed + self._level * 1000)
        self._init_grid(_W, _H)
        self._entities = []
        self._robots_killed = 0
        self._facing = (1, 0)

        # Border walls
        for x in range(_W):
            self._set_cell(x, 0, "█")
            self._set_cell(x, _H - 1, "█")
        for y in range(_H):
            self._set_cell(0, y, "█")
            self._set_cell(_W - 1, y, "█")

        # Interior walls: random vertical and horizontal segments
        n_walls = int(rng.integers(3, 7))
        for _ in range(n_walls):
            if rng.random() < 0.5:
                # Vertical wall
                wx = int(rng.integers(3, _W - 3))
                wy = int(rng.integers(2, _H - 4))
                length = int(rng.integers(2, 5))
                for j in range(length):
                    if wy + j < _H - 1:
                        self._set_cell(wx, wy + j, "█")
            else:
                # Horizontal wall
                wx = int(rng.integers(3, _W - 5))
                wy = int(rng.integers(2, _H - 3))
                length = int(rng.integers(2, 5))
                for j in range(length):
                    if wx + j < _W - 1:
                        self._set_cell(wx + j, wy, "█")

        # Place exit door on right wall
        door_y = int(rng.integers(2, _H - 2))
        self._set_cell(_W - 1, door_y, _EXIT_CHAR)

        # Place player in bottom-left area
        self._player_x = 2
        self._player_y = _H - 3
        # Make sure player start is clear
        self._set_cell(self._player_x, self._player_y, " ")

        # Place robots
        n_robots = min(_MAX_ROBOTS, 3 + self._level)
        self._total_robots = 0
        for _ in range(n_robots):
            for _attempt in range(20):
                rx = int(rng.integers(3, _W - 3))
                ry = int(rng.integers(2, _H - 3))
                if (
                    self._grid_at(rx, ry) == " "
                    and abs(rx - self._player_x) + abs(ry - self._player_y) > 4
                ):
                    robot = self._add_entity("robot", _ROBOT_CHAR, rx, ry)
                    robot.data["shoot_timer"] = int(rng.integers(5, 15))
                    self._total_robots += 1
                    break

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        # Parse action
        move_dir: tuple[int, int] | None = None
        fire = False
        if action_name == "NOOP":
            pass
        elif action_name == "FIRE":
            fire = True
        elif action_name in _DIRS:
            move_dir = _DIRS[action_name]
            self._facing = move_dir
            self._player_dir = move_dir
        elif action_name == "UP_FIRE":
            move_dir = _DIRS["UP"]
            self._facing = move_dir
            self._player_dir = move_dir
            fire = True
        elif action_name == "DOWN_FIRE":
            move_dir = _DIRS["DOWN"]
            self._facing = move_dir
            self._player_dir = move_dir
            fire = True
        elif action_name == "LEFT_FIRE":
            move_dir = _DIRS["LEFT"]
            self._facing = move_dir
            self._player_dir = move_dir
            fire = True
        elif action_name == "RIGHT_FIRE":
            move_dir = _DIRS["RIGHT"]
            self._facing = move_dir
            self._player_dir = move_dir
            fire = True

        # Move player
        if move_dir is not None:
            nx = self._player_x + move_dir[0]
            ny = self._player_y + move_dir[1]
            cell = self._grid_at(nx, ny)
            if cell == _EXIT_CHAR:
                # Exit room - level complete
                self._player_x, self._player_y = nx, ny
                self._level += 1
                self._message = "Room cleared!"
                self._generate_level(self._level * 7919)
                info["room_cleared"] = True
                return reward, False, info
            if not self._is_solid(nx, ny):
                self._player_x, self._player_y = nx, ny

        # Player touched a wall (terminal failure). Additive penalty so any
        # kill / progress reward earned this step survives into cumulative.
        if self._is_solid(self._player_x, self._player_y):
            self._on_life_lost()
            reward += self._DEATH_PENALTY
            return reward, self._game_over, info

        # Fire bullet
        if fire:
            bx = self._player_x + self._facing[0]
            by = self._player_y + self._facing[1]
            if not self._is_solid(bx, by):
                b = self._add_entity("bullet", _BULLET_CHAR, bx, by)
                b.dx = self._facing[0]
                b.dy = self._facing[1]
                b.data["owner"] = "player"

        # Move bullets
        new_bullets: list[AtariEntity] = []
        for e in self._entities:
            if e.etype != "bullet" or not e.alive:
                continue
            e.x += e.dx
            e.y += e.dy
            if self._is_solid(e.x, e.y) or e.x <= 0 or e.x >= _W - 1 or e.y <= 0 or e.y >= _H - 1:
                e.alive = False
            else:
                new_bullets.append(e)

        # Robot AI
        for e in self._entities:
            if e.etype != "robot" or not e.alive:
                continue
            # Simple movement toward player
            if self.rng.random() < 0.3:
                dx = 0
                dy = 0
                if self._player_x > e.x:
                    dx = 1
                elif self._player_x < e.x:
                    dx = -1
                if self._player_y > e.y:
                    dy = 1
                elif self._player_y < e.y:
                    dy = -1
                # Pick one axis
                if self.rng.random() < 0.5:
                    dy = 0
                else:
                    dx = 0
                nx, ny = e.x + dx, e.y + dy
                if not self._is_solid(nx, ny):
                    e.x, e.y = nx, ny

            # Robot shooting
            e.data["shoot_timer"] = e.data.get("shoot_timer", 10) - 1
            if e.data["shoot_timer"] <= 0:
                e.data["shoot_timer"] = int(self.rng.integers(8, 20))
                # Fire toward player
                ddx = 0
                ddy = 0
                if self._player_x > e.x:
                    ddx = 1
                elif self._player_x < e.x:
                    ddx = -1
                if self._player_y > e.y:
                    ddy = 1
                elif self._player_y < e.y:
                    ddy = -1
                if ddx != 0 or ddy != 0:
                    # Pick one axis for bullet direction
                    if abs(self._player_x - e.x) > abs(self._player_y - e.y):
                        ddy = 0
                    else:
                        ddx = 0
                    bx, by = e.x + ddx, e.y + ddy
                    if not self._is_solid(bx, by):
                        rb = self._add_entity("bullet", _BULLET_CHAR, bx, by)
                        rb.dx = ddx
                        rb.dy = ddy
                        rb.data["owner"] = "robot"

        # Bullet-robot collisions. A robot bullet hitting the player adds the
        # death penalty additively (preserves kill credit earned this step) but
        # AT MOST ONCE per step: multiple coincident bullets / a bullet plus a
        # body collision must not stack -N death penalties and break [-1, 1].
        death_penalty_applied = False
        for b in self._entities:
            if b.etype != "bullet" or not b.alive:
                continue
            if b.data.get("owner") == "robot":
                if b.x == self._player_x and b.y == self._player_y:
                    b.alive = False
                    if not death_penalty_applied:
                        self._on_life_lost()
                        reward += self._DEATH_PENALTY
                        death_penalty_applied = True
                continue
            # Player bullets hit robots
            for r in self._entities:
                if r.etype != "robot" or not r.alive:
                    continue
                if b.x == r.x and b.y == r.y:
                    b.alive = False
                    r.alive = False
                    self._robots_killed += 1
                    self._on_point_scored(1)
                    if self._progress_count < self._WIN_TARGET:
                        reward += 1.0 / self._WIN_TARGET
                        self._progress_count += 1

        # Robot-player collision (terminal failure). Additive penalty so
        # kill credit earned this step survives, but only if a bullet death
        # this step did not already apply the (single) death penalty.
        for e in self._entities:
            if e.etype == "robot" and e.alive and e.x == self._player_x and e.y == self._player_y:
                if not death_penalty_applied:
                    self._on_life_lost()
                    reward += self._DEATH_PENALTY
                    death_penalty_applied = True
                break

        # Clean dead entities
        self._entities = [e for e in self._entities if e.alive]

        # Win check
        if self._progress_count >= self._WIN_TARGET and not self._game_over:
            self._game_over = True
            info["won"] = True
            self._message = "All rooms cleared!"

        # Check if all robots are dead
        robots_alive = sum(1 for e in self._entities if e.etype == "robot" and e.alive)
        if robots_alive == 0 and self._total_robots > 0 and not self._game_over:
            self._message = "All robots destroyed!"
            self._level += 1
            self._generate_level(self._level * 7919)
            info["room_cleared"] = True

        info["robots_alive"] = robots_alive
        info["robots_killed"] = self._robots_killed
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        robots = sum(
            1 for e in self._entities
            if e.etype == "robot" and e.alive
        )
        extra = f"Robots: {robots}"
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _advance_entities(self) -> None:
        """Override: entities are moved in _game_step."""
        pass
