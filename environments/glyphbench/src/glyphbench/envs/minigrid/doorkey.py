"""MiniGrid DoorKey environments.

A room divided by a wall with a locked door. Agent must find the key, pick it up,
unlock the door, and navigate to the goal on the other side.
"""

from __future__ import annotations

from glyphbench.envs.minigrid.base import DIR_RIGHT, MiniGridBase
from glyphbench.envs.minigrid.objects import Door, Goal, Key, Wall


class _DoorKeyBase(MiniGridBase):
    _room_size: int = 5  # interior size, grid = room_size + 2

    def _generate_grid(self, seed: int) -> None:
        size = self._room_size + 2
        self._init_grid(size, size)

        # Vertical wall dividing the room
        wall_x = size // 2
        for y in range(1, size - 1):
            self._place_obj(wall_x, y, Wall())

        # Door in the dividing wall at a random y position
        door_y = int(self.rng.integers(1, size - 1))
        self._grid[door_y][wall_x] = None  # clear the wall
        self._place_obj(wall_x, door_y, Door(color="yellow", is_locked=True))

        # Key on the left side at random position
        key_x = int(self.rng.integers(1, wall_x))
        key_y = int(self.rng.integers(1, size - 1))
        self._place_obj(key_x, key_y, Key(color="yellow"))

        # Agent on the left side, not on key
        while True:
            agent_x = int(self.rng.integers(1, wall_x))
            agent_y = int(self.rng.integers(1, size - 1))
            if (agent_x, agent_y) != (key_x, key_y):
                break
        self._place_agent(agent_x, agent_y, DIR_RIGHT)

        # Goal on the right side
        goal_x = int(self.rng.integers(wall_x + 1, size - 1))
        goal_y = int(self.rng.integers(1, size - 1))
        self._place_obj(goal_x, goal_y, Goal())

    def _task_description(self) -> str:
        goal = Goal().render_char()
        yellow_key = Key(color="yellow").render_char()
        # Locked door renders with the closed-door glyph.
        yellow_door = Door(color="yellow", is_locked=True).render_char()
        return (
            f"Navigate a {self._room_size}x{self._room_size} room "
            f"divided by a wall. Find the yellow key ({yellow_key}), pick it up "
            f"with PICKUP, face the locked yellow door ({yellow_door}) and use "
            f"TOGGLE to unlock it, then navigate to the goal ({goal}). "
            "Reward = 1 - 0.9 * (steps / max_steps)."
        )


class MiniGridDoorKey5x5Env(_DoorKeyBase):
    _room_size = 5

    def env_id(self) -> str:
        return "glyphbench/minigrid-doorkey-5x5-v0"


class MiniGridDoorKey6x6Env(_DoorKeyBase):
    _room_size = 6

    def env_id(self) -> str:
        return "glyphbench/minigrid-doorkey-6x6-v0"


class MiniGridDoorKey8x8Env(_DoorKeyBase):
    _room_size = 8

    def env_id(self) -> str:
        return "glyphbench/minigrid-doorkey-8x8-v0"


class MiniGridDoorKey16x16Env(_DoorKeyBase):
    _room_size = 16

    def env_id(self) -> str:
        return "glyphbench/minigrid-doorkey-16x16-v0"
