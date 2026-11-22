"""MiniGrid MultiRoom environments.

N rooms connected in sequence by doors. Agent in first room, goal in last.
"""

from __future__ import annotations

from glyphbench.envs.minigrid.base import DIR_RIGHT, MiniGridBase
from glyphbench.envs.minigrid.objects import Door, Goal, Wall


class _MultiRoomBase(MiniGridBase):
    _num_rooms: int = 2
    _room_size: int = 4  # interior size

    def _generate_grid(self, seed: int) -> None:
        rs = self._room_size
        nr = self._num_rooms
        # Each room occupies (rs + 1) columns (interior + right wall), plus 1
        # for the left outer wall.  Rooms share the dividing wall column.
        grid_w = (rs + 1) * nr + 1
        grid_h = rs + 2
        self._init_grid(grid_w, grid_h)

        # Add interior vertical walls between rooms and place a door in each
        for i in range(1, nr):
            wall_x = (rs + 1) * i
            for y in range(1, grid_h - 1):
                self._place_obj(wall_x, y, Wall())

            # Door at a random y position in the wall
            door_y = int(self.rng.integers(1, grid_h - 1))
            self._grid[door_y][wall_x] = None
            self._place_obj(wall_x, door_y, Door(color="yellow"))

        # Agent in the first room
        ax = int(self.rng.integers(1, rs + 1))
        ay = int(self.rng.integers(1, grid_h - 1))
        self._place_agent(ax, ay, DIR_RIGHT)

        # Goal in the last room
        last_room_start = (rs + 1) * (nr - 1) + 1
        last_room_end = grid_w - 1
        gx = int(self.rng.integers(last_room_start, last_room_end))
        gy = int(self.rng.integers(1, grid_h - 1))
        self._place_obj(gx, gy, Goal())

    def _task_description(self) -> str:
        return (
            f"Navigate through {self._num_rooms} connected "
            "rooms to reach the goal. Each room is connected "
            "by a door. Use TOGGLE to open doors, then walk "
            "through. Reward = 1 - 0.9 * (steps / max_steps)."
        )


class MiniGridMultiRoomN2S4Env(_MultiRoomBase):
    _num_rooms = 2
    _room_size = 4

    def env_id(self) -> str:
        return "glyphbench/minigrid-multiroom-n2-s4-v0"


class MiniGridMultiRoomN4S5Env(_MultiRoomBase):
    _num_rooms = 4
    _room_size = 5

    def env_id(self) -> str:
        return "glyphbench/minigrid-multiroom-n4-s5-v0"


class MiniGridMultiRoomN6Env(_MultiRoomBase):
    _num_rooms = 6
    _room_size = 4

    def env_id(self) -> str:
        return "glyphbench/minigrid-multiroom-n6-v0"
