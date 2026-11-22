"""MiniGrid KeyCorridor environments.

A central corridor with rooms branching off. Key in one room, goal behind
a locked door in another room.
"""

from __future__ import annotations

from glyphbench.envs.minigrid.base import DIR_RIGHT, MiniGridBase
from glyphbench.envs.minigrid.objects import Door, Goal, Key, Wall


class _KeyCorridorBase(MiniGridBase):
    _room_size: int = 3  # interior size of each room
    _num_rows: int = 1  # number of rooms on each side of corridor

    def _generate_grid(self, seed: int) -> None:
        rs = self._room_size
        nr = self._num_rows
        # Total rooms = 2 * nr (nr above, nr below corridor)
        # Width: (rs + 1) * nr + 1  (rooms with shared walls between)
        # Height: top wall + top rooms + corridor + bottom rooms + bottom wall
        grid_w = (rs + 1) * nr + 1
        grid_h = 2 * rs + 5
        self._init_grid(grid_w, grid_h)

        corridor_y = rs + 2  # y position of corridor row

        # Horizontal wall above corridor
        for x in range(1, grid_w - 1):
            self._place_obj(x, corridor_y - 1, Wall())
        # Horizontal wall below corridor
        for x in range(1, grid_w - 1):
            self._place_obj(x, corridor_y + 1, Wall())

        # Vertical walls between rooms (top and bottom)
        for i in range(1, nr):
            wall_x = (rs + 1) * i
            # Top rooms
            for y in range(1, corridor_y - 1):
                self._place_obj(wall_x, y, Wall())
            # Bottom rooms
            for y in range(corridor_y + 2, grid_h - 1):
                self._place_obj(wall_x, y, Wall())

        # Doorways from corridor to rooms
        room_doors: list[tuple[int, int, bool]] = []  # (x, y, is_above)
        for i in range(nr):
            room_center_x = (rs + 1) * i + 1 + rs // 2
            # Door to room above
            self._grid[corridor_y - 1][room_center_x] = None
            room_doors.append((room_center_x, corridor_y - 1, True))
            # Door to room below
            self._grid[corridor_y + 1][room_center_x] = None
            room_doors.append((room_center_x, corridor_y + 1, False))

        total_rooms = 2 * nr
        # Pick which room gets the locked door+goal and which gets the key
        goal_room_idx = int(self.rng.integers(0, total_rooms))
        key_room_idx = int(self.rng.integers(0, total_rooms - 1))
        if key_room_idx >= goal_room_idx:
            key_room_idx += 1

        # Place locked door at goal room entrance, regular doors elsewhere
        for idx, (dx, dy, _is_above) in enumerate(room_doors):
            if idx == goal_room_idx:
                self._place_obj(dx, dy, Door(color="yellow", is_locked=True))
            else:
                self._place_obj(dx, dy, Door(color="green"))

        # Place goal inside the goal room
        ri = goal_room_idx
        room_col = ri // 2
        is_above = ri % 2 == 0
        room_x_start = (rs + 1) * room_col + 1
        room_x_end = room_x_start + rs
        if is_above:
            room_y_start = 1
            room_y_end = corridor_y - 1
        else:
            room_y_start = corridor_y + 2
            room_y_end = grid_h - 1
        gx = int(self.rng.integers(room_x_start, room_x_end))
        gy = int(self.rng.integers(room_y_start, room_y_end))
        self._place_obj(gx, gy, Goal())

        # Place key inside the key room
        ri = key_room_idx
        room_col = ri // 2
        is_above = ri % 2 == 0
        room_x_start = (rs + 1) * room_col + 1
        room_x_end = room_x_start + rs
        if is_above:
            room_y_start = 1
            room_y_end = corridor_y - 1
        else:
            room_y_start = corridor_y + 2
            room_y_end = grid_h - 1
        kx = int(self.rng.integers(room_x_start, room_x_end))
        ky = int(self.rng.integers(room_y_start, room_y_end))
        self._place_obj(kx, ky, Key(color="yellow"))

        # Agent in corridor
        ax = int(self.rng.integers(1, grid_w - 1))
        self._place_agent(ax, corridor_y, DIR_RIGHT)

    def _task_description(self) -> str:
        return (
            f"A corridor with {2 * self._num_rows} rooms "
            "branching off. One room has the goal behind a "
            "locked yellow door. Find the yellow key in "
            "another room, carry it while using TOGGLE on the locked yellow "
            "door to unlock it, and reach the "
            "goal. Reward = 1 - 0.9 * (steps / max_steps)."
        )


class MiniGridKeyCorridorS3R1Env(_KeyCorridorBase):
    _room_size = 3
    _num_rows = 1

    def env_id(self) -> str:
        return "glyphbench/minigrid-keycorridor-s3r1-v0"


class MiniGridKeyCorridorS3R2Env(_KeyCorridorBase):
    _room_size = 3
    _num_rows = 2

    def env_id(self) -> str:
        return "glyphbench/minigrid-keycorridor-s3r2-v0"


class MiniGridKeyCorridorS3R3Env(_KeyCorridorBase):
    _room_size = 3
    _num_rows = 3

    def env_id(self) -> str:
        return "glyphbench/minigrid-keycorridor-s3r3-v0"


class MiniGridKeyCorridorS4R3Env(_KeyCorridorBase):
    _room_size = 4
    _num_rows = 3

    def env_id(self) -> str:
        return "glyphbench/minigrid-keycorridor-s4r3-v0"


class MiniGridKeyCorridorS5R3Env(_KeyCorridorBase):
    _room_size = 5
    _num_rows = 3

    def env_id(self) -> str:
        return "glyphbench/minigrid-keycorridor-s5r3-v0"


class MiniGridKeyCorridorS6R3Env(_KeyCorridorBase):
    _room_size = 6
    _num_rows = 3

    def env_id(self) -> str:
        return "glyphbench/minigrid-keycorridor-s6r3-v0"
