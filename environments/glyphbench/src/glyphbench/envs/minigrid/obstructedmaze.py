"""MiniGrid ObstructedMaze environments.

Multi-room mazes with locked doors and blocking balls.
Progressively harder variants from 1Dl to Full.
"""

from __future__ import annotations

from glyphbench.envs.minigrid.base import DIR_RIGHT, MiniGridBase
from glyphbench.envs.minigrid.objects import Ball, Box, Door, Goal, Key, Wall


class _ObstructedMazeBase(MiniGridBase):
    _num_rooms_x: int = 2  # rooms in x direction
    _num_rooms_y: int = 1  # rooms in y direction
    _room_size: int = 4  # interior size of each room
    _locked: bool = False
    _blocked: bool = False  # balls block doors
    _hidden_key: bool = False  # balls block keys too

    def _generate_grid(self, seed: int) -> None:
        rs = self._room_size
        nrx = self._num_rooms_x
        nry = self._num_rooms_y
        grid_w = (rs + 1) * nrx + 1
        grid_h = (rs + 1) * nry + 1
        self._init_grid(grid_w, grid_h)

        # Add interior walls
        # Vertical walls between room columns
        for i in range(1, nrx):
            wx = (rs + 1) * i
            for y in range(1, grid_h - 1):
                self._place_obj(wx, y, Wall())

        # Horizontal walls between room rows
        for j in range(1, nry):
            wy = (rs + 1) * j
            for x in range(1, grid_w - 1):
                if self._get_obj(x, wy) is None:
                    self._place_obj(x, wy, Wall())

        # Create doorways between adjacent rooms
        door_positions: list[tuple[int, int]] = []

        # Vertical doors (between columns)
        for i in range(1, nrx):
            wx = (rs + 1) * i
            for j in range(nry):
                room_y_start = (rs + 1) * j + 1
                room_y_end = (rs + 1) * (j + 1)
                dy = int(self.rng.integers(room_y_start, room_y_end))
                self._grid[dy][wx] = None
                door_positions.append((wx, dy))

        # Horizontal doors (between rows)
        for j in range(1, nry):
            wy = (rs + 1) * j
            for i in range(nrx):
                room_x_start = (rs + 1) * i + 1
                room_x_end = (rs + 1) * (i + 1)
                dx = int(self.rng.integers(room_x_start, room_x_end))
                self._grid[wy][dx] = None
                door_positions.append((dx, wy))

        # Place doors (locked or unlocked) and optionally balls blocking them
        key_color = "yellow"
        for idx, (dx, dy) in enumerate(door_positions):
            if self._locked and idx == 0:
                self._place_obj(dx, dy, Door(color=key_color, is_locked=True))
            else:
                self._place_obj(dx, dy, Door(color="green"))

            # Place blocking ball in front of door
            if self._blocked and idx < 2:
                # Find an adjacent interior cell for the ball
                for bdx, bdy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    bx, by = dx + bdx, dy + bdy
                    if (
                        1 <= bx < grid_w - 1
                        and 1 <= by < grid_h - 1
                        and self._get_obj(bx, by) is None
                    ):
                        self._place_obj(bx, by, Ball(color="blue"))
                        break

        # Place key in the first room (if locked)
        key_pos: tuple[int, int] | None = None
        if self._locked:
            room0_x_start = 1
            room0_x_end = rs + 1
            room0_y_start = 1
            room0_y_end = rs + 1 if nry > 1 else grid_h - 1
            while True:
                kx = int(self.rng.integers(room0_x_start, room0_x_end))
                ky = int(self.rng.integers(room0_y_start, room0_y_end))
                if self._get_obj(kx, ky) is None:
                    break
            key_obj = Key(color=key_color)
            if self._hidden_key:
                # C5: wrap the key inside a CLOSED box. The agent must
                # TOGGLE the box to reveal the contained key, then PICKUP.
                box = Box(color="purple", contains=key_obj)
                box.can_pickup = False
                self._place_obj(kx, ky, box)
            else:
                self._place_obj(kx, ky, key_obj)
            key_pos = (kx, ky)

        # Agent in first room
        while True:
            ax = int(self.rng.integers(1, min(rs + 1, grid_w - 1)))
            ay = int(self.rng.integers(1, min(rs + 1, grid_h - 1)))
            if self._get_obj(ax, ay) is None and (key_pos is None or (ax, ay) != key_pos):
                break
        self._place_agent(ax, ay, DIR_RIGHT)

        # Goal in last room
        last_room_x_start = (rs + 1) * (nrx - 1) + 1
        last_room_y_start = (rs + 1) * (nry - 1) + 1 if nry > 1 else 1
        while True:
            gx = int(self.rng.integers(last_room_x_start, grid_w - 1))
            gy = int(self.rng.integers(last_room_y_start, grid_h - 1))
            if self._get_obj(gx, gy) is None:
                break
        self._place_obj(gx, gy, Goal())

    def _task_description(self) -> str:
        parts = [
            "Navigate through a maze of rooms to reach "
            "the goal."
        ]
        if self._locked:
            parts.append(
                "A locked yellow door may block a route -- find and carry the "
                "yellow key, face the locked yellow door, and TOGGLE to "
                "unlock/open it; the key stays in your hand."
            )
        if self._blocked:
            parts.append(
                "Balls can obstruct the path through some doorways -- pick "
                "them up and move them. You can carry one object at a time, "
                "so DROP the key before PICKUPing a blocking ball if needed."
            )
        if self._hidden_key:
            parts.append(
                "The key is inside a closed box -- TOGGLE the box to open it "
                "and reveal the key in that cell. To PICKUP the revealed key, "
                "DROP anything you are holding first."
            )
        parts.append("Reward = 1 - 0.9 * (steps / max_steps).")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------


class MiniGridObstructedMaze1DlEnv(_ObstructedMazeBase):
    _num_rooms_x: int = 2
    _num_rooms_y: int = 1
    _room_size: int = 4
    _locked: bool = True

    def env_id(self) -> str:
        return "glyphbench/minigrid-obstructedmaze-1dl-v0"


class MiniGridObstructedMaze1DlhEnv(_ObstructedMazeBase):
    _num_rooms_x: int = 2
    _num_rooms_y: int = 1
    _room_size: int = 4
    _locked: bool = True
    _blocked: bool = True

    def env_id(self) -> str:
        return "glyphbench/minigrid-obstructedmaze-1dlh-v0"


class MiniGridObstructedMaze1DlhbEnv(_ObstructedMazeBase):
    _num_rooms_x: int = 2
    _num_rooms_y: int = 1
    _room_size: int = 4
    _locked: bool = True
    _blocked: bool = True
    _hidden_key: bool = True

    def env_id(self) -> str:
        return "glyphbench/minigrid-obstructedmaze-1dlhb-v0"


class MiniGridObstructedMaze2DlEnv(_ObstructedMazeBase):
    _num_rooms_x: int = 3
    _num_rooms_y: int = 1
    _room_size: int = 4
    _locked: bool = True

    def env_id(self) -> str:
        return "glyphbench/minigrid-obstructedmaze-2dl-v0"


class MiniGridObstructedMaze2DlhEnv(_ObstructedMazeBase):
    _num_rooms_x: int = 3
    _num_rooms_y: int = 1
    _room_size: int = 4
    _locked: bool = True
    _blocked: bool = True

    def env_id(self) -> str:
        return "glyphbench/minigrid-obstructedmaze-2dlh-v0"


class MiniGridObstructedMaze2DlhbEnv(_ObstructedMazeBase):
    _num_rooms_x: int = 3
    _num_rooms_y: int = 1
    _room_size: int = 4
    _locked: bool = True
    _blocked: bool = True
    _hidden_key: bool = True

    def env_id(self) -> str:
        return "glyphbench/minigrid-obstructedmaze-2dlhb-v0"


class MiniGridObstructedMaze1QEnv(_ObstructedMazeBase):
    _num_rooms_x: int = 2
    _num_rooms_y: int = 2
    _room_size: int = 4
    _locked: bool = True
    _blocked: bool = True

    def env_id(self) -> str:
        return "glyphbench/minigrid-obstructedmaze-1q-v0"


class MiniGridObstructedMaze2QEnv(_ObstructedMazeBase):
    _num_rooms_x: int = 3
    _num_rooms_y: int = 2
    _room_size: int = 4
    _locked: bool = True
    _blocked: bool = True

    def env_id(self) -> str:
        return "glyphbench/minigrid-obstructedmaze-2q-v0"


class MiniGridObstructedMazeFullEnv(_ObstructedMazeBase):
    _num_rooms_x: int = 3
    _num_rooms_y: int = 3
    _room_size: int = 4
    _locked: bool = True
    _blocked: bool = True
    _hidden_key: bool = True

    def env_id(self) -> str:
        return "glyphbench/minigrid-obstructedmaze-full-v0"
