"""MiniGrid RedBlueDoors environments.

Two rooms separated by a wall with a red door and a blue door. The agent
must reach the goal in the right-hand room.

M3 — topological constraint: the red door must be toggled before the
blue one. If the blue door is toggled first, the episode ends with -1.
This makes the order constraint a real gate instead of a cheaper route.
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.observation import GridObservation
from glyphbench.envs.minigrid.base import DIR_RIGHT, MiniGridBase
from glyphbench.envs.minigrid.objects import Door, Goal, Wall


class _RedBlueDoorsBase(MiniGridBase):
    _room_size: int = 6  # interior size per room

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__(max_turns=max_turns)
        self._red_toggled: bool = False

    def _generate_grid(self, seed: int) -> None:
        rs = self._room_size
        grid_w = 2 * rs + 3  # two rooms + shared wall + outer walls
        grid_h = rs + 2
        self._init_grid(grid_w, grid_h)

        # Vertical dividing wall
        wall_x = rs + 1
        for y in range(1, grid_h - 1):
            self._place_obj(wall_x, y, Wall())

        # Two distinct door positions in the wall
        pos1 = int(self.rng.integers(1, grid_h - 1))
        pos2 = pos1
        while pos2 == pos1:
            pos2 = int(self.rng.integers(1, grid_h - 1))

        if self.rng.integers(0, 2) == 0:
            red_y, blue_y = pos1, pos2
        else:
            red_y, blue_y = pos2, pos1

        self._grid[red_y][wall_x] = None
        self._place_obj(wall_x, red_y, Door(color="red"))
        self._grid[blue_y][wall_x] = None
        self._place_obj(wall_x, blue_y, Door(color="blue"))

        # Agent in left room
        ax = int(self.rng.integers(1, wall_x))
        ay = int(self.rng.integers(1, grid_h - 1))
        self._place_agent(ax, ay, DIR_RIGHT)

        # Goal in right room
        gx = int(self.rng.integers(wall_x + 1, grid_w - 1))
        gy = int(self.rng.integers(1, grid_h - 1))
        self._place_obj(gx, gy, Goal())

        self._red_toggled = False

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        # Detect a TOGGLE attempt on a door BEFORE the parent step actuates it.
        name = self.action_spec.names[action]
        toggle_target: Door | None = None
        if name == "TOGGLE":
            fx, fy = self._front_pos()
            if 0 <= fx < self._grid_w and 0 <= fy < self._grid_h:
                cell = self._get_obj(fx, fy)
                if isinstance(cell, Door):
                    toggle_target = cell

        # If the agent is about to toggle the BLUE door before the RED door,
        # fail immediately. Letting blue open made the constraint a cheap
        # alternate route rather than a real ordering task.
        if (
            toggle_target is not None
            and toggle_target.color == "blue"
            and not self._red_toggled
        ):
            info: dict[str, Any] = {
                "wrong_door_order": True,
                "agent_pos": self._agent_pos,
            }
            return self._render_current_observation(), -1.0, True, False, info

        obs, reward, terminated, truncated, info = super()._step(action)

        # Track legitimate red-first toggle.
        if (
            toggle_target is not None
            and toggle_target.color == "red"
        ):
            self._red_toggled = True

        return obs, reward, terminated, truncated, info

    def _task_description(self) -> str:
        return (
            "Two rooms separated by a wall with a red door and a blue door. "
            "Toggle the RED door BEFORE the blue one. If you toggle blue "
            "first, the episode ends immediately with -1 reward. After the "
            "red door has been toggled, reach the goal in the right-hand "
            "room; opening the blue door is optional. "
            "Reward = 1 - 0.9 * (steps / max_steps) on goal."
        )


class MiniGridRedBlueDoors6x6Env(_RedBlueDoorsBase):
    _room_size = 6

    def env_id(self) -> str:
        return "glyphbench/minigrid-redbluedoors-6x6-v0"


class MiniGridRedBlueDoors8x8Env(_RedBlueDoorsBase):
    _room_size = 8

    def env_id(self) -> str:
        return "glyphbench/minigrid-redbluedoors-8x8-v0"
