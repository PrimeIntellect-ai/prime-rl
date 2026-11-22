"""MiniGrid Unlock and UnlockPickup environments.

Unlock: Pick up key, unlock the door (terminates on TOGGLE-unlock).
UnlockPickup: Pick up key, unlock door, pick up the box behind it.

Reference Minigrid Unlock terminates on toggling the locked door open;
reference UnlockPickup terminates on picking up the target object.
We follow the reference — the previous implementation placed a Goal ★
inside the right room, which was visible THROUGH the wall and trivialised
the puzzle.
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.observation import GridObservation
from glyphbench.envs.minigrid.base import DIR_RIGHT, MiniGridBase
from glyphbench.envs.minigrid.objects import Box, Door, Key, Wall


class MiniGridUnlockEnv(MiniGridBase):
    """Two rooms separated by a locked door. Pick up key, unlock the door."""

    def env_id(self) -> str:
        return "glyphbench/minigrid-unlock-v0"

    def _generate_grid(self, seed: int) -> None:
        # 11x7 grid: two 4x5-interior rooms separated by a wall with locked door
        self._init_grid(11, 7)
        self._door_pos: tuple[int, int] = (5, 1)

        # Vertical dividing wall at x=5
        for y in range(1, 6):
            self._place_obj(5, y, Wall())

        # Locked door in the dividing wall
        door_y = int(self.rng.integers(1, 6))
        self._grid[door_y][5] = None
        self._place_obj(5, door_y, Door(color="yellow", is_locked=True))
        self._door_pos = (5, door_y)

        # Key in left room
        key_x = int(self.rng.integers(1, 5))
        key_y = int(self.rng.integers(1, 6))
        self._place_obj(key_x, key_y, Key(color="yellow"))

        # Agent in left room, not on key
        while True:
            ax = int(self.rng.integers(1, 5))
            ay = int(self.rng.integers(1, 6))
            if (ax, ay) != (key_x, key_y):
                break
        self._place_agent(ax, ay, DIR_RIGHT)

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super()._step(action)
        if terminated or truncated:
            return obs, reward, terminated, truncated, info
        # Success = locked door is now unlocked AND open.
        dx, dy = self._door_pos
        cell = self._grid[dy][dx]
        if isinstance(cell, Door) and cell.is_open and not cell.is_locked:
            shaped = 1 - 0.9 * (self._turn / self.max_turns)
            return obs, float(shaped), True, False, info
        return obs, reward, terminated, truncated, info

    def _task_description(self) -> str:
        return (
            "Two rooms separated by a locked door. Find the "
            "yellow key, pick it up with PICKUP, face the "
            "locked yellow door and TOGGLE to unlock + open it. "
            "The key stays in your hand. "
            "Reward = 1 - 0.9 * (steps / max_steps) on unlock."
        )


class MiniGridUnlockPickupEnv(MiniGridBase):
    """Two rooms separated by a locked door. Unlock, then pick up the box."""

    def env_id(self) -> str:
        return "glyphbench/minigrid-unlockpickup-v0"

    def _generate_grid(self, seed: int) -> None:
        self._init_grid(11, 7)

        # Vertical dividing wall at x=5
        for y in range(1, 6):
            self._place_obj(5, y, Wall())

        # Locked door in the dividing wall
        door_y = int(self.rng.integers(1, 6))
        self._grid[door_y][5] = None
        self._place_obj(5, door_y, Door(color="yellow", is_locked=True))

        # Key in left room
        key_x = int(self.rng.integers(1, 5))
        key_y = int(self.rng.integers(1, 6))
        self._place_obj(key_x, key_y, Key(color="yellow"))

        # Agent in left room, not on key
        while True:
            ax = int(self.rng.integers(1, 5))
            ay = int(self.rng.integers(1, 6))
            if (ax, ay) != (key_x, key_y):
                break
        self._place_agent(ax, ay, DIR_RIGHT)

        # Box in right room — the target the agent must pick up.
        while True:
            bx = int(self.rng.integers(6, 10))
            by = int(self.rng.integers(1, 6))
            if (bx, by) != (6, door_y):
                break
        self._target_box = Box(color="green")
        self._place_obj(bx, by, self._target_box)

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super()._step(action)
        if terminated or truncated:
            return obs, reward, terminated, truncated, info
        # Success = the green box is in the agent's hand.
        if self._carrying is self._target_box:
            shaped = 1 - 0.9 * (self._turn / self.max_turns)
            return obs, float(shaped), True, False, info
        return obs, reward, terminated, truncated, info

    def _task_description(self) -> str:
        return (
            "Two rooms separated by a locked door. Find the "
            "yellow key and pick it up with PICKUP. While carrying the "
            "matching yellow key, face the locked yellow door and TOGGLE to "
            "unlock + open it; the key stays in your hand. DROP the key into "
            "an empty cell before PICKUPing the green box in the other room. "
            "Reward = 1 - 0.9 * (steps / max_steps) on pickup."
        )
