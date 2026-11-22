"""MiniGrid LockedRoom and BlockedUnlockPickup environments.

LockedRoom: Three rooms, goal behind locked door, key in another room.
BlockedUnlockPickup: Door blocked by ball, must clear path then unlock.
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.observation import GridObservation
from glyphbench.envs.minigrid.base import DIR_RIGHT, MiniGridBase
from glyphbench.envs.minigrid.objects import (
    Ball,
    Box,
    Door,
    Goal,
    Key,
    Pad,
    Wall,
)


class MiniGridLockedRoomEnv(MiniGridBase):
    """Three rooms in a row. Goal behind locked door, key in another room."""

    def env_id(self) -> str:
        return "glyphbench/minigrid-lockedroom-v0"

    def _generate_grid(self, seed: int) -> None:
        # 19x7 grid: three rooms of 5-cell interior each
        self._init_grid(19, 7)

        # Vertical walls dividing into 3 rooms
        # Wall at x=6
        for y in range(1, 6):
            self._place_obj(6, y, Wall())
        # Wall at x=12
        for y in range(1, 6):
            self._place_obj(12, y, Wall())

        # Door between left and middle room
        left_door_y = int(self.rng.integers(1, 6))
        self._grid[left_door_y][6] = None
        self._place_obj(6, left_door_y, Door(color="green"))

        # Door between middle and right room
        right_door_y = int(self.rng.integers(1, 6))
        self._grid[right_door_y][12] = None

        # Randomly decide which side has goal (locked) and which has key
        if int(self.rng.integers(0, 2)) == 0:
            # Goal in right room (locked), key in left room
            self._place_obj(12, right_door_y, Door(color="yellow", is_locked=True))
            # Key in left room
            kx = int(self.rng.integers(1, 6))
            ky = int(self.rng.integers(1, 6))
            self._place_obj(kx, ky, Key(color="yellow"))
            # Goal in right room
            gx = int(self.rng.integers(13, 18))
            gy = int(self.rng.integers(1, 6))
            self._place_obj(gx, gy, Goal())
        else:
            # Goal in left room (swap: make left door locked)
            self._grid[left_door_y][6] = None
            self._place_obj(6, left_door_y, Door(color="yellow", is_locked=True))
            self._place_obj(12, right_door_y, Door(color="green"))
            # Key in right room
            kx = int(self.rng.integers(13, 18))
            ky = int(self.rng.integers(1, 6))
            self._place_obj(kx, ky, Key(color="yellow"))
            # Goal in left room
            gx = int(self.rng.integers(1, 6))
            gy = int(self.rng.integers(1, 6))
            while (gx, gy) == (kx, ky):
                gx = int(self.rng.integers(1, 6))
                gy = int(self.rng.integers(1, 6))
            self._place_obj(gx, gy, Goal())

        # Agent in middle room
        ax = int(self.rng.integers(7, 12))
        ay = int(self.rng.integers(1, 6))
        self._place_agent(ax, ay, DIR_RIGHT)

    def _task_description(self) -> str:
        goal = Goal().render_char()
        yellow_key = Key(color="yellow").render_char()
        return (
            f"Three rooms in a row. The goal ({goal}) is behind a locked yellow door "
            f"in one side room. The yellow key ({yellow_key}) is in the other side room "
            "(behind a closed green door; no key needed, use TOGGLE to open it). "
            "Navigate to the key, pick it up, unlock the yellow door, and reach the goal. "
            "Reward = 1 - 0.9 * (steps / max_steps)."
        )


class MiniGridBlockedUnlockPickupEnv(MiniGridBase):
    """K4 twin goals: unlock + pickup + drop-on-pad.

    Two rooms separated by a locked yellow door. A blue ball blocks the
    door from the left side. The agent must:
      1. Pick up the yellow key in the left room and drop the blocking ball
         out of the way (or just walk around once cleared).
      2. Unlock the yellow door with the key.
      3. PICKUP the green box in the right room.
      4. Carry the box to a coloured pad and DROP it directly onto the pad.

    Both halves of the chain are required — there is no free-standing goal
    star. Success fires only when the *target box* is dropped on the pad.
    """

    def env_id(self) -> str:
        return "glyphbench/minigrid-blockedunlockpickup-v0"

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__(max_turns=max_turns)
        self._target_box: Box | None = None
        self._pad_pos: tuple[int, int] = (0, 0)
        self._pad_color: str = "green"

    def _generate_grid(self, seed: int) -> None:
        # 11x7 grid: two rooms
        self._init_grid(11, 7)

        # Vertical wall at x=5
        for y in range(1, 6):
            self._place_obj(5, y, Wall())

        # Locked door (not at edges to leave room for ball)
        door_y = int(self.rng.integers(2, 5))
        self._grid[door_y][5] = None
        self._place_obj(5, door_y, Door(color="yellow", is_locked=True))

        # Ball blocking the door (in front of door on left side)
        self._place_obj(4, door_y, Ball(color="blue"))

        # Key in left room (not on ball position)
        while True:
            kx = int(self.rng.integers(1, 5))
            ky = int(self.rng.integers(1, 6))
            if (kx, ky) != (4, door_y):
                break
        self._place_obj(kx, ky, Key(color="yellow"))

        # Agent in left room (not on key or ball)
        while True:
            ax = int(self.rng.integers(1, 5))
            ay = int(self.rng.integers(1, 6))
            if (ax, ay) != (kx, ky) and (ax, ay) != (4, door_y):
                break
        self._place_agent(ax, ay, DIR_RIGHT)

        # Box in right room (target).
        bx = int(self.rng.integers(6, 10))
        by = int(self.rng.integers(1, 6))
        target_box = Box(color="green")
        self._place_obj(bx, by, target_box)
        self._target_box = target_box

        # Pad in right room (drop target). Distinct cell from box.
        while True:
            px = int(self.rng.integers(6, 10))
            py = int(self.rng.integers(1, 6))
            if (px, py) != (bx, by):
                break
        self._pad_pos = (px, py)
        self._pad_color = "green"
        assert target_box.color == self._pad_color
        self._place_obj(px, py, Pad(color=self._pad_color))
        # No Goal star — twin-goal contract requires box-on-pad only.

    # -- step: handle DROP onto pad -------------------------------------

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        name = self.action_spec.names[action]
        # Twin-goal DROP: if the agent DROPs the target box onto the pad,
        # success fires. We have to handle this BEFORE the parent step,
        # because the parent's DROP rule rejects "front cell occupied",
        # and the pad cell is occupied by the Pad object.
        if name == "DROP" and isinstance(self._carrying, Box):
            fx, fy = self._front_pos()
            if (fx, fy) == self._pad_pos and self._carrying is self._target_box:
                # Place the box ON the pad (replace pad). Mark success.
                self._grid[fy][fx] = self._carrying
                self._carrying = None
                obs = self._render_current_observation()
                return obs, self._reward_on_goal(), True, False, {
                    "twin_goal_success": True,
                    "goal_reached": True,
                }
        obs, reward, terminated, truncated, info = super()._step(action)
        return obs, reward, terminated, truncated, info

    def _task_description(self) -> str:
        pad_glyph = Pad(color="green").render_char()
        return (
            "Two rooms separated by a locked yellow door. A blue ball "
            "blocks the door on the left side. In the left room is also "
            "the yellow key. In the right room are a green box and a green "
            f"drop pad ({pad_glyph}). To win: clear the ball out of the "
            "doorway (PICKUP then DROP elsewhere), pick up the yellow key, "
            "TOGGLE the door open, DROP the key on an empty floor cell, walk "
            "to the green box, PICKUP it, then carry it to the pad and DROP it "
            "directly onto the pad. The "
            "goal fires only when the target box is dropped on the pad. "
            "Reward = 1 - 0.9 * (steps / max_steps)."
        )
