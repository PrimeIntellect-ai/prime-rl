"""MiniGrid LavaGap environments.

A wall of lava with one gap. Agent must find and use the gap.
"""

from __future__ import annotations

from glyphbench.envs.minigrid.base import DIR_RIGHT, MiniGridBase
from glyphbench.envs.minigrid.objects import Goal, Lava


class _LavaGapBase(MiniGridBase):
    _room_size: int = 5  # interior size, grid = room_size + 2

    def _generate_grid(self, seed: int) -> None:
        size = self._room_size + 2
        self._init_grid(size, size)

        lava_x = size // 2

        # Lava strip
        for y in range(1, size - 1):
            self._place_obj(lava_x, y, Lava())

        # Gap
        gap_y = int(self.rng.integers(1, size - 1))
        self._grid[gap_y][lava_x] = None

        # Agent on left
        ax = int(self.rng.integers(1, lava_x))
        ay = int(self.rng.integers(1, size - 1))
        self._place_agent(ax, ay, DIR_RIGHT)

        # Goal on right
        gx = int(self.rng.integers(lava_x + 1, size - 1))
        gy = int(self.rng.integers(1, size - 1))
        self._place_obj(gx, gy, Goal())

    def _task_description(self) -> str:
        lava = Lava().render_char()
        goal = Goal().render_char()
        return (
            f"A wall of lava ({lava}) blocks your path in a "
            f"{self._room_size}x{self._room_size} interior room. "
            f"Find the one gap in the lava wall and cross to reach the goal ({goal}). "
            f"Stepping on lava ends the episode with zero reward. "
            f"Reward = 1 - 0.9 * (steps / max_steps)."
        )


class MiniGridLavaGapS5Env(_LavaGapBase):
    _room_size = 5

    def env_id(self) -> str:
        return "glyphbench/minigrid-lavagap-s5-v0"


class MiniGridLavaGapS6Env(_LavaGapBase):
    _room_size = 6

    def env_id(self) -> str:
        return "glyphbench/minigrid-lavagap-s6-v0"


class MiniGridLavaGapS7Env(_LavaGapBase):
    _room_size = 7

    def env_id(self) -> str:
        return "glyphbench/minigrid-lavagap-s7-v0"
