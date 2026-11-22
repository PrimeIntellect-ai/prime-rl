"""MiniGrid FourRooms environment.

19x19 grid divided into 4 quadrants by cross-shaped walls with doorways.
Agent and goal placed randomly in different quadrants.
"""

from __future__ import annotations

from glyphbench.envs.minigrid.base import MiniGridBase
from glyphbench.envs.minigrid.objects import Goal, Wall


class MiniGridFourRoomsEnv(MiniGridBase):
    _SIZE = 19  # 17x17 interior + walls

    def env_id(self) -> str:
        return "glyphbench/minigrid-fourrooms-v0"

    def _generate_grid(self, seed: int) -> None:
        s = self._SIZE
        self._init_grid(s, s)

        mid = s // 2  # = 9

        # Horizontal wall at y=mid
        for x in range(1, s - 1):
            self._place_obj(x, mid, Wall())

        # Vertical wall at x=mid
        for y in range(1, s - 1):
            self._place_obj(mid, y, Wall())

        # Create doorways (gaps) in each wall segment
        # Top half of vertical wall (y from 1 to mid-1)
        dy1 = int(self.rng.integers(1, mid))
        self._grid[dy1][mid] = None

        # Bottom half of vertical wall (y from mid+1 to s-2)
        dy2 = int(self.rng.integers(mid + 1, s - 1))
        self._grid[dy2][mid] = None

        # Left half of horizontal wall (x from 1 to mid-1)
        dx1 = int(self.rng.integers(1, mid))
        self._grid[mid][dx1] = None

        # Right half of horizontal wall (x from mid+1 to s-2)
        dx2 = int(self.rng.integers(mid + 1, s - 1))
        self._grid[mid][dx2] = None

        # Define quadrants (interior cells only)
        quadrants = [
            [(x, y) for x in range(1, mid) for y in range(1, mid)],  # top-left
            [(x, y) for x in range(mid + 1, s - 1) for y in range(1, mid)],  # top-right
            [(x, y) for x in range(1, mid) for y in range(mid + 1, s - 1)],  # bottom-left
            [
                (x, y) for x in range(mid + 1, s - 1) for y in range(mid + 1, s - 1)
            ],  # bottom-right
        ]

        # Pick agent quadrant and goal quadrant (different)
        agent_q = int(self.rng.integers(0, 4))
        goal_q = int(self.rng.integers(0, 3))
        if goal_q >= agent_q:
            goal_q += 1

        # Random position within each quadrant
        agent_cells = quadrants[agent_q]
        goal_cells = quadrants[goal_q]

        agent_idx = int(self.rng.integers(0, len(agent_cells)))
        ax, ay = agent_cells[agent_idx]
        direction = int(self.rng.integers(0, 4))
        self._place_agent(ax, ay, direction)

        goal_idx = int(self.rng.integers(0, len(goal_cells)))
        gx, gy = goal_cells[goal_idx]
        self._place_obj(gx, gy, Goal())

    def _task_description(self) -> str:
        goal = Goal().render_char()
        return (
            "Navigate a 17x17 room divided into four quadrants by walls. "
            "Each wall has one doorway you can pass through. "
            f"Find and reach the goal ({goal}) in another quadrant. "
            "Reward = 1 - 0.9 * (steps / max_steps)."
        )
