"""Procgen Maze environment.

Procedurally generated maze using recursive backtracking.
Agent navigates from (1,1) to find cheese at the far corner.

Gym ID: glyphbench/procgen-maze-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation
from glyphbench.envs.procgen.base import ProcgenBase


class MazeEnv(ProcgenBase):
    """Procgen Maze: navigate a procedural maze to reach the cheese."""

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "UP", "DOWN"),
        descriptions=(
            "do nothing",
            "move left",
            "move right",
            "move up",
            "move down",
        ),
    )

    MAZE_W = 21
    MAZE_H = 21
    # Reward shaping (Pattern A): single terminal goal worth +1.0.
    _GOAL_REWARD = 1.0

    def env_id(self) -> str:
        return "glyphbench/procgen-maze-v0"

    def _generate_level(self, seed: int) -> None:
        w, h = self.MAZE_W, self.MAZE_H
        self._init_world(w, h, fill="\u2588")
        self._gen_maze(w, h)
        # Place agent at top-left open cell
        self._agent_x, self._agent_y = 1, 1
        # Place cheese at far corner (bottom-right open cell)
        self._cheese_x, self._cheese_y = w - 2, h - 2
        self._set_cell(self._cheese_x, self._cheese_y, "C")

    def _gen_maze(self, w: int, h: int) -> None:
        """Carve a maze using recursive backtracking. w, h must be odd."""
        self._set_cell(1, 1, "\u00b7")
        stack = [(1, 1)]
        while stack:
            cx, cy = stack[-1]
            neighbors: list[tuple[int, int, int, int]] = []
            for ddx, ddy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = cx + ddx, cy + ddy
                if (
                    1 <= nx < w - 1
                    and 1 <= ny < h - 1
                    and self._world_at(nx, ny) == "\u2588"
                ):
                    neighbors.append((nx, ny, cx + ddx // 2, cy + ddy // 2))
            if neighbors:
                idx = int(self.rng.integers(0, len(neighbors)))
                nx, ny, wx, wy = neighbors[idx]
                self._set_cell(wx, wy, "\u00b7")
                self._set_cell(nx, ny, "\u00b7")
                stack.append((nx, ny))
            else:
                stack.pop()

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        terminated = False
        info: dict[str, Any] = {}

        if action_name == "LEFT":
            self._try_move(-1, 0)
        elif action_name == "RIGHT":
            self._try_move(1, 0)
        elif action_name == "UP":
            self._try_move(0, -1)
        elif action_name == "DOWN":
            self._try_move(0, 1)

        # Check if agent reached cheese
        if self._agent_x == self._cheese_x and self._agent_y == self._cheese_y:
            reward = self._GOAL_REWARD
            terminated = True
            self._message = "You found the cheese!"

        return reward, terminated, info

    def _render_current_observation(self) -> GridObservation:
        # Cheese coordinates were leaked here — that defeats the navigation
        # task. The cheese glyph is visible in the legend; the agent must
        # locate it visually.
        return super()._render_current_observation()

    def _task_description(self) -> str:
        return (
            "Navigate through the maze to reach the cheese (C). "
            "Walls (\u2588) block movement. Reach the cheese for +1 reward."
        )

    def _symbol_meaning(self, ch: str) -> str:
        meanings = {"\u00b7": "empty", "\u2588": "wall", "C": "cheese (goal)"}
        return meanings.get(ch, super()._symbol_meaning(ch))
