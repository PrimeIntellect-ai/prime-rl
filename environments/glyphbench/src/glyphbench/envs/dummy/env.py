"""Tiny 3x3 walk-to-goal env used as a test fixture."""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.observation import GridObservation


class DummyEnv(BaseGlyphEnv):
    """3x3 walk-to-goal. Used by later plans as a test fixture for harness,
    runner, and provider infrastructure before real pilot envs exist."""

    action_spec = ActionSpec(
        names=("NORTH", "SOUTH", "EAST", "WEST", "NOOP"),
        descriptions=(
            "move one cell up (y decreases)",
            "move one cell down (y increases)",
            "move one cell right (x increases)",
            "move one cell left (x decreases)",
            "do nothing this turn",
        ),
    )
    noop_action_name = "NOOP"

    _WIDTH = 3
    _HEIGHT = 3
    _GOAL_X = 2
    _GOAL_Y = 2

    def __init__(self, max_turns: int = 20) -> None:
        super().__init__(max_turns=max_turns)
        self._x = 0
        self._y = 0

    def env_id(self) -> str:
        return "glyphbench/__dummy-v0"

    def system_prompt(self) -> str:
        return (
            "You are playing a 3x3 test env. Navigate from the top-left corner to "
            "the goal 'G' in the bottom-right corner. You earn +1 on reaching the "
            "goal and the episode ends. Otherwise 0."
        )

    def _reset(self, seed: int) -> GridObservation:
        self._x = 0
        self._y = 0
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        name = self.action_spec.names[action]
        new_x, new_y = self._x, self._y
        if name == "NORTH":
            new_y -= 1
        elif name == "SOUTH":
            new_y += 1
        elif name == "EAST":
            new_x += 1
        elif name == "WEST":
            new_x -= 1
        # NOOP: do nothing

        # Bounds check: bumping into wall = stay put, no penalty
        if 0 <= new_x < self._WIDTH and 0 <= new_y < self._HEIGHT:
            self._x, self._y = new_x, new_y

        reached = (self._x == self._GOAL_X) and (self._y == self._GOAL_Y)
        reward = 1.0 if reached else 0.0
        terminated = reached
        info: dict[str, Any] = {
            "agent_pos": (self._x, self._y),
            "goal_reached": reached,
        }
        return self._render_current_observation(), reward, terminated, False, info

    def _render_current_observation(self) -> GridObservation:
        rows: list[str] = []
        for row_y in range(self._HEIGHT):
            cells: list[str] = []
            for col_x in range(self._WIDTH):
                if (col_x, row_y) == (self._x, self._y):
                    cells.append("@")
                elif (col_x, row_y) == (self._GOAL_X, self._GOAL_Y):
                    cells.append("G")
                else:
                    cells.append(".")
            rows.append("".join(cells))
        grid_str = "\n".join(rows)
        legend = "@ — you\nG — goal\n. — floor"
        hud = f"Step: {self._turn} / {self.max_turns}"
        return GridObservation(grid=grid_str, legend=legend, hud=hud, message="")
