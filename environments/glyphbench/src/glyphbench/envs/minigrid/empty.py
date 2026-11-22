"""MiniGrid Empty room environments.

Includes fixed-start variants (5x5, 6x6, 8x8, 16x16) and random-start
variants (random-5x5, random-6x6).

Gym IDs:
  glyphbench/minigrid-empty-5x5-v0
  glyphbench/minigrid-empty-6x6-v0
  glyphbench/minigrid-empty-8x8-v0
  glyphbench/minigrid-empty-16x16-v0
  glyphbench/minigrid-empty-random-5x5-v0
  glyphbench/minigrid-empty-random-6x6-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.envs.minigrid.base import DIR_RIGHT, MiniGridBase
from glyphbench.envs.minigrid.objects import Goal

_GOAL_GLYPH = Goal().render_char()


def _sample_distant_start_goal(
    env: MiniGridBase, interior_size: int
) -> tuple[tuple[int, int], tuple[int, int], int]:
    min_dist = max(3, interior_size)
    for _ in range(200):
        start = (
            int(env.rng.integers(1, interior_size + 1)),
            int(env.rng.integers(1, interior_size + 1)),
        )
        goal = (
            int(env.rng.integers(1, interior_size + 1)),
            int(env.rng.integers(1, interior_size + 1)),
        )
        if start == goal:
            continue
        if abs(start[0] - goal[0]) + abs(start[1] - goal[1]) >= min_dist:
            direction = int(env.rng.integers(0, 4))
            return start, goal, direction
    # Defensive fallback for very small/custom sizes.
    return (1, 1), (interior_size, interior_size), DIR_RIGHT


class MiniGridEmpty5x5Env(MiniGridBase):
    """MiniGrid Empty-5x5: navigate a 5x5 room to reach the goal.

    Grid is 7x7 (5x5 interior + wall border). Agent starts at grid position
    (1,1) facing right. Goal is at grid position (5,5).

    Actions: TURN_LEFT, TURN_RIGHT, MOVE_FORWARD, PICKUP, DROP, TOGGLE, DONE
    Only the first three are meaningful; the rest are no-ops.

    Reward: 1 - 0.9 * (step_count / max_steps) on reaching goal, 0 otherwise.
    """

    def __init__(self, max_turns: int = 100) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/minigrid-empty-5x5-v0"

    def _generate_grid(self, seed: int) -> None:
        self._init_grid(7, 7)
        self._place_agent(1, 1, DIR_RIGHT)
        self._place_obj(5, 5, Goal())

    def _task_description(self) -> str:
        return (
            f"Navigate a 5x5 room to reach the goal {_GOAL_GLYPH}. "
            "You earn a reward based on how quickly you reach the goal: "
            f"reward = 1 - 0.9 * (steps_taken / {self.max_turns}). "
            "The faster you reach the goal, the higher your reward."
        )


class MiniGridEmpty6x6Env(MiniGridBase):
    """MiniGrid Empty-6x6: navigate a 6x6 room to reach the goal.

    Grid is 8x8 (6x6 interior + wall border). Agent and goal are sampled
    deterministically from the reset seed with a minimum Manhattan distance.
    """

    def __init__(self, max_turns: int = 100) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/minigrid-empty-6x6-v0"

    def _generate_grid(self, seed: int) -> None:
        self._init_grid(8, 8)
        start, goal, direction = _sample_distant_start_goal(self, 6)
        self._place_agent(start[0], start[1], direction)
        self._place_obj(goal[0], goal[1], Goal())

    def _task_description(self) -> str:
        return (
            f"Navigate a 6x6 room to reach the goal {_GOAL_GLYPH}. "
            "You earn a reward based on how quickly you reach the goal: "
            f"reward = 1 - 0.9 * (steps_taken / {self.max_turns}). "
            "The faster you reach the goal, the higher your reward."
        )


class MiniGridEmpty8x8Env(MiniGridBase):
    """MiniGrid Empty-8x8: navigate an 8x8 room to reach the goal.

    Grid is 10x10 (8x8 interior + wall border). Agent and goal are sampled
    deterministically from the reset seed with a minimum Manhattan distance.
    """

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/minigrid-empty-8x8-v0"

    def _generate_grid(self, seed: int) -> None:
        self._init_grid(10, 10)
        start, goal, direction = _sample_distant_start_goal(self, 8)
        self._place_agent(start[0], start[1], direction)
        self._place_obj(goal[0], goal[1], Goal())

    def _task_description(self) -> str:
        return (
            f"Navigate an 8x8 room to reach the goal {_GOAL_GLYPH}. "
            "You earn a reward based on how quickly you reach the goal: "
            f"reward = 1 - 0.9 * (steps_taken / {self.max_turns}). "
            "The faster you reach the goal, the higher your reward."
        )


class MiniGridEmpty16x16Env(MiniGridBase):
    """MiniGrid Empty-16x16: navigate a 16x16 room to reach the goal.

    Grid is 18x18 (16x16 interior + wall border). Agent and goal are sampled
    deterministically from the reset seed with a minimum Manhattan distance.
    """

    def __init__(self, max_turns: int = 300) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/minigrid-empty-16x16-v0"

    def _generate_grid(self, seed: int) -> None:
        self._init_grid(18, 18)
        start, goal, direction = _sample_distant_start_goal(self, 16)
        self._place_agent(start[0], start[1], direction)
        self._place_obj(goal[0], goal[1], Goal())

    def _task_description(self) -> str:
        return (
            f"Navigate a 16x16 room to reach the goal {_GOAL_GLYPH}. "
            "You earn a reward based on how quickly you reach the goal: "
            f"reward = 1 - 0.9 * (steps_taken / {self.max_turns}). "
            "The faster you reach the goal, the higher your reward."
        )


class MiniGridEmptyRandom5x5Env(MiniGridBase):
    """MiniGrid Empty-Random-5x5: navigate a 5x5 room to reach the goal.

    Grid is 7x7 (5x5 interior + wall border). Agent starts at a random interior
    position. Goal is fixed at (5,5); if the agent spawns on the goal it moves
    to (1,5).
    """

    def __init__(self, max_turns: int = 100) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/minigrid-empty-random-5x5-v0"

    def _generate_grid(self, seed: int) -> None:
        self._init_grid(7, 7)
        ax = int(self.rng.integers(1, 6))
        ay = int(self.rng.integers(1, 6))
        self._place_agent(ax, ay, DIR_RIGHT)
        goal_pos = (1, 5) if (ax, ay) == (5, 5) else (5, 5)
        self._place_obj(goal_pos[0], goal_pos[1], Goal())

    def reset(self, seed: int) -> tuple[str, dict[str, Any]]:
        obs, info = super().reset(seed)
        info["agent_pos"] = self._agent_pos
        return obs, info

    def _task_description(self) -> str:
        return (
            f"Navigate a 5x5 room from a random starting position to the goal {_GOAL_GLYPH}. "
            "You earn a reward based on how quickly you reach the goal: "
            f"reward = 1 - 0.9 * (steps_taken / {self.max_turns}). "
            "The faster you reach the goal, the higher your reward."
        )


class MiniGridEmptyRandom6x6Env(MiniGridBase):
    """MiniGrid Empty-Random-6x6: navigate a 6x6 room to reach the goal.

    Grid is 8x8 (6x6 interior + wall border). Agent starts at a random interior
    position. Goal is fixed at (6,6); if the agent spawns on the goal it moves
    to (1,6).
    """

    def __init__(self, max_turns: int = 100) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/minigrid-empty-random-6x6-v0"

    def _generate_grid(self, seed: int) -> None:
        self._init_grid(8, 8)
        ax = int(self.rng.integers(1, 7))
        ay = int(self.rng.integers(1, 7))
        self._place_agent(ax, ay, DIR_RIGHT)
        goal_pos = (1, 6) if (ax, ay) == (6, 6) else (6, 6)
        self._place_obj(goal_pos[0], goal_pos[1], Goal())

    def reset(self, seed: int) -> tuple[str, dict[str, Any]]:
        obs, info = super().reset(seed)
        info["agent_pos"] = self._agent_pos
        return obs, info

    def _task_description(self) -> str:
        return (
            f"Navigate a 6x6 room from a random starting position to the goal {_GOAL_GLYPH}. "
            "You earn a reward based on how quickly you reach the goal: "
            f"reward = 1 - 0.9 * (steps_taken / {self.max_turns}). "
            "The faster you reach the goal, the higher your reward."
        )
